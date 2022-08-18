import math
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from sklearn.model_selection import train_test_split


def main():
    train(seed=4, should_load_existing_data=False)

def labelSmiles():
    limit = 100000
    print_header = True

    smiles = pd.read_csv("data/validMolecules.txt", sep='\n', header=None)
    print("Converting Smiles to molecules")

    num_rounds = math.ceil(len(smiles)/limit)
    #Sequentialy label smiles due to memory issues
    for i in range(num_rounds):
        featureRows = []
        morgans_fingerprints = []

        #output_df = None
        molecules_subset = smiles[0][limit * i:limit * (i+1)].reset_index(drop=True)
        output_df = pd.DataFrame(molecules_subset.rename("smiles"))
        molecules = [Chem.MolFromSmiles(x) for x in output_df.iloc[:,0]]
        print("{} out of {} rounds".format(i + 1,num_rounds))
        print("Converting Molecules to Features")
        for molecule in molecules:
            if molecule != None:
                featureRows.append(convertMoleculeToFeatures(molecule))
        X = pd.DataFrame(np.vstack(featureRows))
        print("Finished converting Molecules to MACCS")


        for file in getFileNames():
            bst = xgb.Booster()
            bst.load_model("output/model_0/{}.model".format(file))
            y_pred_proba = bst.predict(xgb.DMatrix(data = X))
            y_pred = getLabelFromProb(y_pred_proba)
            col_name = file.split(".")[0]
            output_df[col_name] = pd.Series(y_pred, dtype='int32').astype(int)
        output_df["toxicity"] = (output_df.iloc[:,1:].sum(axis=1) > 0).astype(int)
        output_df.to_csv("output/labeledMolecules.csv",header=print_header,index=False,mode= 'a')
        print_header = False
    print("===========================")
    print("Finished labeling Molecules")


def train(seed = 0, should_load_existing_data = True):
    params = {
        'eta': 0.01,
        'gamma': 0,
        'max_depth': 10,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bynode': 0.9,
        'tree_method': 'auto',
        'objective': 'binary:logistic',
        'eval_metric': ['auc'],
        'seed': seed
    }
    files = getFileNames()
    score_dict = dict()
    radii = range(7)

    for file in files:
        if not should_load_existing_data:
            X, y = load_data([file])
            print("Loaded data")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

            # Undersampling
            X_train = pd.concat([X_train, y_train], axis=1)
            toxic = X_train[y_train == 1]
            non_toxic = X_train[y_train == 0][:len(toxic)]
            X_train = toxic.append(non_toxic, ignore_index=True).sample(frac=1, random_state=0)
            y_train = X_train.iloc[:, -1]
            X_train = X_train.iloc[:, :-1][0]

        for radius in radii:
            print("==========================")
            print("|Train on ", file)
            print("|Radius ", radius)

            if should_load_existing_data:
                d_train = xgb.DMatrix('data/preprocessed/radius_{}/train_{}_{}.buffer'.format(radius,file,radius))
                d_val = xgb.DMatrix('data/preprocessed/radius_{}/validation_{}_{}.buffer'.format(radius,file,radius))
            else:
                X_train_fp, y_train_fp = convertSmilesToFeatures(X_train, y_train, radius)
                X_val_fp, y_val_fp = convertSmilesToFeatures(X_val, y_val, radius)

                d_train = xgb.DMatrix(data = X_train_fp, label=y_train_fp)
                d_val = xgb.DMatrix(data = X_val_fp, label=y_val_fp)

                d_train.save_binary('data/preprocessed/radius_{}/train_{}_{}.buffer'.format(radius, file, radius))
                d_val.save_binary('data/preprocessed/radius_{}/validation_{}_{}.buffer'.format(radius, file, radius))

            print("Split data into training, validation and test set")

            evallist = [(d_train, 'train'), (d_val, 'eval')]

            # fit model to training data
            bst = xgb.train(params, d_train, evals=evallist, early_stopping_rounds=100,
                            num_boost_round = 2000, verbose_eval = 10000)
            score_dict[(file,radius)] = bst.best_score
            if radius == radii[0]:# or score_dict[(file,radius)] > score_dict[(file,0)]:
                bst.save_model('output/model_{}/{}.model'.format(seed,file))
            print("Saved model")
    print(score_dict)
    for file in files:
        aucList = []
        for radius in radii:
            aucList.append(score_dict[(file, radius)])
        print("Scores for {}: {}".format(file, aucList))


def load_data(files):
    index = 0
    for file in files:
        df = pd.read_csv("data/original/{}".format(file), sep='\t',header=None)
        if index == 0:
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)
        index += 1

    return dataset.iloc[:,0], dataset.iloc[:,2]


def convertSmilesToFeatures(X, y, radius = 0):
    featureRows = []
    indeces = []
    index = 0
    for molecule in [Chem.MolFromSmiles(x) for x in X]:
        if molecule != None:
            featureRows.append(convertMoleculeToFeatures(molecule, radius))
            indeces.append(index)
        index += 1
    X = pd.DataFrame(np.vstack(featureRows))
    y = y.iloc[indeces].reset_index(drop=True)
    return X,y


def getFileNames():
    return ["nr-ahr.smiles","nr-ar.smiles","nr-ar-lbd.smiles","nr-aromatase.smiles","nr-er.smiles","nr-er-lbd.smiles",
            "nr-ppar-gamma.smiles","sr-are.smiles","sr-atad5.smiles","sr-hse.smiles","sr-mmp.smiles","sr-p53.smiles"]


def getLabelFromProb(probabilites):
    return [np.random.choice([0, 1], p=[1 - pred, pred]) for pred in probabilites]


def convertMoleculeToFeatures(molecule, radius = 0):
    if radius == 0:
        featureVector = list(map(int,list(MACCSkeys.GenMACCSKeys(molecule).ToBitString())))
    else:
        featureVector = list(map(int, list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius).ToBitString())))
    featureVector.append(Descriptors.ExactMolWt(molecule))
    featureVector.append(Descriptors.HeavyAtomMolWt(molecule))
    featureVector.append(Descriptors.MolWt(molecule))
    featureVector.append(Descriptors.FpDensityMorgan1(molecule))
    featureVector.append(Descriptors.FpDensityMorgan2(molecule))
    featureVector.append(Descriptors.FpDensityMorgan3(molecule))
    featureVector.append(Descriptors.MaxAbsPartialCharge(molecule, force=False))
    featureVector.append(Descriptors.MaxPartialCharge(molecule, force=False))
    featureVector.append(Descriptors.MinAbsPartialCharge(molecule, force=False))
    featureVector.append(Descriptors.MinPartialCharge(molecule, force=False))
    featureVector.append(Descriptors.NumRadicalElectrons(molecule))
    featureVector.append(Descriptors.NumValenceElectrons(molecule))
    return featureVector


if __name__ == '__main__':
    main()