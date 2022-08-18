import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def main():
    sensivity_specifity()
    #roc_auc()
    #precision_recall()

def sensivity_specifity():
    df = pd.read_csv("data/original/tox21_10k_challenge_score.txt", na_values="x")
    X = df.iloc[:, 0]
    columns = df.columns[1:]
    fnrs_mean = []
    fnrs_std = []
    fors_mean = []
    fors_std = []

    for column in columns:
        plt.figure()
        predictions = []
        tpr = []
        tnr = []
        false_omission_rate = []
        i = 0
        file = "{}.smiles".format(column.lower())
        y = df[column].dropna().astype(int)
        X_model = X[y.index]

        X_model, y = convertSmilesToFeatures(X_model, y)
        d_test = xgb.DMatrix(data=X_model, label=y)

        toxic = y == 1
        non_toxic = y == 0

        for model_number in range(0,5):
            bst = xgb.Booster()
            bst.load_model("output/model_{}/{}.model".format(model_number, file))
            probas_ = bst.predict(d_test)
            # Compute ROC curve and area the curve
            predictions.append(probas_)
            tpr.append(sum(probas_[toxic] >= 0.5) * 100/sum(toxic))
            tnr.append(sum(probas_[non_toxic] < 0.5) * 100/sum(non_toxic))
            fn = sum(probas_[toxic] < 0.5)
            tn = sum(probas_[non_toxic] < 0.5)
            false_omission_rate.append(100 * fn/(fn + tn))
            i += 1

        fnrs = np.subtract(100,tpr)
        fnrs_mean.append(np.mean(fnrs))
        fnrs_std.append(np.std(fnrs))
        fors_mean.append(np.mean(false_omission_rate))
        fors_std.append(np.std(false_omission_rate))

        mean_predictions = np.mean(predictions, axis=0)
        bins = np.linspace(0, 1, 10)
        # Show distributions of estimated probabilities for the two classes.
        plt.hist(mean_predictions[y == 1], alpha=0.5, color='red', bins=bins, density=True,
                 label='toxic - TPR: {:.4} $\pm$ {:.2}'.format(np.mean(tpr), np.std(tpr)))
        plt.hist(mean_predictions[y == 0], alpha=0.5, color='green', bins=bins, density=True,
                 label='non_toxic - TNR: {:.4} $\pm$ {:.2}'.format(np.mean(tnr), np.std(tnr)))

        # Show the threshold.
        plt.axvline(0.5, c='black', ls='dashed')

        # Add labels
        plt.title("Normalized distributions for {}".format(column))
        plt.legend(loc="best")
        #plt.savefig("sensitivity_specifity/SSC-{}".format(column))

    plt.figure(figsize=(16,12))
    plt.barh(range(len(columns)), fnrs_mean, xerr=fnrs_std, align='center', alpha=0.5,
             ecolor='black', capsize=10)
    plt.yticks(range(len(columns)), columns)
    plt.title("Chance of misclassifying a toxic molecule as non toxic (FNR) for each assay type")
    plt.ylabel("Assay Type")
    plt.xlabel("Percentage")
    plt.xticks(range(0,76,5))
    plt.rcParams.update({'font.size': 22})
    plt.savefig("FNR_200", dpi = 180)
    #plt.show()

    plt.figure(figsize=(16,12))
    plt.barh(range(len(columns)), fors_mean, xerr=fors_std, align='center', alpha=0.5,
             ecolor='black', capsize=10)
    plt.yticks(range(len(columns)), columns)
    plt.title("False Omission Rate for each assay type")
    plt.ylabel("Assay Type")
    plt.xlabel("Percentage")
    #plt.savefig("FOR")
    #print(np.mean(every_fnr), np.std(every_fnr)) # 35.395224398485794 15.026731342489105


def precision_recall():
    df = pd.read_csv("data/original/tox21_10k_challenge_score.txt", na_values="x")
    X = df.iloc[:, 0]
    mean_recall = np.linspace(0, 1, 1000)
    columns = df.columns[1:]

    for column in columns:
        #plt.subplot(2, 2, (columns.get_loc(column) + 1))
        plt.figure()
        precisions = []
        aucs = []
        i = 0
        file = "{}.smiles".format(column.lower())
        y = df[column].dropna().astype(int)
        chance = len(y[y==1])/len(y[y==0])
        X_model = X[y.index]

        X_model, y = convertSmilesToFeatures(X_model, y)
        d_test = xgb.DMatrix(data=X_model, label=y)

        for model_number in range(0,5):
            bst = xgb.Booster()
            bst.load_model("output/model_{}/{}.model".format(model_number, file))
            probas_ = bst.predict(d_test)
            # Compute ROC curve and area the curve
            precision, recall, thresholds = precision_recall_curve(y, probas_)
            precisions.append(interp(mean_recall,sorted(recall),precision[::-1]))
            roc_auc = auc(recall, precision)
            aucs.append(roc_auc)

            plt.plot(recall, precision, lw=1, alpha=0.3,
                     label='Model %d (AUC = %0.2f)' % (i + 1, roc_auc))

            i += 1
        plt.plot([0, 1], [chance, chance], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_precision = np.mean(precisions, axis=0)
        mean_auc = auc(mean_recall, mean_precision)
        std_auc = np.std(aucs)
        plt.plot(mean_recall, mean_precision, color='b',
                 label=r'Mean (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(precisions, axis=0)
        tprs_upper = np.minimum(mean_precision + std_tpr, 1)
        tprs_lower = np.maximum(mean_precision - std_tpr, 0)
        plt.fill_between(mean_recall, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve for {}'.format(column))
        plt.legend(loc="best")
        plt.savefig("precision_recall/PRC-{}".format(column))
    #plt.show()


def roc_auc():
    df = pd.read_csv("data/original/tox21_10k_challenge_score.txt", na_values="x")
    X = df.iloc[:, 0]
    mean_fpr = np.linspace(0, 1, 1000)
    columns = df.columns[1:]

    for column in columns[:7]:
        #plt.subplot(2, 2, (columns.get_loc(column) + 1))
        #plt.figure()
        tprs = []
        aucs = []
        i = 0
        file = "{}.smiles".format(column.lower())
        y = df[column].dropna().astype(int)
        X_model = X[y.index]

        X_model, y = convertSmilesToFeatures(X_model, y)
        d_test = xgb.DMatrix(data=X_model, label=y)

        for model_number in range(0,5):
            bst = xgb.Booster()
            bst.load_model("output/model_{}/{}.model".format(model_number, file))
            probas_ = bst.predict(d_test)
            tpr, fpr, thresholds = roc_curve(y, probas_)
            tprs.append(interp(mean_fpr,tpr,fpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i += 1

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr,# color='b',
                 label=column +  r' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of NR assays')
    plt.legend(loc="best")
    plt.savefig("roc_auc/ROC-NR")
    #plt.show()


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