import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import cycle
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve

def main():
    getTestAUC()

def getTestAUC():
    df = pd.read_csv("data/original/tox21_10k_challenge_score.txt", na_values= "x")
    X = df.iloc[:,0]
    auc_list = dict()
    n_classes = df.iloc[:,1:].shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pre = dict()
    rec = dict()
    prc_auc = dict()
    metric_dict = dict()

    #Use Tox21 test set
    for column in df.iloc[:,1:]:
        file = "{}.smiles".format(column.lower())
        y = df[column].dropna().astype(int)
        X_model = X[y.index]

        X_model, y = convertSmilesToFeatures(X_model, y)
        d_test = xgb.DMatrix(data = X_model, label=y)

        bst = xgb.Booster()
        bst.load_model("output/original_model_0/{}.model".format(file))
        y_pred_proba = bst.predict(d_test)
        auc_list[column] = roc_auc_score(y, y_pred_proba)

        i = df.iloc[:,1:].columns.get_loc(column)
        fpr[i], tpr[i], _ = roc_curve(y, y_pred_proba)
        roc_auc[i] = auc(fpr[i], tpr[i])
        metric_dict[column] = (tpr[i], fpr[i], roc_auc[i])

        p, c, _ = precision_recall_curve(y, y_pred_proba)

        pre[i], rec[i], _ = precision_recall_curve(y, y_pred_proba)
        prc_auc[i] = auc(c, p)
        print(auc(c, p))
        #plot_confusion_matrix(y, getLabelFromProb(y_pred_proba), classes=["Non Toxic","Toxic"], normalize=True,
        #                      title='Normalized confusion matrix for {}'.format(column))
        #plt.show()
    #pd.DataFrame.from_dict(metric_dict).to_csv("metrics.csv") # TPR; FPR; AUC
    #return
    #print(auc_list)
    #print(roc_auc)
    plot_multiclass_ROC_prc(n_classes,rec,pre,prc_auc,df.columns)
    #plot_multiclass_ROC(n_classes,fpr,tpr,roc_auc,df.columns)

def plot_multiclass_ROC_prc(n_classes, rec, pre, roc_auc, col_names):
    #SR starts with 7
    index = 7
    # Compute macro-average ROC curve and ROC area
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([rec[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, rec[i], pre[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    rec["macro"] = all_fpr
    pre["macro"] = mean_tpr
    roc_auc["macro"] = auc(rec["macro"], pre["macro"])

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(index,n_classes), colors):
        plt.plot(rec[i], pre[i], lw=lw,
                 label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                       ''.format(col_names[i + 1], roc_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of SR assays')
    plt.legend(loc="upper right")
    plt.show()



def plot_multiclass_ROC(n_classes, fpr, tpr, roc_auc, col_names):
    #SR starts with 7
    index = 7
    # Compute macro-average ROC curve and ROC area
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(index,n_classes), colors):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(col_names[i + 1], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of SR assays')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
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