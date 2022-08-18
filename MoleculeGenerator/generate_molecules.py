import torch, json
import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
from torch import nn

from model import MoleculeGenerator
from smilesDataset import SMILESDataset

def main():
    data_test = SMILESDataset("data/test_set.csv")
    model = MoleculeGenerator(data_test.getAmountOfUniqueChars())
    model.load_state_dict(torch.load("DataForBook/006_000.pt"))

    #sampleMolecules(model,data_test,100000)
    getTestLoss(model, data_test)

def getTestLoss(model, data_test):
    loss_list_test = []
    for epoch in range(1,12):
        model.load_state_dict(torch.load("DataForBook/0{:02}_000.pt".format(epoch)))
        data_test_loader = DataLoader(data_test, batch_size=128, shuffle=True,num_workers=4)
        loss_func = nn.CrossEntropyLoss(ignore_index=0)

        model.eval()
        with torch.no_grad():
            for batch_idx, samples in enumerate(data_test_loader):
                samples, targets = data_test.split(samples)
                samples = samples.to(getDEVICE())
                targets = targets.to(getDEVICE())

                h, c = model.init_hidden(samples.shape[1])
                h = h.to(getDEVICE())
                c = c.to(getDEVICE())

                pred, _ = model(samples, (h, c))
                loss = loss_func(pred.view(-1, pred.shape[2]), targets.view(-1)).item()
                loss_list_test.append(loss)

            print("Test Loss for Epoch {} of: {}".format(epoch + 1, loss))
    saveCollection('output/lossListTest.txt', loss_list_test)


def getDEVICE():
    #CPU or GPU
    DEVICE = torch.device("cpu")
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DEVICE

def saveCollection(path, collection):
    with open(path, 'w') as f:
        json.dump(collection, f)

def sampleMolecules(model, smiles_dataset, count = 150):
    model = model.to(getDEVICE())
    model.eval()

    moleculeCount = 0
    while moleculeCount <= count:
        molecule = ""
        index = 1

        h, c = model.init_hidden(1)
        h = h.to(getDEVICE())
        c = c.to(getDEVICE())
        while index != 2:
            input = smiles_dataset.oneHotEncoding(torch.tensor([index]))[None, :, :].to(torch.device(getDEVICE()))
            output, (h,c) = model(input, (h,c))
            index = getMolecule(output.to("cpu"))
            if index == 2 and len(molecule) == 0:
                continue
            molecule += smiles_dataset.int2char[index]
            #print(index, data_train.int2char[index])
        if isMoleculeValid(molecule[:-1]):
            with open('output/molecules.txt', 'a') as file:
                file.write(molecule)
            moleculeCount += 1
            if  moleculeCount % 10 == 0:
                print("Generated Molecule number {}".format(moleculeCount))
        else:
            print("\tInvalid Molecule: {}".format(molecule[:-1]))
    print("\nFinished Generating Molecules")

def isMoleculeValid(molecule):
    return Chem.MolFromSmiles(molecule) != None

def getMolecule(output):
    """
    Extract molecule from distribution
    :param output: the prediction of a classifier
    :return: an index for a molecule
    """
    probabilities = torch.nn.functional.softmax(output, dim=2).data
    probabilities = np.array(np.squeeze(probabilities))
    index = np.random.choice(np.arange(output.shape[2]), p=probabilities)
    return index

if __name__ == '__main__':
    main()