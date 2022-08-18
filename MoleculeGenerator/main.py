import torch, json
import numpy as np
#from rdkit import Chem
from torch.utils.data import DataLoader
from torch import nn

from model import MoleculeGenerator
from smilesDataset import SMILESDataset

def main():
    data_train = SMILESDataset("data/train_set.csv")
    data_val = SMILESDataset("data/validation_set.csv")
    trainModel(data_train, data_val)

def trainModel(data_train, data_val, batch_size = 128):
    """

    :param data_train:
    :param data_val:
    :return:
    """
    loss_list_train = []
    loss_list_val = []
    num_batches_train = int(data_train.len / batch_size)
    num_batches_val = int(data_val.len / batch_size)


    model = MoleculeGenerator(data_train.getAmountOfUniqueChars()).to(getDEVICE())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True,num_workers=4)
    data_val_loader = DataLoader(data_val,batch_size=batch_size,shuffle=True,num_workers=4)

    for epoch in range(1, 11):
        model.train()

        print("Epoch {:03d}".format(epoch))
        print("===========")

        for batch_idx, samples in enumerate(data_train_loader):
            samples, targets = data_train.split(samples)
            samples = samples.to(getDEVICE())
            targets = targets.to(getDEVICE())

            h, c = model.init_hidden(samples.shape[1])
            h = h.to(getDEVICE())
            c = c.to(getDEVICE())

            optimizer.zero_grad()
            pred, _ = model(samples, (h,c))
            loss = loss_func(pred.view(-1,pred.shape[2]), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if batch_idx % 100 == 0:
                print("-Epoch {:03d}: Batch {:04d} of {} with Training Loss of {}".format(epoch, batch_idx, num_batches_train, loss.item()))

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), "{}/{:03d}_{:03d}.pt".format("states", epoch, batch_idx))
                print("Saved state to: {}/{:03d}_{:03d}.pt".format("states", epoch, batch_idx))

            loss_list_train.append(loss.item())

        saveCollection('output/lossListTrain.txt', loss_list_train)

        # after each epoch, evaluate model on validation set
        model.eval()
        with torch.no_grad():
            for batch_idx, samples in enumerate(data_val_loader):
                samples, targets = data_train.split(samples)
                samples = samples.to(getDEVICE())
                targets = targets.to(getDEVICE())

                h, c = model.init_hidden(samples.shape[1])
                h = h.to(getDEVICE())
                c = c.to(getDEVICE())

                pred, _ = model(samples, (h,c))

                loss = loss_func(pred.view(-1, pred.shape[2]), targets.view(-1)).item()

                if batch_idx % 100 == 0:
                    print("-Epoch {:03d}: Validation Batch {:04d} of {} with Validation Loss of {}".format(
                        epoch, batch_idx, num_batches_val, loss.item()))

                loss_list_val.append(loss)

            print("Validation Loss for Epoch {} of: {}".format(epoch + 1,loss))
            saveCollection('output/lossListVal.txt', loss_list_val)


def getDEVICE():
    #CPU or GPU
    DEVICE = torch.device("cpu")
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DEVICE

def saveCollection(path, collection):
    with open(path, 'w') as f:
        json.dump(collection, f)

if __name__ == '__main__':
    main()