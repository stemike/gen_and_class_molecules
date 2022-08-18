import pandas as pd
from rdkit import Chem

def main():
    molecules = loadCollection("molecules.txt")
    lenAll = len(molecules)
    moleculesSet = set(molecules)
    lenUnique = len(moleculesSet)
    print("All: {}\nUnique: {}\nDiff: {}".format(lenAll, lenUnique, lenAll - lenUnique))
    moleculeSeries = pd.Series(list(moleculesSet))
    validMolecules = moleculeSeries.apply(lambda x: isMoleculeValid(x))
    validMoleculesSeries = moleculeSeries[validMolecules]
    lenValid = len(validMoleculesSeries)
    print("Valid: {}\nDiff to Unique: {}\nDiff to all: {}".format(lenValid, lenUnique - lenValid,lenAll - lenValid))

    with open("validMolecules.txt", 'w') as f:
        f.write(''.join(validMoleculesSeries.tolist()))

def loadCollection(path):
    with open(path, 'r') as f:
        x = f.readlines()
    return x

def isMoleculeValid(molecule):
    return Chem.MolFromSmiles(molecule) != None

if __name__ == '__main__':
    main()