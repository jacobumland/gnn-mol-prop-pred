import pandas as pd
import numpy as np
import os
from ogb.graphproppred import PygGraphPropPredDataset

def create_smiles_files():
    """ Reads SMILES strings from OGB dataset and creates individual .smi files needed for calculating descriptors and fingerprints with PaDEL-descriptor"""
    
    # load SMILES strings for the molecules (same order as the molecule graphs)
    smiles_mapping = pd.read_csv("ogbg-molhiv dataset/ogbg_molhiv/mapping/mol.csv.gz")

    # check if folder already exists    
    try:
        os.mkdir("ogbg-molhiv SMILES")
    except:
        print(f"The folder already exists")
    else:    
        # create .smi files for PaDEL-Descriptor
        for i in range(len(smiles_mapping)):
            graph_smiles = smiles_mapping.smiles[i]

            file = open(f"ogbg-molhiv SMILES/{i}.smi", "w")
            file.write(graph_smiles)
            file.close()


def handle_raw_data(input_path, save=True):
    """Returns raw data

    Args:
        input_path: path to the output from PaDEL-Descriptor)
    """

    try:
        os.mkdir("raw data")
    except:
        print(f"Raw data was already saved, please load it")
        return
    else:    
        # load featurized PaDEL output
        df = pd.read_csv(input_path)

        # load original data
        dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = "ogbg-molhiv dataset/")

        # new first column
        df.insert(0,"mol_index",'')

        # get index from SMILES data name which is based on the original OGB data
        df["mol_index"] = df["Name"].str.extract('(\d+)').astype(int)

        # sort by new index
        df = df.sort_values(df.columns[0])

        # reset pandas index 
        df = df.reset_index(drop=True)

        # check if ordering is the same
        if list(df.index.values) == list(df["mol_index"]):
            
            # drop name column
            df = df.drop(["Name", "mol_index"], axis=1)
            
            # add label column and fill in the labels from the original dataset
            # matching labels from original dataset
            for i in range(len(df)):
                df.loc[i, ["y"]] = int(dataset[i].y)
                        
        # save
        if save == True:
            print("saving file ...")
            df.to_csv("raw data/1D_2D_PubChemFP_SubFP_raw.csv", index=False)
            print(f"saved to 'raw data/1D_2D_PubChemFP_SubFP_raw.csv'")

        return df  

def get_processed_data(input_path, var_thresh=0.05, corr_thresh=0.95, save=True):
    """Applies preprocessing steps

    Args:
        input_path: path to raw data csv (output from handle_raw_data)
        var_thresh: threshold vlaue for feature variance  
        corr_thresh: threshold vlaue for feature correlation 
    """

    try:
        os.mkdir("preprocessed data")
    except:
        print(f"Preprocessed data was already saved, please load it.")
        return
    else:    

        # load raw data
        df = pd.read_csv(input_path)

        # handling missing values
        df = df.dropna(axis=1)

        # check if all missing values were handled
        if df.isnull().sum().sum() != 0:
            print("a problem occurred when handling missing values")
            return
            
        # exclude labels from preprocessing
        df_prepro = df.copy()
        df_prepro = df_prepro.drop(columns=["y"], axis=1)
        
        # remove low varianve features
        df_prepro = df_prepro.loc[:, df_prepro.var() >= var_thresh]

        ### remove heavily correlated features ###
        # create correlation matrix
        corr_matrix = df_prepro.corr().abs() 
        # select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # find features with correlation greater than set threshold
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        # drop features 
        df_prepro = df_prepro.drop(to_drop, axis=1)

        # standardization
        df_prepro = (df_prepro-df_prepro.mean())/df_prepro.std()

        # add back labels
        df_prepro["y"] = df["y"]
        
        print(f"preprocessing resulted in {df_prepro.shape[1]-1} standardized features")

        # if saving is activated
        if save == True:
            print("saving file ...")
            df_prepro.to_csv("preprocessed data/1D_2D_PubChemFP_SubFP_preprocessed.csv", index=False)
            print(f"saved to 'preprocessed data/1D_2D_PubChemFP_SubFP_preprocessed.csv'")

        return df_prepro

def get_split_data(input_path, sep_y = True):
    """Applies same scaffold splitting as done in the original graph OGB datase

    Args: 
        input_path: path to csv file with preprocessed data
    """

    # load processed data
    df = pd.read_csv("../data/preprocessed data/1D_2D_PubChemFP_SubFP_preprocessed.csv")

    # load splitting indices from OGB scaffold splitting
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = "../data/ogbg-molhiv dataset/")
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    if sep_y == False:
        return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    else:
        return  df.iloc[train_idx, df.columns != "y"], df.loc[train_idx, ["y"]],df.iloc[valid_idx, df.columns != "y"], df.loc[valid_idx, ["y"]], df.iloc[test_idx, df.columns != "y"], df.loc[test_idx, ["y"]],