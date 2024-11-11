"""
Determine which reactions from /data/raw/... are converged and already in the dropbox folder
"""
import os
import pandas as pd

PATH_TO_DROPBOX = "/home/calvin/Dropbox/PersonalFolders/Calvin"
RAW_DATA_PATH = "/home/calvin/Code/HAbstractioNet_Clean/data/raw"
SAVE_PATH = "/home/calvin/Code/HAbstractioNet_Clean/data/processed"


def read_connect_zeus_data(data):

    df = pd.read_csv(data)

    converged_zeus_data_folder = os.path.join(PATH_TO_DROPBOX, "HAb_Converged")

    output_data_frame = df.copy()
    output_data_frame['reaction'] = output_data_frame['reaction'].str.replace('reaction_', 'rxn_')

    list_of_folders = pd.DataFrame(os.listdir(converged_zeus_data_folder), columns=['reaction'])
    
    # Merge the two dataframes via an inner join, on the 'reaction' column
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')

    output_data_frame.to_csv(os.path.join(SAVE_PATH, "converged_zeus_data.csv"), index=False)


def read_connect_atlas_data_rmg(data):
    
    df = pd.read_csv(data)
    
    converged_atlas_data_rmg_folder = os.path.join(PATH_TO_DROPBOX, "ATLAS_Converged")

    output_data_frame = df.copy()

    list_of_folders = pd.DataFrame(os.listdir(converged_atlas_data_rmg_folder), columns=['reaction'])
    
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')
    
    output_data_frame.to_csv(os.path.join(SAVE_PATH, "converged_atlas_data_rmg.csv"), index=False)
    
def read_connect_atlas_data(data):
    
    df = pd.read_csv(data)
    
    converged_atlas_data_folder = os.path.join(PATH_TO_DROPBOX, "ATLAS_Converged", "NonRMG")
    
    output_data_frame = df.copy()

    list_of_folders = pd.DataFrame(os.listdir(converged_atlas_data_folder), columns=['reaction'])
    
    output_data_frame = pd.merge(output_data_frame, list_of_folders, on='reaction', how='inner')
    
    output_data_frame.to_csv(os.path.join(SAVE_PATH, "converged_atlas_data.csv"), index=False)

# Run the functions
read_connect_zeus_data(os.path.join(RAW_DATA_PATH, "zeus_data.csv"))
read_connect_atlas_data_rmg(os.path.join(RAW_DATA_PATH, "atlas_data_rmg.csv"))
read_connect_atlas_data(os.path.join(RAW_DATA_PATH, "atlas_data.csv"))
