from dataclasses import dataclass, field
from typing import Optional, Union, List

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

import pandas as pd
import numpy as np
import torch
import scipy.io
from scipy.stats import zscore
from sklearn.manifold import TSNE
from hsi_label import *
from synthetic import *
from hsi_full import *

class DataNotFoundError(Exception):
    """Raised when the dataset name is not found in the available datasets."""
    pass

@dataclass
class UserDataset:
  """
  A dataclass to hold user-provided data for analysis.

  Attributes:
    name: The name of the dataset.
    data: The dataset itself, can be provided in various formats.
    labels: Optional labels for the data.
  """
  name: str
  data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None 
  startval:Optional[int]=18 # Default value
  endval:Optional[int]=22 # Defaul value
  labels: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None
  



def my_function(user_data: UserDataset):
  """
  This function processes the user-provided data.

  Args:
    user_data: An instance of the UserDataset class containing the dataset name, data, and optional labels.

  Raises:
    DataNotFoundError: If the provided dataset name is not found in the available datasets.
  """
  if user_data.name in ["SalinasA", "KSL_layer3", "KSL", "Syn_label","Syn_label_shift", "Syn_label_shift_20_error"  ]:
    print(f"Loading pre-defined dataset: {user_data.name}")
    
    if user_data.name=="SalinasA":
        
        SalinasA= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA.mat')['salinasA'])
        SalinasA_corrected= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_corrected.mat')['salinasA_corrected'])
        SalinasA_gt= np.array(scipy.io.loadmat('./HSI_Salinas/SalinasA_gt.mat')['salinasA_gt'])
        
        # Arrange the label in groundtruth
        i=0
        label_data = [0,6,1,5,4,3,2]
        
        for ele in np.unique(SalinasA_gt):
            mask = SalinasA_gt==ele
            SalinasA_gt[mask] = label_data[i]
            i=i+1
        SalinasA_gt = 7 - SalinasA_gt
        
        ######################################################################
        ## Arrange Data as concatationation of spacial co-ordinate and pixel values
        ###########################################################################
        H, W = SalinasA_gt.shape # get the shape of groud truth
        n_features = SalinasA_corrected.shape[2]+4 # get the number of features including co-ordinates and label
        
        # Create a dataset which has "X","Y","Z", "Label", and spectral channel information
        data_hsi = torch.zeros((H*W, n_features ))
        for i in range(H):
            for j in range(W):
                data_hsi[i*W+j,0] = j
                data_hsi[i*W +j,2] = - i
                data_hsi[i*W +j,3] = SalinasA_gt[i,j]
                data_hsi[i*W +j,4:] = torch.tensor(SalinasA_corrected[i,j,:])
                
        # Create a list of column name
        column_name=["X","Y","Z", "Label"]
        for i in range(SalinasA_corrected.shape[2]):
            column_name.append("feature_"+str(i+1))
            
        # Create a pandas dataframe to store the database
        df_hsi = pd.DataFrame(data_hsi,columns=column_name)
        
        # Create a database by removing the non labelled pixel information 
        df_with_non_labelled_pixel = df_hsi.loc[(df_hsi['Label']!=7)]
        
        # Normalise along the spectral lines 
        df_with_spectral_normalised = df_with_non_labelled_pixel.copy()
        df_with_spectral_normalised.iloc[:, 4:] = df_with_spectral_normalised.iloc[:, 4:].apply(zscore,axis=1)
       
        ###########################################################################
        ## Obtain the preprocessed data
        ###########################################################################
        normalised_data = df_with_spectral_normalised.loc[(df_with_spectral_normalised["X"]>=user_data.startval)&(df_with_spectral_normalised["X"]<=user_data.endval)]
        y_obs_label = torch.tensor(normalised_data.iloc[:,3].to_numpy())
        user_data.labels = y_obs_label
        
        normalised_hsi =torch.tensor(normalised_data.drop('Label', axis=1).to_numpy())
        print("Shape of Dataset:", normalised_hsi.shape)
        user_data.data = normalised_hsi
        
    
    elif user_data.name=="KSL" or user_data.name=="KSL_layer3" :
        # Load KSL_file file
        import joblib
        filename_a = './Fw__Hyperspectral_datasets_from_the_KSL_cores/CuSp131.pkl'
        with open(filename_a, 'rb') as myfile:
            a =joblib.load(myfile)
        column_name =[]
        
        for keys, _ in a.items():
            if keys=='XYZ':
                column_name.append("Borehole_id")
                column_name.append("X")
                column_name.append("Y")
                column_name.append("Z")
            else:
                column_name.append(keys+"_R")
                column_name.append(keys+"_G")
                column_name.append(keys+"_B")
        data_a =[]
        for keys, values in a.items():
            if keys=='XYZ':
                label_a = np.ones((235,1))
                data_a.append(label_a)
                data_a.append(values)
            else:
                data_a.append(values)

        # Concatenate the arrays horizontally to create an array of size 235x30
        concatenated_array_a = np.hstack(data_a)
        
        # sort the data based on the depth
        sorted_indices = np.argsort(-concatenated_array_a[:, 3])
        concatenated_array_a = concatenated_array_a[sorted_indices]
        concatenated_array_a.shape
        
        
        dataframe_KSL = pd.DataFrame(concatenated_array_a,columns=column_name)
        
        ######################################################################
        ## Arrange Data as concatationation of spacial co-ordinate and pixel values
        ###########################################################################
        dataframe_KSL = dataframe_KSL[(dataframe_KSL["Z"]<=-700)]
        dataframe_KSL["X"] = 300
        dataframe_KSL["Y"] = 0
        
        df_spectral_normalised = dataframe_KSL.copy()
        df_spectral_normalised.iloc[:,4:] =df_spectral_normalised.iloc[:,4:].apply(zscore,axis=1)
        
        data_hsi = df_spectral_normalised.iloc[:,1:]
        
        # Normalise along the spectral lines 
        df_with_spectral_normalised = data_hsi.copy()
        #df_with_spectral_normalised_ = df_with_spectral_normalised.drop(df_with_spectral_normalised.columns[-10:-5], axis=1)
        
        ###########################################################################
        ## Obtain the preprocessed data
        ###########################################################################
        normalised_data =torch.tensor(df_with_spectral_normalised.to_numpy())
        #normalised_hsi =torch.tensor(df_with_spectral_normalised_.to_numpy(), device =device, dtype =dtype)
        print("Shape of Dataset:", normalised_data.shape)
        user_data.data = normalised_data
        
    elif user_data.name=="Syn_label":
        br1 = -845
        br2 = -825
        z = np.linspace(-900, -700, 250)
        labels = np.ones_like(z)
        labels[z<br1] = 3
        labels[(z>=br1) & (z < br2)] =2
        user_data.labels = torch.tensor(labels, dtype=torch.float64)
        x_loc = 300
        y_loc = 0
        z_loc = z
        xyz_coord = torch.tensor([[x_loc, y_loc, z] for z in z_loc], dtype=torch.float64)
        user_data.data =xyz_coord
    
    
        
    elif user_data.name=="Syn_label_shift":
        br1 = -845 +50
        br2 = -825 +50
        z = np.linspace(-900, -700, 250)
        labels = np.ones_like(z)
        labels[z<br1] = 3
        labels[(z>=br1) & (z < br2)] =2
        user_data.labels = torch.tensor(labels, dtype=torch.float64)
        x_loc = 300
        y_loc = 0
        z_loc = z
        xyz_coord = torch.tensor([[x_loc, y_loc, z] for z in z_loc], dtype=torch.float64)
        user_data.data =xyz_coord
        
    elif user_data.name=="Syn_label_shift_20_error":
        # Define boundaries and labels
        br1 = -845 + 50
        br2 = -825 + 50
        z = np.linspace(-900, -700, 250)
        label = np.ones_like(z)
        label[z < br1] = 3
        label[(z >= br1) & (z < br2)] = 2

        # Randomly change 20% of each label to other labels
        np.random.seed(42)  # For reproducibility
        unique_labels = np.unique(label)  # Get all unique labels
        new_label = label.copy()

        for lbl in unique_labels:
            # Get indices of the current label
            label_indices = np.where(label == lbl)[0]
            # Number of labels to change
            n_to_change = int(0.2 * len(label_indices))
            # Randomly select indices to change
            change_indices = np.random.choice(label_indices, n_to_change, replace=False)
            # Assign new labels randomly (different from the current label)
            for idx in change_indices:
                new_label[idx] = np.random.choice(unique_labels[unique_labels != lbl])
        # Convert to tensor
        user_data.labels = torch.tensor(new_label, dtype=torch.float64)
        x_loc = 300
        y_loc = 0
        z_loc = z
        xyz_coord = torch.tensor([[x_loc, y_loc, z] for z in z_loc], dtype=torch.float64)
        user_data.data =xyz_coord
        
  elif user_data.data is None and user_data.labels is None:
    raise DataNotFoundError(f"Dataset '{user_data.name}' not found and no data provided.")

  print(f"Dataset Name: {user_data.name}")

  if isinstance(user_data.data, pd.DataFrame):
    print("Data provided as pandas DataFrame.")
    user_data.data = torch.tensor(user_data.data.values, dtype=torch.float64)
    if user_data.labels is not None:
      user_data.labels = torch.tensor(user_data.labels.values, dtype=torch.float64)
    
    # Process pandas DataFrame
    # ...

  elif isinstance(user_data.data, np.ndarray):
    print("Data provided as NumPy array.")
    user_data.data = torch.tensor(user_data.data, dtype=torch.float64)
    if user_data.labels is not None:
      user_data.labels = torch.tensor(user_data.labels, dtype=torch.float64)
    # Process NumPy array
    # ...

  elif isinstance(user_data.data, torch.Tensor):
    print("Data provided as PyTorch tensor.")
    # Process PyTorch tensor
    # ...

  # else:
  #   print("Unsupported data type.")

  if user_data.labels is not None:
    print("Labels are provided.")
    # Process labels
    # ...
  else:
      print("Labels are not provided.")

  
def run_files(user_data):
  if user_data.data.shape[1]>3 and user_data.labels is not None:
    print("Salinas type dataset")
  elif user_data.data.shape[1]>3 and user_data.labels is None:
    print("KSL type dataset")
  elif user_data.data.shape[1]==3  and user_data.labels is not None:
    print("synthetic type dataset")
  


# Example Usage:
def setting_dataset():
    # Use a pre-defined dataset
    user_data_1 = UserDataset(name="KSL_layer3")  # No need to provide data explicitly
    my_function(user_data_1)
    # #print(user_data_1.labels)
    # user_data_2 = UserDataset(name="KSL_layer3")  # No need to provide data explicitly
    # my_function(user_data_2)
    
    # # Provide data directly
    # df = pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]})
    # user_data_3 = UserDataset(name="MyDataFrame", data=df)
    # my_function(user_data_3) 

    # Provide data and labels
    # np_array = np.array([[7, 8], [9, 10], [11, 12]])
    # labels_np = np.array([0, 1, 0])
    # user_data_4 = UserDataset(name="MyNumPyArray", data=np_array, labels=labels_np)
    # my_function(user_data_4)

    # labels_np = np.array([0, 1, 0])
    # user_data_5 = UserDataset(name="MyNum_Labels", labels=labels_np)
    # my_function(user_data_5)
    
    run_files(user_data_1)
    # Attempt to use a non-existent dataset without providing data
    # user_data_6 = UserDataset(name="non_existent_dataset") 
    # try:
    #     my_function(user_data_6)
    # except DataNotFoundError as e:
    #     print(f"Error: {e}") 
    return user_data_1
def model_creation(dataset):
    geo_model = None
    return geo_model
    
        
def main():
  dataset= setting_dataset()
  geo_model_init = "Abc"
  geo_model_final= "cds"
  if dataset.data.shape[1]>3 :
    #run_hsi_label(dataset, geo_model_init, geo_model_final=geo_model_final)
    run_hsi_full(dataset, geo_model_init, geo_model_final)
  elif dataset.data.shape[1]==3 and dataset.labels is not None:
    run_synthetic(dataset, geo_model_init, geo_model_final)
    
if __name__ == "__main__":
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()

    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")
