import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
import h5py
import copy
import wandb
import numpy as np
import random
import pandas as pd
import os 
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import Callback
import tracemalloc
import os
import psutil

def rotate_tensor(tensor, angle='0'):
    if angle == '0':
        return tensor
    elif angle == 'pi/2':
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    elif angle == 'pi':
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    elif angle == '-pi/2':
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    else:
        raise ValueError(f"Invalid angle: {angle}")

transform = transforms.Compose([transforms.ToTensor()])
def pad(x,value=0): 
    return torch.nn.functional.pad(x, (12, 12, 7, 7),value=value)

def extract_map_parameters(folder_path):
    """
    Extract map parameters K and CostHab from specified files in a given folder.
    
    Args:
        folder_path (str): Path to the folder containing the map and parameter files
    
    Returns:
        tuple: (K_map, CostHab_map) as torch tensors
    """
    # Find the map file (ending with shp_0.01ha.txt)
    map_file = None
    for filename in os.listdir(folder_path):
        if filename.endswith('shp_0.01ha.txt'):
            map_file = os.path.join(folder_path, filename)
            break
    
    if not map_file:
        raise FileNotFoundError("No map file found ending with shp_0.01ha.txt")
    
    # Find parameter files
    k_file = os.path.join(folder_path, 'parameter_base.txt')
    costhab_file = os.path.join(folder_path, 'transfer_base.txt')
    
    if not (os.path.exists(k_file) and os.path.exists(costhab_file)):
        raise FileNotFoundError("Parameter files not found")
    
    # Read map file (skipping 6 metadata lines)
    with open(map_file, 'r') as f:
        # Skip first 6 lines of metadata
        for _ in range(6):
            f.readline()
        
        # Read map data
        map_data = [line.strip().split() for line in f.readlines()]
        map_array = np.array(map_data, dtype=int)
    
    # Get unique pixel categories from the map
    unique_categories = np.unique(map_array)
    
    # Read K parameters using pandas
    k_df = pd.read_csv(k_file, sep='\t')
    
    # Read CostHab parameters using pandas
    costhab_df = pd.read_csv(costhab_file, sep='\t')
    
    # Create K and CostHab maps
    k_map = np.zeros_like(map_array, dtype=float)
    costhab_map = np.zeros_like(map_array, dtype=float)
    
    # Process K parameters
    k_params = {}
    for cat in unique_categories:
        k_col = f'K{cat}'
        if k_col in k_df.columns:
            k_value = k_df[k_col].values[0]
            k_params[f'K{cat}'] = float(k_value)
        else:
            print(f"Warning: No K column found for category {cat}")
            k_params[f'K{cat}'] = 0.0
    
    # Process CostHab parameters
    costhab_params = {}
    for cat in unique_categories:
        costhab_col = f'CostHab{cat}'
        if costhab_col in costhab_df.columns:
            costhab_value = costhab_df[costhab_col].values[0]
            costhab_params[f'CostHab{cat}'] = float(costhab_value)
        else:
            print(f"Warning: No CostHab column found for category {cat}")
            costhab_params[f'CostHab{cat}'] = 0.0
    
    # Fill maps with corresponding parameters
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            pixel_type = map_array[i, j]
            
            # Get K parameters for this pixel type
            k_map[i, j] = k_params.get(f'K{pixel_type}', 0)
            
            # Get CostHab parameters for this pixel type
            costhab_map[i, j] = costhab_params.get(f'CostHab{pixel_type}', 0)
    
    # Convert to torch tensors
    k_tensor = torch.from_numpy(k_map).float()
    costhab_tensor = torch.from_numpy(costhab_map).float()
    
    return k_tensor, costhab_tensor

def map_to_tensor(input_map):
    """
    Convert a numpy map to a 3-channel tensor with channels:
    1. input_map
    2. row x (pixel value is its row)
    3. column y (pixel value is its column)
    """
    # Get the shape of the input map
    input_map = input_map.numpy()
    # print(input_map.shape)
    height, width = input_map.shape[-2:]
    
    # Create a tensor with the same shape as the input map
    tensor = torch.zeros((3, height, width))
    
    # Channel 1: input_map
    tensor[0, :, :] = torch.from_numpy(input_map)
    
    # Channel 2: row x (pixel value is its row)
    tensor[1, :, :] = torch.from_numpy(np.arange(height)[:, None].repeat(width, axis=1))
    
    # Channel 3: column y (pixel value is its column)
    tensor[2, :, :] = torch.from_numpy(np.arange(width)[None, :].repeat(height, axis=0))
    
    return tensor

def load_dataset(normalize=True, concat_mode='concat', folders=[], reps=range(20), years=range(51), vae=None, prediction_step=1):
    dataset = []
    for folder in folders:
        parameters_path = os.path.join(folder, "Inputs")
        k_map, costhab_map = extract_map_parameters(parameters_path)

        file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        with h5py.File(file_path, 'r') as pop_map_h5:
            for rep in reps:
                for year in years[:-1 * prediction_step]:
                    map_data = torch.flip(transform(pop_map_h5[f"rep_{rep}_year_{year}"][()]), [0, 1])
                    map_data_after_step = torch.flip(transform(pop_map_h5[f"rep_{rep}_year_{year + prediction_step}"][()]), [0, 1])

                    if normalize:
                        k_map = (k_map / (torch.amax(k_map, dim=(-2, -1), keepdim=True) + 1e-8)) 
                        costhab_map = (costhab_map / (torch.amax(costhab_map, dim=(-2, -1), keepdim=True) + 1e-8)) 
                        map_data = (map_data / (torch.amax(map_data) + 1e-8)) 
                        map_data_after_step = (map_data_after_step / (torch.amax(map_data_after_step) + 1e-8)) 

                    if concat_mode == 'concat':
                        cond_pop = pad(map_data)
                        cond_land = torch.cat((pad(k_map[None]), pad(costhab_map[None])), dim=0)
                        pred = pad(map_data_after_step)
                    else:
                        cond_pop = pad(map_data).repeat(3, 1, 1)
                        cond_land = pad(k_map[None]).repeat(3, 1, 1)
                        pred = pad(map_data_after_step).repeat(3, 1, 1)

                    dataset.append((cond_pop, cond_land, pred))
    return dataset
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def start_memory_profiling():
    tracemalloc.start()

def log_memory_snapshot():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
def split_folders(folders, train_split=0.8, valid_split=0.1):
    """
    Splits the folders into train, valid, and test sets based on the provided split ratios.

    Args:
        folders (list): List of folders to split.
        train_split (float, optional): Proportion of folders to use for training. Defaults to 0.8.
        valid_split (float, optional): Proportion of folders to use for validation. Defaults to 0.1.

    Returns:
        tuple: Tuple containing the train, valid, and test folders.
    """
    num_folders = len(folders)
    train_size = int(num_folders * train_split)
    valid_size = int(num_folders * valid_split)

    train_folders = folders[:train_size]
    valid_folders = folders[train_size:train_size + valid_size]
    test_folders = folders[train_size + valid_size:]

    return train_folders, valid_folders, test_folders
def get_datasets(args, folders, reps=range(20), years=range(51), data_type=torch.float32):
    """
    Returns train, valid, and test datasets
    """
    # Split folders into train, valid, and test sets
    train_folders, valid_folders, test_folders = split_folders(folders, args.train_split, args.valid_split)

    # Create datasets
    train_dataset = MyDataset(args, train_folders, reps, years, data_type)
    valid_dataset = MyDataset(args, valid_folders, reps, years, data_type)
    test_dataset = MyDataset(args, test_folders, reps, years, data_type)

#     return train_dataset, valid_dataset, test_dataset

# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, args, folders, reps=range(20), years=range(51), data_type=torch.float32):
#         self.args = args
#         self.folders = folders
#         self.reps = list(reps)
#         self.years = list(years)
#         self.data_type = data_type
#         self.prediction_step = args.prediction_step

#         # Pre-collect all rep-year pairs (with associated static maps)
#         self.all_pairs = []
#         for folder in self.folders:
#             parameters_path = os.path.join(folder, "Inputs")
#             k_map, costhab_map = extract_map_parameters(parameters_path)

#             file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
#             if not os.path.exists(file_path):
#                 continue

#             with h5py.File(file_path, 'r') as pop_map_h5:
#                 for rep in self.reps:
#                     for year in self.years:
#                         key_now = f"rep_{rep}_year_{year}"
#                         key_future = f"rep_{rep}_year_{year + self.prediction_step}"
#                         if key_now in pop_map_h5 and key_future in pop_map_h5:
#                             self.all_pairs.append((folder, k_map, costhab_map, rep, year))

#     def __len__(self):
#         return len(self.all_pairs)

#     def __getitem__(self, idx):
#         folder, k_map, costhab_map, rep, year = self.all_pairs[idx]

#         file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
#         if not os.path.exists(file_path):
#             print(f"File {file_path} does not exist, skipping")
#             return None

#         with h5py.File(file_path, 'r') as pop_map_h5:
#             key_now = f"rep_{rep}_year_{year}"
#             key_future = f"rep_{rep}_year_{year + self.prediction_step}"
#             if key_now not in pop_map_h5 or key_future not in pop_map_h5:
#                 return None

#             # Load maps
#             map_now = torch.flip(transform(pop_map_h5[key_now][()]), [0, 1])
#             map_future = torch.flip(transform(pop_map_h5[key_future][()]), [0, 1])

#         # Normalize if required
#         if self.args.normalize:
#             k_map = k_map / (torch.amax(k_map, dim=(-2, -1), keepdim=True) + 1e-8)
#             costhab_map = costhab_map / (torch.amax(costhab_map, dim=(-2, -1), keepdim=True) + 1e-8)
#             map_now = map_now / (torch.amax(map_now, dim=(-2, -1), keepdim=True) + 1e-8)
#             map_future = map_future / (torch.amax(map_future, dim=(-2, -1), keepdim=True) + 1e-8)

#         # Prepare input tensors
#         cond_pop = pad(map_now).repeat(1, 3, 1, 1)
#         cond_land = torch.cat([
#             torch.full((1, 1024, 1024), 1), 
#             pad(k_map[None]),
#             pad(costhab_map[None])
#         ], dim=0).unsqueeze(0)
#         pred = pad(map_future).repeat(1, 3, 1, 1)

#         return cond_pop, cond_land, pred
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, args, folders, reps=range(20), years=range(10,51),data_type=torch.float32, precompressed_data=True, year_leap=1):
#         super().__init__()
#         self.__args = args
#         self.__folders = folders
#         self.__reps = list(reps)
#         self.__years = list(years)
#         self.__prediction_step = args.prediction_step
#         self.__data_type = data_type
#         self.precompressed_data = precompressed_data
#         self.year_leap = year_leap

#         # print(f"[Rank {rank} Worker {worker_id}] Starting iteration with {len(self.__folders)} folders")

#         # Pre-collect all rep-year pairs (with associated static maps)
#         self.all_pairs = []
#         self.data = []
#         for folder in self.__folders:
#             parameters_path = os.path.join(folder, "Inputs")
#             angles = ['0', 'pi/2', 'pi', '-pi/2']
#             if self.precompressed_data:
#                 latent_tensors = torch.load(os.path.join(parameters_path, "latent_tensors.pt"))
#             else:
#                 k_map, costhab_map = extract_map_parameters(parameters_path)            
            
#             for rep in self.__reps:
#                 angle = random.choice(angles)
#                 start_year = self.__years[0] + rep%self.year_leap
#                 rep_years = list(range(start_year, self.__years[-1] + 1 - self.__prediction_step, self.year_leap))
#                 if self.precompressed_data:
#                     k_map = latent_tensors[f'cond_k_latent_{angle}']
#                     costhab_map = latent_tensors[f'cond_costhab_latent_{angle}']
#                     del latent_tensors
#                 else:
#                     k_map = rotate_tensor(pad(k_map[None]), angle)
#                     costhab_map = rotate_tensor(pad(costhab_map[None]), angle)
#                 if rep%self.year_leap != 0:
#                     if rep%self.year_leap < self.year_leap//2:
#                         rep_years.append(self.__years[-1] - self.__prediction_step)
#                     else: 
#                         rep_years = [self.__years[0]] + rep_years
#                 for year in rep_years:
#                     # print(f"[Rank {rank} Worker {worker_id}] Processing folder: {folder}, rep: {rep}, year: {year}, year prediction step: {year +self.__prediction_step}")
#                     # key_now = f"rep_{rep}_year_{year}"
#                     # key_future = f"rep_{rep}_year_{year + self.__prediction_step}"
#                     # if key_now in pop_map_h5 and key_future in pop_map_h5:
#                     self.all_pairs.append((folder, k_map, costhab_map, rep, year, angle))

#         #     all_pairs_folder = [t for t in self.all_pairs if t[0] == folder]
#         #     if self.precompressed_data: #all files have been prepared, no further preprocessing
#         #         file_path = os.path.join(folder, "Output_Maps", "Population_maps_latent.pt") 
#         #         if not os.path.exists(file_path):
#         #             print(f" File {file_path} does not exist, skipping")
#         #             pass   
#         #         file = torch.load(file_path)
#         #     for _, k_map, costhab_map, rep, year, angle in all_pairs_folder:
#         #         if self.precompressed_data:
#         #             cond_pop = file[f"rep_{rep}_year_{year}_angle_{angle}"]
#         #             pred = file[f"rep_{rep}_year_{year + self.__prediction_step}_angle_{angle}"]
#         #             cond_k = k_map
#         #             cond_costhab = costhab_map
#         #         # print(f"[Rank {rank} Worker {worker_id}] Processing folder: {folder}, rep: {rep}, year: {year}", flush=True)
#         #         else:
#         #             file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
#         #             if not os.path.exists(file_path):
#         #                 print(f"File {file_path} does not exist, skipping")
#         #                 pass
#         #             with h5py.File(file_path, 'r') as pop_map_h5:
#         #                 key_now = f"rep_{rep}_year_{year}"
#         #                 key_future = f"rep_{rep}_year_{year + self.__prediction_step}"
#         #                 if key_now not in pop_map_h5 or key_future not in pop_map_h5:
#         #                     pass

#         #                 # Load maps
#         #                 map_now = torch.flip(transform(pop_map_h5[key_now][()]), [0, 1])
#         #                 map_future = torch.flip(transform(pop_map_h5[key_future][()]), [0, 1])
#         #             # Rotate them like the landscape maps
#         #             map_now = rotate_tensor(pad(map_now), angle)
#         #             map_future = rotate_tensor(pad(map_future), angle)
#         #             # delta_map = map_future - map_now
                
#         #             # Normalize if required
#         #             # if self.__args.normalize:
#         #             #     map_now = map_now / (torch.amax(map_now, dim=(-2, -1), keepdim=True) + 1e-8)
#         #             #     map_future = map_future / (torch.amax(map_future, dim=(-2, -1), keepdim=True) + 1e-8)
#         #                 # delta_map = delta_map / (torch.amax(delta_map, dim=(-2, -1), keepdim=True) + 1e-8)
                    
#         #             map_now_norm_k =  torch.where(k_map!= 0,  map_now/k_map, torch.zeros_like(map_now)).repeat(3, 1, 1).type(self.__data_type)
#         #             map_future_norm_k =  torch.where(k_map != 0,  map_future/k_map, torch.zeros_like(map_future)).repeat(3, 1, 1).type(self.__data_type)
#         #             threshold = 0.01
#         #             cond_pop = torch.where(map_now_norm_k > threshold, torch.ones_like(map_now_norm_k)*threshold, map_now_norm_k)*(1/threshold)
#         #             pred = torch.where(map_future_norm_k > threshold, torch.ones_like(map_future_norm_k)*threshold, map_future_norm_k)*(1/threshold)

#         #             k_map_norm = k_map / (torch.amax(k_map, dim=(-2, -1), keepdim=True) + 1e-8)
#         #             costhab_map_norm = costhab_map / (torch.amax(costhab_map, dim=(-2, -1), keepdim=True) + 1e-8)
#         #             cond_k = k_map_norm.repeat(3, 1, 1).type(self.__data_type)
#         #             cond_costhab = costhab_map_norm.repeat(3, 1, 1).type(self.__data_type)

#         #         # Prepare input tensors
#         #         # cond_pop = pad(map_now).repeat(1, 3, 1, 1)
#         #         # cond_land = torch.cat([
#         #         #     torch.full((1, 1024, 1024), 1), 
#         #         #     pad(k_map[None]),
#         #         #     pad(costhab_map[None])
#         #         # ], dim=0).unsqueeze(0)

#         #         # pred = pad(delta_map).repeat(1, 3, 1, 1)
#         #         # pred = pad(map_future).repeat(1, 3, 1, 1)

#         #         # cond_pop = map_to_tensor(pad(map_now)).type(self.__data_type)
#         #         # print('map now shape:', map_now.shape,flush=True)
#         #         # print('k map shape:', k_map.shape, flush=True)
#         #         # print("Avant la ligne problématique")



#         #         # pred = map_to_tensor(pad(delta_map)).type(self.__data_type)
#         #         # pred = map_to_tensor(pad(map_future)).type(self.__data_type)
                    
#         #         self.data.append({
#         #             'condition_data_pop': cond_pop,
#         #             'prediction_data': pred,
#         #             'condition_data_k': cond_k,
#         #             'condition_data_costhab': cond_costhab,
#         #         })
#         # for i in range(len(self.data)):
#         #     self.data[i]['condition_data_pop']['mu'] = self.data[i]['condition_data_pop']['mu'].share_memory_()
#         #     self.data[i]['condition_data_pop']['log_var'] = self.data[i]['condition_data_pop']['log_var'].share_memory_()
#         #     self.data[i]['prediction_data']['mu'] = self.data[i]['prediction_data']['mu'].share_memory_()
#         #     self.data[i]['prediction_data']['log_var'] = self.data[i]['prediction_data']['log_var'].share_memory_()
#         #     self.data[i]['condition_data_k'] = self.data[i]['condition_data_k'].share_memory_()
#         #     self.data[i]['condition_data_costhab'] = self.data[i]['condition_data_costhab'].share_memory_()
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         folder, k_map, costhab_map, rep, year, angle = self.all_pairs[idx]

#         if self.precompressed_data: #all files have been prepared, no further preprocessing
#             file_path = os.path.join(folder, "Output_Maps", "Population_maps_latent.pt") 
#             if not os.path.exists(file_path):
#                 print(f" File {file_path} does not exist, skipping")
#                 pass   
#             file = torch.load(file_path)
#             cond_pop = file[f"rep_{rep}_year_{year}_angle_{angle}"]
#             pred = file[f"rep_{rep}_year_{year + self.__prediction_step}_angle_{angle}"]
#             cond_k = k_map
#             cond_costhab = costhab_map
#             del file
#         # print(f"[Rank {rank} Worker {worker_id}] Processing folder: {folder}, rep: {rep}, year: {year}", flush=True)
#         else:
#             file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
#             if not os.path.exists(file_path):
#                 print(f"File {file_path} does not exist, skipping")
#                 pass
#             with h5py.File(file_path, 'r') as pop_map_h5:
#                 key_now = f"rep_{rep}_year_{year}"
#                 key_future = f"rep_{rep}_year_{year + self.__prediction_step}"
#                 if key_now not in pop_map_h5 or key_future not in pop_map_h5:
#                     pass

#                 # Load maps
#                 map_now = torch.flip(transform(pop_map_h5[key_now][()]), [0, 1])
#                 map_future = torch.flip(transform(pop_map_h5[key_future][()]), [0, 1])
#                 del pop_map_h5
#             # Rotate them like the landscape maps
#             map_now = rotate_tensor(pad(map_now), angle)
#             map_future = rotate_tensor(pad(map_future), angle)
#             # delta_map = map_future - map_now
        
#             # Normalize if required
#             # if self.__args.normalize:
#             #     map_now = map_now / (torch.amax(map_now, dim=(-2, -1), keepdim=True) + 1e-8)
#             #     map_future = map_future / (torch.amax(map_future, dim=(-2, -1), keepdim=True) + 1e-8)
#                 # delta_map = delta_map / (torch.amax(delta_map, dim=(-2, -1), keepdim=True) + 1e-8)
            
#             map_now_norm_k =  torch.where(k_map!= 0,  map_now/k_map, torch.zeros_like(map_now)).repeat(3, 1, 1).type(self.__data_type)
#             map_future_norm_k =  torch.where(k_map != 0,  map_future/k_map, torch.zeros_like(map_future)).repeat(3, 1, 1).type(self.__data_type)
#             threshold = 0.01
#             cond_pop = torch.where(map_now_norm_k > threshold, torch.ones_like(map_now_norm_k)*threshold, map_now_norm_k)*(1/threshold)
#             pred = torch.where(map_future_norm_k > threshold, torch.ones_like(map_future_norm_k)*threshold, map_future_norm_k)*(1/threshold)

#             k_map_norm = k_map / (torch.amax(k_map, dim=(-2, -1), keepdim=True) + 1e-8)
#             costhab_map_norm = costhab_map / (torch.amax(costhab_map, dim=(-2, -1), keepdim=True) + 1e-8)
#             cond_k = k_map_norm.repeat(3, 1, 1).type(self.__data_type)
#             cond_costhab = costhab_map_norm.repeat(3, 1, 1).type(self.__data_type)

#         # Prepare input tensors
#         # cond_pop = pad(map_now).repeat(1, 3, 1, 1)
#         # cond_land = torch.cat([
#         #     torch.full((1, 1024, 1024), 1), 
#         #     pad(k_map[None]),
#         #     pad(costhab_map[None])
#         # ], dim=0).unsqueeze(0)

#         # pred = pad(delta_map).repeat(1, 3, 1, 1)
#         # pred = pad(map_future).repeat(1, 3, 1, 1)

#         # cond_pop = map_to_tensor(pad(map_now)).type(self.__data_type)
#         # print('map now shape:', map_now.shape,flush=True)
#         # print('k map shape:', k_map.shape, flush=True)
#         # print("Avant la ligne problématique")



#         # pred = map_to_tensor(pad(delta_map)).type(self.__data_type)
#         # pred = map_to_tensor(pad(map_future)).type(self.__data_type)
            
#         return({
#             'condition_data_pop': cond_pop,
#             'prediction_data': pred,
#             'condition_data_k': cond_k,
#             'condition_data_costhab': cond_costhab,
#         })
#     # yield (
#     #     torch.stack(batch_condition_data_pop),
#     #     torch.stack(batch_condition_data_landscape),
#     #     torch.stack(batch_prediction_data),
#         # )    

class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        folders,
        reps=range(20),
        years=range(10, 51),
        data_type=torch.float32,
        year_leap=1,
    ):
        super().__init__()

        self.args = args
        self.folders = folders
        self.reps = list(reps)
        self.years = list(years)
        self.prediction_step = args.prediction_step
        self.data_type = data_type
        self.year_leap = year_leap

        self.angles = ['0', 'pi/2', 'pi', '-pi/2']

        # 🔹 Build lightweight index only
        self.samples = []

        for folder in self.folders:
            for rep in self.reps:
                angle = random.choice(self.angles)

                start_year = self.years[0] + rep % self.year_leap
                rep_years = list(
                    range(
                        start_year,
                        self.years[-1] + 1 - self.prediction_step,
                        self.year_leap,
                    )
                )

                if rep % self.year_leap != 0:
                    if rep % self.year_leap < self.year_leap // 2:
                        rep_years.append(self.years[-1] - self.prediction_step)
                    else:
                        rep_years = [self.years[0]] + rep_years

                for year in rep_years:
                    self.samples.append((folder, rep, year, angle))

        if self.args.data_shuffle:
            random.shuffle(self.samples)

        # 🔹 Per-worker cache (created per process)
        self.population_cache = {}
        self.latent_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_population_file(self, folder):
        file_path = os.path.join(
            folder, "Output_Maps", "Population_maps_latent.pt"
        )

        if file_path not in self.population_cache:
            self.population_cache[file_path] = torch.load(
                file_path, map_location="cpu"
            )

        return self.population_cache[file_path]

    def _load_latent_file(self, folder):
        file_path = os.path.join(folder, "Inputs", "latent_tensors.pt")

        if file_path not in self.latent_cache:
            self.latent_cache[file_path] = torch.load(
                file_path, map_location="cpu"
            )

        return self.latent_cache[file_path]

    def __getitem__(self, idx):
        folder, rep, year, angle = self.samples[idx]
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        # 🔹 Load files lazily (once per worker per file)
        pop_file = self._load_population_file(folder)
        latent_file = self._load_latent_file(folder)

        # 🔹 Extract tensors
        cond_pop = pop_file[f"rep_{rep}_year_{year}_angle_{angle}"]#.type(self.data_type)

        pred = pop_file[
            f"rep_{rep}_year_{year + self.prediction_step}_angle_{angle}"
        ]#.type(self.data_type)

        cond_k = latent_file[f"cond_k_latent_{angle}"]#.type(self.data_type)
        cond_costhab = latent_file[f"cond_costhab_latent_{angle}"]#.type(self.data_type)

        mem_after = process.memory_info().rss
        print(f"Iteration {idx}: Memory increase: {(mem_after - mem_before) / 1024 ** 2:.2f} MB")
        return {
            "condition_data_pop": cond_pop,
            "prediction_data": pred,
            "condition_data_k": cond_k,
            "condition_data_costhab": cond_costhab,
        }

class MyDistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        args,
        folders,
        reps=range(20),
        years=range(10, 51),
        data_type=torch.float32,
        precompressed_data=True,
        year_leap=1,
    ):
        super().__init__()

        self.args = args
        self.folders = folders
        self.reps = list(reps)
        self.years = list(years)
        self.prediction_step = args.prediction_step
        self.data_type = data_type
        self.precompressed_data = precompressed_data
        self.year_leap = year_leap
        self.epoch = 0
        self.angles = ['0', 'pi/2', 'pi', '-pi/2']
        # self.rep_to_angle = {
        #             rep: self.angles[rep % len(self.angles)]
        #             for rep in self.reps
        #         } #Option 1.1: fixed angle per rep for consistency across epochs (deterministic version)
        self.rep_to_angle = {
            rep: random.choice(self.angles)
            for rep in self.reps
        } #Option 1.2: fixed angle per rep for consistency across epochs (random version)
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __iter__(self):

        #DDP partition
        is_ddp = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_ddp else 0
        world_size = torch.distributed.get_world_size() if is_ddp else 1

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        #Random shuffle per epoch (deterministic)
        seed = self.epoch * 1000 + worker_id
        rng = random.Random(seed)
        folders = self.folders.copy()
        rng.shuffle(folders)

        folders = folders[worker_id::num_workers]

        for folder in folders:

            #Load files ONCE per folder per worker
            if self.precompressed_data:
                pop_path = os.path.join(folder, "Output_Maps", "Population_maps_latent.pt")
                latent_path = os.path.join(folder, "Inputs", "latent_tensors.pt")

                if not os.path.exists(pop_path):
                    continue

                pop_file = torch.load(pop_path, map_location="cpu")
                latent_file = torch.load(latent_path, map_location="cpu")
            
            reps = self.reps.copy()
            rng.shuffle(reps)
            for rep in reps:
                angle = self.rep_to_angle[rep] #Option 1: fixed angle per rep for consistency across epochs
                start_year = self.years[0] + rep % self.year_leap
                rep_years = list(
                    range(
                        start_year,
                        self.years[-1] + 1 - self.prediction_step,
                        self.year_leap,
                    )
                )
                if rep%self.year_leap < self.year_leap//2:
                    rep_years.append(self.years[-1] - self.prediction_step)
                else: 
                    rep_years = [self.years[0]] + rep_years

                years = rep_years.copy()
                rng.shuffle(years)

                for year in years:

                    if self.precompressed_data:
                        cond_pop = pop_file[f"rep_{rep}_year_{year}_angle_{angle}"]
                        pred = pop_file[f"rep_{rep}_year_{year + self.prediction_step}_angle_{angle}"]
                        cond_k = latent_file[f"cond_k_latent_{angle}"]
                        cond_costhab = latent_file[f"cond_costhab_latent_{angle}"]

                        yield {
                            "condition_data_pop": cond_pop,
                            "prediction_data": pred,
                            "condition_data_k": cond_k,
                            "condition_data_costhab": cond_costhab,
                        }

                    else:
                        raise NotImplementedError("Non-compressed mode omitted for clarity.")

# class MyDistributedIterableDataset(IterableDataset):
#     def __init__(self, args, folders, reps=range(20), years=range(10,51),data_type=torch.float32, precompressed_data=True, year_leap=1):
#         super().__init__()
#         self.__args = args
#         self.__folders = folders
#         self.__reps = list(reps)
#         self.__years = list(years)
#         self.__prediction_step = args.prediction_step
#         self.__data_type = data_type
#         self.precompressed_data = precompressed_data
#         self.year_leap = year_leap
#     def __iter__(self):
#         """
#         Yields:
#             Tuple[Tensor, Tensor, Tensor]: condition_data_pop, condition_data_landscape, prediction_data
#             Each of shape (batch_size, C, H, W)
#         """
#         # DDP info
#         is_ddp = torch.distributed.is_initialized()
#         rank = torch.distributed.get_rank() if is_ddp else 0
#         world_size = torch.distributed.get_world_size() if is_ddp else 1

#         # Worker info
#         worker_info = torch.utils.data.get_worker_info()
#         worker_id = worker_info.id if worker_info is not None else 0
#         num_workers = worker_info.num_workers if worker_info is not None else 1

#         # Combined ID
#         global_worker_id = rank * num_workers + worker_id
#         global_num_workers = world_size * num_workers

#         # print(f"[Rank {rank} Worker {worker_id}] Starting iteration with {len(self.__folders)} folders")

#         # Pre-collect all rep-year pairs (with associated static maps)
#         all_pairs = []
#         for folder in self.__folders:
#             parameters_path = os.path.join(folder, "Inputs")
#             angles = ['0', 'pi/2', 'pi', '-pi/2']

#             else:
#                 k_map, costhab_map = extract_map_parameters(parameters_path)            
            
#             for rep in self.__reps:
#                 angle = random.choice(angles)
#                 start_year = self.__years[0] + rep%self.year_leap
#                 rep_years = list(range(start_year, self.__years[-1] + 1 - self.__prediction_step, self.year_leap))
#                 if self.precompressed_data:
#                     k_map = latent_tensors[f'cond_k_latent_{angle}']
#                     costhab_map = latent_tensors[f'cond_k_latent_{angle}']
#                 else:
#                     k_map = rotate_tensor(pad(k_map[None]), angle)
#                     costhab_map = rotate_tensor(pad(costhab_map[None]), angle)
#                 if rep%self.year_leap != 0:
#                     if rep%self.year_leap < self.year_leap//2:
#                         rep_years.append(self.__years[-1] - self.__prediction_step)
#                     else: 
#                         rep_years = [self.__years[0]] + rep_years
#                 for year in rep_years:
#                     # print(f"[Rank {rank} Worker {worker_id}] Processing folder: {folder}, rep: {rep}, year: {year}, year prediction step: {year +self.__prediction_step}")
#                     # key_now = f"rep_{rep}_year_{year}"
#                     # key_future = f"rep_{rep}_year_{year + self.__prediction_step}"
#                     # if key_now in pop_map_h5 and key_future in pop_map_h5:
#                     all_pairs.append((folder, rep, year, angle))

#         # Shuffle and split across all workers
#         if self.__args.data_shuffle:
#             random.shuffle(all_pairs)
#         all_pairs = all_pairs[global_worker_id::global_num_workers]

#         # Batching
#         batch_condition_data_pop = []
#         batch_condition_data_k = []
#         batch_condition_data_costhab = []
#         batch_prediction_data = []

#         # Variable initialisation to avoid memory leak
#         current_folder = None
#         file_pop = None
#         file_landscape = None

#         cond_k = None
#         cond_costhab = None
#         cond_pop = None
#         pred = None

#         for folder, rep, year, angle in all_pairs:
#             if folder != current_folder:
#                 if self.precompressed_data: #all files have been prepared, no further preprocessing
#                     #load pre-encoded landscape tensors
#                     parameters_path = os.path.join(folder, "Inputs")
#                     file_landscape = torch.load(os.path.join(parameters_path, "latent_tensors.pt"))
#                     cond_k = file_landscape[f'cond_k_latent_{angle}']
#                     cond_costhab = file_landscape[f'cond_costhab_latent_{angle}']

#                     file_path = os.path.join(folder, "Output_Maps", "Population_maps_latent.pt") 
#                     if not os.path.exists(file_path):
#                         print(f"[Rank {rank} Worker {worker_id}] File {file_path} does not exist, skipping")
#                         continue   
#                     file_pop = torch.load(file_path)
#                     cond_pop = file_pop[f"rep_{rep}_year_{year}_angle_{angle}"]
#                     pred = file_pop[f"rep_{rep}_year_{year + self.__prediction_step}_angle_{angle}"]

#                 # print(f"[Rank {rank} Worker {worker_id}] Processing folder: {folder}, rep: {rep}, year: {year}", flush=True)
#                 else:
#                     file_path = os.path.join(folder, "Output_Maps", "Population_maps.h5")
#                     if not os.path.exists(file_path):
#                         print(f"[Rank {rank} Worker {worker_id}] File {file_path} does not exist, skipping")
#                         continue
#                     with h5py.File(file_path, 'r') as pop_map_h5:
#                         key_now = f"rep_{rep}_year_{year}"
#                         key_future = f"rep_{rep}_year_{year + self.__prediction_step}"
#                         if key_now not in pop_map_h5 or key_future not in pop_map_h5:
#                             continue

#                         # Load maps
#                         map_now = torch.flip(transform(pop_map_h5[key_now][()]), [0, 1])
#                         map_future = torch.flip(transform(pop_map_h5[key_future][()]), [0, 1])
#                     # Rotate them like the landscape maps
#                     map_now = rotate_tensor(pad(map_now), angle)
#                     map_future = rotate_tensor(pad(map_future), angle)
#                     # delta_map = map_future - map_now
                
#                     # Normalize if required
#                     # if self.__args.normalize:
#                     #     map_now = map_now / (torch.amax(map_now, dim=(-2, -1), keepdim=True) + 1e-8)
#                     #     map_future = map_future / (torch.amax(map_future, dim=(-2, -1), keepdim=True) + 1e-8)
#                         # delta_map = delta_map / (torch.amax(delta_map, dim=(-2, -1), keepdim=True) + 1e-8)
                    
#                     map_now_norm_k =  torch.where(k_map!= 0,  map_now/k_map, torch.zeros_like(map_now)).repeat(3, 1, 1).type(self.__data_type)
#                     map_future_norm_k =  torch.where(k_map != 0,  map_future/k_map, torch.zeros_like(map_future)).repeat(3, 1, 1).type(self.__data_type)
#                     threshold = 0.01
#                     cond_pop = torch.where(map_now_norm_k > threshold, torch.ones_like(map_now_norm_k)*threshold, map_now_norm_k)*(1/threshold)
#                     pred = torch.where(map_future_norm_k > threshold, torch.ones_like(map_future_norm_k)*threshold, map_future_norm_k)*(1/threshold)

#                     k_map_norm = k_map / (torch.amax(k_map, dim=(-2, -1), keepdim=True) + 1e-8)
#                     costhab_map_norm = costhab_map / (torch.amax(costhab_map, dim=(-2, -1), keepdim=True) + 1e-8)
#                     cond_k = k_map_norm.repeat(3, 1, 1).type(self.__data_type)
#                     cond_costhab = costhab_map_norm.repeat(3, 1, 1).type(self.__data_type)

#             # Prepare input tensors
#             # cond_pop = pad(map_now).repeat(1, 3, 1, 1)
#             # cond_land = torch.cat([
#             #     torch.full((1, 1024, 1024), 1), 
#             #     pad(k_map[None]),
#             #     pad(costhab_map[None])
#             # ], dim=0).unsqueeze(0)

#             # pred = pad(delta_map).repeat(1, 3, 1, 1)
#             # pred = pad(map_future).repeat(1, 3, 1, 1)

#             # cond_pop = map_to_tensor(pad(map_now)).type(self.__data_type)
#             # print('map now shape:', map_now.shape,flush=True)
#             # print('k map shape:', k_map.shape, flush=True)
#             # print("Avant la ligne problématique")



#             # pred = map_to_tensor(pad(delta_map)).type(self.__data_type)
#             # pred = map_to_tensor(pad(map_future)).type(self.__data_type)
                
#             batch_condition_data_pop.append(cond_pop)
#             batch_condition_data_k.append(cond_k)
#             batch_condition_data_costhab.append(cond_costhab)
#             batch_prediction_data.append(pred)


#             if len(batch_condition_data_pop) == self.__args.batch_size:
#                 batch = {}
#                 if self.precompressed_data:
#                     batch["condition_data_pop"] = {'mu':torch.stack([d['mu'] for d in batch_condition_data_pop]), 'log_var': torch.stack([d['log_var'] for d in batch_condition_data_pop])}
#                     batch["prediction_data"] = {'mu':torch.stack([d['mu'] for d in batch_prediction_data]), 'log_var': torch.stack([d['log_var'] for d in batch_prediction_data])}
#                 else:
#                     batch["condition_data_pop"] = torch.stack(batch_condition_data_pop)
#                     batch["prediction_data"] = torch.stack(batch_prediction_data)
#                 batch["condition_data_k"] = torch.stack(batch_condition_data_k)
#                 batch["condition_data_costhab"] = torch.stack(batch_condition_data_costhab)

#                 yield ( batch )
#                 # yield (
#                 #     torch.stack(batch_condition_data_pop),
#                 #     torch.stack(batch_condition_data_landscape),
#                 #     torch.stack(batch_prediction_data),
#                 # )
#                 batch_condition_data_pop = []
#                 batch_condition_data_k = []
#                 batch_condition_data_costhab = []
#                 batch_prediction_data = []

#         # Final partial batch
#         if batch_condition_data_pop != []:
#             batch = {}
#             if self.precompressed_data:
#                 batch["condition_data_pop"] = {'mu':torch.stack([d['mu'] for d in batch_condition_data_pop]), 'log_var': torch.stack([d['log_var'] for d in batch_condition_data_pop])}
#                 batch["prediction_data"] = {'mu':torch.stack([d['mu'] for d in batch_prediction_data]), 'log_var': torch.stack([d['log_var'] for d in batch_prediction_data])}
#             else:
#                 batch["condition_data_pop"] = torch.stack(batch_condition_data_pop)
#                 batch["prediction_data"] = torch.stack(batch_prediction_data)
#             batch["condition_data_k"] = torch.stack(batch_condition_data_k)
#             batch["condition_data_costhab"] = torch.stack(batch_condition_data_costhab)

#             yield ( batch )
#             # yield (
#             #     torch.stack(batch_condition_data_pop),
#             #     torch.stack(batch_condition_data_landscape),
#             #     torch.stack(batch_prediction_data),
#             # )


def load_data(train_dataset, validation_dataset, test_dataset, file_path, normalize=False, device='cpu', train_ratio=0.8, validation_ratio=0.1):
    train_threshold = int(train_ratio * 20)
    validation_threshold = int((train_ratio + validation_ratio) * 20)
    pop_map_h5 = h5py.File(file_path, 'r')
    for rep in range (20):
        for year in range(51):
            map_data = pop_map_h5["rep_{rep}_year_{year}".format(rep=rep, year=year)][()]
            tensor_map = transform(map_data).float()
            if normalize:
                tensor_map = tensor_map / tensor_map.max()
            if rep < train_threshold:
                train_dataset.append(torch.nn.functional.pad(tensor_map, (12, 12, 7, 7)))
            elif rep < validation_threshold:
                validation_dataset.append(torch.nn.functional.pad(tensor_map, (12, 12, 7, 7)))
            else:
                tensor_map = torch.nn.functional.pad(tensor_map, (12, 12, 7, 7))
                test_dataset.append(tensor_map.repeat(3,1,1))

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args, train_folders, validation_folders, years,iterable_dataset=True, data_type=torch.float32,year_leap=1):
        super().__init__()
        self.args = args
        self.train_folders = train_folders
        self.validation_folders = validation_folders
        self.years = years
        self.data_type = data_type
        self.year_leap = year_leap
        self.iterable_dataset = iterable_dataset

    # def setup(self, stage):
    #     # ... data loading and preparation code ...
    #     # print("DataModule attributes:", self.__dict__)
    def train_dataloader(self):
        # print("DataModule attributes:", self.__dict__)
        if self.iterable_dataset:
            print("Iterable dataset")
            dataset = MyDistributedIterableDataset(
                args=self.args,
                folders=self.train_folders,
                reps=range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
            data_sampler = None
        else: 
            print("Not Iterable dataset")
            dataset = MyDataset(
                args=self.args,
                folders=self.train_folders,
                reps=range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                shuffle=True,
                num_replicas=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size, 
            num_workers=self.args.cpus,
            sampler=data_sampler,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
    def val_dataloader(self):
        if self.iterable_dataset:
            dataset = MyDistributedIterableDataset(
                args=self.args,
                folders=self.validation_folders,
                reps=range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
            data_sampler = None
        else:
            dataset = MyDataset(
                args=self.args,
                folders=self.validation_folders,
                reps=range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                shuffle=True,
                num_replicas=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size, 
            num_workers=self.args.cpus,
            sampler=data_sampler,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )

    def test_dataloader(self):
        if self.iterable_dataset:
            dataset = MyDistributedIterableDataset(
                args=self.args,
                folders=self.validation_folders,
                reps= range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
        else:
            dataset = MyDataset(
                args=self.args,
                folders=self.validation_folders,
                reps=range(20),
                years=self.years,
                data_type=self.data_type,
                year_leap=self.year_leap
            )
        return DataLoader(
            dataset,
            batch_size=None if self.iterable_dataset else self.args.batch_size,  # Important for IterableDataset
            num_workers=self.args.cpus,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
    
@rank_zero_only
def log_wandb_config(wandb_logger, args):
    wandb_logger.experiment.config.update({
        # 'lr_min': args.lr_min,
        'lr_start': args.lr_start,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_channel': args.num_channel,
        'loss': args.criterion,
        'trainer strategy': args.trainer_strategy,
        'prediction_step': args.prediction_step,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'nb_of_simulation_folders_train': args.nb_of_simulation_folders_train,
        'nb_of_simulation_folders_valid': args.nb_of_simulation_folders_valid,
    })


class EMACallback(Callback):
    def __init__(self, decay=0.9999, save_dir=None, ema_filename="best_ema.ckpt"):
        """
        EMA callback for PyTorch Lightning.

        Args:
            decay (float): EMA decay factor.
            save_dir (str): Directory to save EMA checkpoint.
            ema_filename (str): Filename for the EMA checkpoint.
        """
        super().__init__()
        self.decay = decay
        self.ema_state = None
        self.original_state = None
        self.save_dir = save_dir
        self.ema_filename = ema_filename

    def on_train_start(self, trainer, pl_module):
        # Initialize EMA weights
        self.ema_state = {
            k: v.detach().clone()
            for k, v in pl_module.state_dict().items()
        }

    @torch.no_grad()
    def on_after_backward(self, trainer, pl_module):
        # Update EMA weights after each backward step
        model_state = pl_module.state_dict()
        for k, v in model_state.items():
            if k in self.ema_state and v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                self.ema_state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    # --- Swap in EMA weights for validation / test / predict ---
    def _swap_in_ema(self, pl_module):
        if self.ema_state is None:
            return
        self.original_state = copy.deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_state, strict=False)

    def on_validation_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._swap_in_ema(pl_module)

    # --- Restore original weights after validation / test / predict ---
    def _restore_original(self, pl_module):
        if self.original_state is not None:
            pl_module.load_state_dict(self.original_state, strict=False)

    def on_validation_end(self, trainer, pl_module):
        self._restore_original(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_original(pl_module)

    def on_predict_end(self, trainer, pl_module):
        self._restore_original(pl_module)

    # --- Save EMA weights along with Lightning checkpoint ---
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state"] = self.ema_state
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save({"state_dict": self.ema_state}, os.path.join(self.save_dir, self.ema_filename))
            print(f"[EMACallback] Saved EMA weights to {os.path.join(self.save_dir, self.ema_filename)}")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.ema_state = checkpoint.get("ema_state", None)

