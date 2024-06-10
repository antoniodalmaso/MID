# the aim of this program is to aggregate the results relative to all the scripts, for all the experiments already performed

import pickle
import os

base_folder = "partial_exp_data"

for ds_name in os.listdir(base_folder):
    if ds_name != "EQUAL_FREQUENCY":
        for parameters in os.listdir(f"{base_folder}/{ds_name}"):
            results = []
            
            for split_name in os.listdir(f"{base_folder}/{ds_name}/{parameters}/"):
                with open(f"{base_folder}/{ds_name}/{parameters}/{split_name}", 'rb') as fold:
                    results.append(pickle.load(fold))
            
            # save aggregated data related to a single experiment
            if not os.path.exists(f'exp_data/{ds_name}'):
                os.makedirs(f'exp_data/{ds_name}')

            file_name = f"exp_data/{ds_name}/{ds_name}_{parameters}_impurity"
            with open(file_name, "wb") as f:
                pickle.dump(results, f)