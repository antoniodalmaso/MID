# the aim of this program is to aggregate the results relative to all the scripts, for all the experiments already performed

import numpy as np
import pickle
import os

base_folder = "partial_exp_data"

# AGGREGATING DATA RELATED TO MID
discretizer = "MID"

for ds_name in os.listdir(base_folder + "/" + discretizer):
    for constraints in os.listdir(base_folder + "/" + discretizer + "/" + ds_name):
        dl85_entropy = {}
        cart_entropy = {}
        dl85_gini = {}
        cart_gini = {}

        for fold in os.listdir(base_folder + "/" + discretizer + "/" + ds_name + "/" + constraints):
            fold_path = base_folder + "/" + discretizer + "/" + ds_name + "/" + constraints + "/" + fold

            with open(fold_path, 'rb') as f:
                fold = pickle.load(f)

            fold_entropy = fold["entropy"][0]
            fold_gini = fold["gini"][0]

            for (n_input_features, train_accuracies, test_accuracies, runtimes, n_used_features) in fold_entropy:
                if n_input_features not in dl85_entropy:
                    dl85_entropy[n_input_features] = {key: [] for key in ["train_accuracy", "test_accuracy", "runtime", "n_used_features"]}
                if n_input_features not in cart_entropy:
                    cart_entropy[n_input_features] = {key: [] for key in ["train_accuracy", "test_accuracy", "runtime", "n_used_features"]}
                
                dl85_entropy[n_input_features]["train_accuracy"].append(train_accuracies["dl85"])
                dl85_entropy[n_input_features]["test_accuracy"].append(test_accuracies["dl85"])
                dl85_entropy[n_input_features]["runtime"].append(runtimes["dl85"])
                dl85_entropy[n_input_features]["n_used_features"].append(n_used_features["dl85"])

                cart_entropy[n_input_features]["train_accuracy"].append(train_accuracies["cart"])
                cart_entropy[n_input_features]["test_accuracy"].append(test_accuracies["cart"])
                cart_entropy[n_input_features]["runtime"].append(runtimes["cart"])
                cart_entropy[n_input_features]["n_used_features"].append(n_used_features["cart"])

            for (n_input_features, train_accuracies, test_accuracies, runtimes, n_used_features) in fold_gini:
                if n_input_features not in dl85_gini:
                    dl85_gini[n_input_features] = {key: [] for key in ["train_accuracy", "test_accuracy", "runtime", "n_used_features"]}
                if n_input_features not in cart_gini:
                    cart_gini[n_input_features] = {key: [] for key in ["train_accuracy", "test_accuracy", "runtime", "n_used_features"]}
                
                dl85_gini[n_input_features]["train_accuracy"].append(train_accuracies["dl85"])
                dl85_gini[n_input_features]["test_accuracy"].append(test_accuracies["dl85"])
                dl85_gini[n_input_features]["runtime"].append(runtimes["dl85"])
                dl85_gini[n_input_features]["n_used_features"].append(n_used_features["dl85"])

                cart_gini[n_input_features]["train_accuracy"].append(train_accuracies["cart"])
                cart_gini[n_input_features]["test_accuracy"].append(test_accuracies["cart"])
                cart_gini[n_input_features]["runtime"].append(runtimes["cart"])
                cart_gini[n_input_features]["n_used_features"].append(n_used_features["cart"])

            # aggregating the results (the average values among the folds are computed)
            aggregated_data = []

            aggregated_data.append([[key] + np.mean(np.array(list(dl85_entropy[key].values())), axis=1).tolist() for key in dl85_entropy])
            aggregated_data.append([[key] + np.mean(np.array(list(cart_entropy[key].values())), axis=1).tolist() for key in cart_entropy])
            aggregated_data.append([[key] + np.mean(np.array(list(dl85_gini[key].values())), axis=1).tolist() for key in dl85_gini])
            aggregated_data.append([[key] + np.mean(np.array(list(cart_gini[key].values())), axis=1).tolist() for key in cart_gini])

            # saving the results in a pickle file
            folder_path = "exp_data/MID/" + ds_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_name = folder_path + "/" + constraints
            with open(file_name, "wb") as f:
                pickle.dump(aggregated_data, f)   



# AGGREGATING DATA RELATED TO OTHER DISCRETIZERS
for discretizer in ["EQUAL_FREQUENCY", "MDLP"]:
    for ds_name in os.listdir(base_folder + "/" + discretizer):
        for constraints in os.listdir(base_folder + "/" + discretizer + "/" + ds_name):
            # the results related to the 10 folds must be aggregated into a single one
            data_discretizer = []
            data_mid_entropy = []
            data_mid_gini = []

            for fold in os.listdir(base_folder + "/" + discretizer + "/" + ds_name + "/" + constraints):
                fold_path = base_folder + "/" + discretizer + "/" + ds_name + "/" + constraints + "/" + fold
                
                # reading the data related to the current fold
                with open(fold_path, 'rb') as f:
                    fold_data = pickle.load(f)
                
                # the results related to the same discretization technique are aggregated together
                fold_data_discretizer, fold_data_mid_entropy, fold_data_mid_gini = fold_data

                data_discretizer.append(fold_data_discretizer)
                data_mid_entropy.append(fold_data_mid_entropy)
                data_mid_gini.append(fold_data_mid_gini)

            # aggregating the results (the average values among the folds are computed)
            aggregated_data = []

            data_discretizer = np.array(data_discretizer)
            data_mid_entropy = np.array(data_mid_entropy)
            data_mid_gini = np.array(data_mid_gini)

            aggregated_data.append(np.mean(data_discretizer[:,:-1], axis=0).tolist() + [np.any(data_discretizer[:,-1])])
            aggregated_data.append(np.mean(data_mid_entropy[:,:-1], axis=0).tolist() + [np.any(data_mid_entropy[:,-1])])
            aggregated_data.append(np.mean(data_mid_gini[:,:-1], axis=0).tolist() + [np.any(data_mid_gini[:,-1])])

            # saving the results in a pickle file
            folder_path = "exp_data/" + discretizer + "/" + ds_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_name = folder_path + "/" + constraints
            with open(file_name, "wb") as f:
                pickle.dump(aggregated_data, f)

print("Done!")