import numpy as np
import pickle
import os

def print_test_accuracies(ds_name, max_depth, min_sup, print_best = False):
    base_folder = "exp_data/"
    constraints = f"{max_depth}_{min_sup}"

    feature_idx = 2 # test accuracy

    mid_data_path = base_folder + "MID/" + ds_name + "/" + constraints
    mdlp_data_path = base_folder + "MDLP/" + ds_name + "/" + constraints
    equal_frequency_data_path = base_folder + "EQUAL_FREQUENCY/" + ds_name + "/" + constraints
    cart_only_data_path = base_folder + "CART_ONLY/aggregated_data/" + f"{ds_name}_{constraints}_CART_only"

    # MID data
    with open(mid_data_path, 'rb') as f:
        mid_data = pickle.load(f)

    dl85_entropy = np.array(mid_data[0])[:, :-1]
    cart_entropy = np.array(mid_data[1])[:, :-1]
    dl85_gini = np.array(mid_data[2])[:, :-1]
    cart_gini = np.array(mid_data[3])[:, :-1]

    # Specify what to print for the classifiers trained over MID data
    if not print_best:
        # data related to 45 features (without n_features)
        dl85_entropy = dl85_entropy[-1, feature_idx]
        cart_entropy = cart_entropy[-1, feature_idx]
        dl85_gini = dl85_gini[-1, feature_idx]
        cart_gini = cart_gini[-1, feature_idx]
    else:
        dl85_entropy = np.max(dl85_entropy[:, feature_idx])
        cart_entropy = np.max(cart_entropy[:, feature_idx])
        dl85_gini = np.max(dl85_gini[:, feature_idx])
        cart_gini = np.max(cart_gini[:, feature_idx])

    # MDLP data
    with open(mdlp_data_path, 'rb') as f:
        mdlp_data = pickle.load(f)
    mdlp_data = mdlp_data[0][feature_idx]

    # EQUAL FREQUENCY data
    with open(equal_frequency_data_path, 'rb') as f:
        equal_frequency_data = pickle.load(f)
    equal_frequency_data = equal_frequency_data[0][feature_idx]


    # CART only data
    with open(cart_only_data_path, 'rb') as f:
        cart_only_data = pickle.load(f)
    cart_only_entropy = cart_only_data["entropy"][feature_idx]
    cart_only_gini = cart_only_data["gini"][feature_idx]

    print(f"{round(dl85_gini*100, 2)} {round(dl85_entropy*100, 2)} {round(cart_gini*100, 2)} {round(cart_entropy*100, 2)} {round(cart_only_gini*100, 2)} {round(cart_only_entropy*100, 2)} {round(mdlp_data*100, 2)} {round(equal_frequency_data*100, 2)} ")


def print_equal_frequency_train_accuracies(ds_name, max_depth, min_sup, feature_idx):
    base_folder = "exp_data/"
    constraints = f"{max_depth}_{min_sup}"

    mid_data_path = base_folder + "MID/" + ds_name + "/" + constraints
    equal_frequency_data_path = base_folder + "EQUAL_FREQUENCY/" + ds_name + "/" + constraints

    # MID data
    with open(mid_data_path, 'rb') as f:
        mid_data = pickle.load(f)
    mid_max = np.max(np.array(mid_data[0])[:, feature_idx])

    # EQUAL FREQUENCY data
    with open(equal_frequency_data_path, 'rb') as f:
        equal_frequency_data = pickle.load(f)
    n_features = equal_frequency_data[0][0]
    equal_frequency = equal_frequency_data[0][feature_idx]
    mid = equal_frequency_data[1][feature_idx] # 1 for entropy, 2 for gini

    # print line
    output = f"{n_features}"
    output += f" {round(equal_frequency*100, 2)}"
    output += f" {round(mid*100, 2)}"
    # output += f" {round(mid_max*100, 2)}"

    print(output)


def print_mdlp_train_accuracies(ds_name, max_depth, min_sup, feature_idx):
    base_folder = "exp_data/"
    constraints = f"{max_depth}_{min_sup}"

    mid_data_path = base_folder + "MID/" + ds_name + "/" + constraints
    mdlp_data_path = base_folder + "MDLP/" + ds_name + "/" + constraints

    # MID data
    with open(mid_data_path, 'rb') as f:
        mid_data = pickle.load(f)
    mid_max = np.max(np.array(mid_data[0])[:, feature_idx])

    # MDLP data
    with open(mdlp_data_path, 'rb') as f:
        mdlp_data = pickle.load(f)
    n_features = mdlp_data[0][0]
    mdlp = mdlp_data[0][feature_idx]
    mid = mdlp_data[1][feature_idx] # 1 for entropy, 2 for gini

    # print line
    output = f"{n_features}"
    output += f" {round(mdlp*100, 2)}"
    output += f" {round(mid*100, 2)}"
    # output += f" {round(mid_max*100, 2)}"

    print(output)



datasets = [
    "banknote",
    "breast_cancer_cleaned",
    "forest_covtype_cleaned",
    "ionosphere_cleaned",
    "iris_cleaned",
    "letter_recognition",
    "pendigits",
    "penguins_cleaned",
    "pima_indians_diabetes",
    "shuttle",
    "sonar_cleaned",
    "spambase",
    "wine_cleaned",
    "yeast_cleaned"
]

print("TEST ACCURACIES - 45 features")
for ds_name in datasets:
    for max_depth in [3,4,5,6]:
        print_test_accuracies(ds_name, max_depth, 5, print_best = False)

print("\nTEST ACCURACIES - BEST")
for ds_name in datasets:
    for max_depth in [3,4,5,6]:
        print_test_accuracies(ds_name, max_depth, 5, print_best = True)


feature_idx = 1 # 1 for train accuracy, 2 for test accuracy

print("\nTRAIN ACCURACIES - EQUAL FREQUENCY")
for ds_name in datasets:
    print_equal_frequency_train_accuracies(ds_name, 3, 5, feature_idx)

print("\nTRAIN ACCURACIES - MDLP")
for ds_name in datasets:
    print_mdlp_train_accuracies(ds_name, 3, 5, feature_idx)