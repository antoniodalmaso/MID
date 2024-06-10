from utils import count_unique_splits_cart, count_unique_splits_dl85
from discretizers.MinimumImpurityDiscretizer import MinimumImpurityDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from pydl85 import DL85Classifier
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import time
import sys
import os

rs = 42
np.random.seed(rs)


############################
# --      FUNCTIONS     -- #
############################

# training function to be parallelized
def parallel_train_tree(parameters):
    discretizer, X_train, y_train, X_test, y_test, n_features = parameters

    if n_features > len(discretizer._thresholds):
        print(f"ERROR! The specified number of features ({n_features}) cannot be used. The maximum is {len(discretizer._thresholds)}")
        return

    # applying the discretizer using the specified n. of features
    X_train = discretizer.transform(X_train, n_features)
    X_test = discretizer.transform(X_test, n_features)

    # definition of the classifier
    clf_dl85 = DL85Classifier(
        max_depth = max_depth,
        min_sup = min_sup,
        time_limit = 300 # the maximum training time is 5 minutes
    )
    clf_cart = DecisionTreeClassifier(
        criterion = discretizer._metric, # the same criterion used by the discretizer
        max_depth = max_depth,
        min_samples_leaf = min_sup,
        random_state = rs
    )

    # testing
    clf_dl85.fit(X_train, y_train)
    y_pred_dl85 = clf_dl85.predict(X_test)

    start_time = time.time()
    clf_cart.fit(X_train, y_train)
    runtime_cart = time.time() - start_time
    
    y_pred_cart = clf_cart.predict(X_test)
    
    # return the partial results related to a single tree
    train_accuracies = {
        "dl85": round(clf_dl85.accuracy_, 4),
        "cart": round(accuracy_score(y_train, clf_cart.predict(X_train)), 4)
    }
    test_accuracies = {
        "dl85": round(accuracy_score(y_test, y_pred_dl85), 4),
        "cart": round(accuracy_score(y_test, y_pred_cart), 4)
    }
    runtimes = {
        "dl85": clf_dl85.runtime_,
        "cart": runtime_cart
    }
    n_used_features = {
        "dl85": count_unique_splits_dl85(clf_dl85.tree_),
        "cart": count_unique_splits_cart(clf_cart.tree_)
    }

    return (n_features, train_accuracies, test_accuracies, runtimes, n_used_features)


############################
# --        CODE        -- #
############################

# PARSING THE PARAMETERS
ds_name = sys.argv[1]         # dataset name
max_depth = int(sys.argv[2])  # maximum depth constraint
min_sup = int(sys.argv[3])    # minimum support constraint
fold_idx = int(sys.argv[4])   # idx specifying the fold of the cross-validation

# RETRIEVING THE DATASET
dataset = pd.read_csv(f"continuous/{ds_name}.csv", header=None)
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# COMPUTE THE NUMBERS OF FEATURES TO BE TESTED
n_sub_workers = 9
n_tests_per_split = 45

n_features_values = np.linspace(1, n_tests_per_split, n_tests_per_split).round().astype(int).tolist()

# DEFINING THE PORTION OF THE DATASET ON TOP OF WHICH THE TESTS ARE EXECUTED
n_splits = 10
kf = KFold(n_splits = n_splits, shuffle = True, random_state = rs)

folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]

train_idx, test_idx = folds[fold_idx]

X_train = X[train_idx, :]
y_train = y[train_idx]

X_test = X[test_idx, :]
y_test = y[test_idx]

# PREPROCESSING
entropy_discretizer = MinimumImpurityDiscretizer(metric = "entropy", max_n_features = n_features_values[-1])
gini_discretizer = MinimumImpurityDiscretizer(metric = "gini", max_n_features = n_features_values[-1])

entropy_discretizer.fit(X_train, y_train)
gini_discretizer.fit(X_train, y_train)

# TRAINING AND TESTING THE TREES (parallelized)
if __name__ == '__main__':
    with Pool(n_sub_workers) as p:
        entropy_partial_results = p.map(parallel_train_tree, [(entropy_discretizer, X_train, y_train, X_test, y_test, n_features) for n_features in n_features_values])
        gini_partial_results = p.map(parallel_train_tree, [(gini_discretizer, X_train, y_train, X_test, y_test, n_features) for n_features in n_features_values])

partial_results = {
    "entropy": [entropy_partial_results, entropy_discretizer._thresholds],
    "gini": [gini_partial_results, gini_discretizer._thresholds]
}

# SAVING THE RESULTS IN A PICKLE FILE
if not os.path.exists(f'partial_exp_data/MID/{ds_name}/{max_depth}_{min_sup}'):
    os.makedirs(f'partial_exp_data/MID/{ds_name}/{max_depth}_{min_sup}')

file_name = f"partial_exp_data/MID/{ds_name}/{max_depth}_{min_sup}/fold_{fold_idx}"
with open(file_name, "wb") as f:
    pickle.dump(partial_results, f)