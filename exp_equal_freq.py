from discretizers.MinimumImpurityDiscretizer import MinimumImpurityDiscretizer
from discretizers.EqualFrequencyDiscretizer import EqualFrequencyDiscretizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from pydl85 import DL85Classifier
import pandas as pd
import numpy as np
import pickle
import sys
import os

rs = 42
np.random.seed(rs)


#######################
# - EXPERIMENT CODE - #
#######################

# parsing arguments from the command line
ds_name = sys.argv[1]          # dataset name
exp_idx = int(sys.argv[2])     # between 0 and 39 --> max_depth from 3 to 6 and split_idx from 0 to 9

max_depth = 3 + exp_idx // 10  # maximum depth of the tree to be tested
split_idx = exp_idx % 10       # split of the dataset to use to train the classifier


# retrieving the dataset
dataset = pd.read_csv(f"continuous/{ds_name}.csv", header=None)
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


# cross validation over 10 folds
n_splits = 10
kf = KFold(n_splits = n_splits, shuffle = True, random_state = rs)

kf_indexes = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]


# retrieving the train and test splits of the dataset
train_idx, test_idx = kf_indexes[split_idx]

X_train = X[train_idx, :]
y_train = y[train_idx]

X_test = X[test_idx, :]
y_test = y[test_idx]


# discretizing the dataset using equal frequency
discr_eq_freq = EqualFrequencyDiscretizer(
    n_bins = 8,
    random_state = rs
)
X_train_eq_freq = discr_eq_freq.fit_transform(X_train)
X_test_eq_freq = discr_eq_freq.transform(X_test)

# discretizing the dataset using MID for a comparison
discr_mid_entropy = MinimumImpurityDiscretizer(metric = "entropy")
X_train_mid_entropy = discr_mid_entropy.fit_transform(X_train, y_train, n_features=X_train_eq_freq.shape[1])
X_test_mid_entropy = discr_mid_entropy.transform(X_test, n_features=X_train_eq_freq.shape[1])

discr_mid_gini = MinimumImpurityDiscretizer(metric = "gini")
X_train_mid_gini = discr_mid_gini.fit_transform(X_train, y_train, n_features=X_train_eq_freq.shape[1])
X_test_mid_gini = discr_mid_gini.transform(X_test, n_features=X_train_eq_freq.shape[1])


# training the trees
clf_eq_freq = DL85Classifier(
    max_depth = max_depth,
    min_sup = 5,
    time_limit = 300 # 5 minuti
)
clf_eq_freq.fit(X_train_eq_freq, y_train)

clf_mid_entropy = DL85Classifier(
    max_depth = max_depth,
    min_sup = 5,
    time_limit = 300 # 5 minuti
)
clf_mid_entropy.fit(X_train_mid_entropy, y_train)

clf_mid_gini = DL85Classifier(
    max_depth = max_depth,
    min_sup = 5,
    time_limit = 300 # 5 minuti
)
clf_mid_gini.fit(X_train_mid_gini, y_train)


# testing the trees
y_pred_eq_freq = clf_eq_freq.predict(X_test_eq_freq)
y_pred_mid_entropy = clf_mid_entropy.predict(X_test_mid_entropy)
y_pred_mid_gini = clf_mid_gini.predict(X_test_mid_gini)


# saving the results
results_eq_freq = (
    X_train_eq_freq.shape[1],                 # n. features
    clf_eq_freq.accuracy_,                    # train accuracy
    accuracy_score(y_test, y_pred_eq_freq),   # test accuracy
    clf_eq_freq.runtime_,                     # runtime
    clf_eq_freq.timeout_                      # timeout reached (boolean)
)
results_mid_entropy = (
    X_train_mid_entropy.shape[1],                 # n. features
    clf_mid_entropy.accuracy_,                    # train accuracy
    accuracy_score(y_test, y_pred_mid_entropy),   # test accuracy
    clf_mid_entropy.runtime_,                     # runtime
    clf_mid_entropy.timeout_                      # timeout reached (boolean)
)
results_mid_gini = (
    X_train_mid_gini.shape[1],                 # n. features
    clf_mid_gini.accuracy_,                    # train accuracy
    accuracy_score(y_test, y_pred_mid_gini),   # test accuracy
    clf_mid_gini.runtime_,                     # runtime
    clf_mid_gini.timeout_                      # timeout reached (boolean)
)

results = (results_eq_freq, results_mid_entropy, results_mid_gini)

if not os.path.exists(f'partial_exp_data/EQUAL_FREQUENCY/{ds_name}/{max_depth}_5'):
    os.makedirs(f'partial_exp_data/EQUAL_FREQUENCY/{ds_name}/{max_depth}_5')

file_name = f"partial_exp_data/EQUAL_FREQUENCY/{ds_name}/{max_depth}_5/fold_{split_idx}"
with open(file_name, "wb") as f:
    pickle.dump(results, f)