from sklearn.tree import DecisionTreeClassifier
from utils import count_unique_splits_cart
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pickle
import time
import os

rs = 42


def test_CART_on_dataset(X, y , metric, max_depth, min_sup, n_splits = 10):
    train_accuracies = []
    test_accuracies = []
    n_features = []
    runtimes = []

    # instantiating the cross valudation
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = rs)
    
    for train_idx, test_idx in kf.split(X):
        # Splitting dataset into train set and test set
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        # training the classifier
        clf = DecisionTreeClassifier(
            criterion = metric,
            max_depth = max_depth,
            min_samples_leaf = min_sup,
            random_state = rs
        )
        start_time = time.time()
        clf.fit(X_train, y_train)
        runtime = time.time() - start_time

        # testing the classifier
        train_accuracy = round(accuracy_score(y_train, clf.predict(X_train)), 4)
        test_accuracy = round(accuracy_score(y_test, clf.predict(X_test)), 4)

        # saving the results related to the current split
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        runtimes.append(runtime)

        n_features.append(count_unique_splits_cart(clf.tree_)) # needed to insert the data inside the plot

    return n_features, train_accuracies, test_accuracies, runtimes


datasets = [
    "iris_cleaned",
    "wine_cleaned",
    "sonar_cleaned",
    "penguins_cleaned",
    "ionosphere_cleaned",
    "breast_cancer_cleaned",
    "pima_indians_diabetes",
    "banknote",
    "yeast_cleaned",
    "spambase",
    "pendigits",
    "letter_recognition",
    "shuttle",
    "forest_covtype_cleaned"
]
max_depth_list = [3, 4, 5, 6]
min_sup_list = [5]


for ds_name in datasets:
    print(f"Dataset: {ds_name}")
    
    dataset = pd.read_csv(f"continuous/{ds_name}.csv", header=None)
    X = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()

    for max_depth in max_depth_list:
        for min_sup in min_sup_list:
            # running the experiment
            results = {
                "entropy": test_CART_on_dataset(X, y, "entropy", max_depth, min_sup),
                "gini": test_CART_on_dataset(X, y, "gini", max_depth, min_sup)
            }

            # aggregating the results so that they are ready to be plotted
            results_aggregated = {
                "entropy": (np.mean(results["entropy"][0]), np.mean(results["entropy"][1]), np.mean(results["entropy"][2]), np.mean(results["entropy"][3])),
                "gini": (np.mean(results["gini"][0]), np.mean(results["gini"][1]), np.mean(results["gini"][2]), np.mean(results["gini"][3]))
            }

            # saving the results
            if not os.path.exists(f'exp_data/CART_ONLY/raw_data/'):
                os.makedirs(f'exp_data/CART_ONLY/raw_data/')
            if not os.path.exists(f'exp_data/CART_ONLY/aggregated_data/'):
                os.makedirs(f'exp_data/CART_ONLY/aggregated_data/')

            with open(f"exp_data/CART_ONLY/raw_data/{ds_name}_{max_depth}_{min_sup}_CART_only", "wb") as f:
                pickle.dump(results, f)
            with open(f"exp_data/CART_ONLY/aggregated_data/{ds_name}_{max_depth}_{min_sup}_CART_only", "wb") as f:
                pickle.dump(results_aggregated, f)

print("\nDone!")