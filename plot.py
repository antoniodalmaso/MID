import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

###########################
# - AUXILIARY FUNCTIONS - #
###########################

# Function used to aggregate the data related to different folds
def aggr_function(x):
    return np.mean(x)

# Basic function that plots the data related to a single plot
def fill_subplot(ax, x, y1, y2, y3, y4, y1_label, y2_label, y3_label, y4_label, x_label, y_label):
    fontsize = 15
    linestyle = (0, (5, 5))

    # Plot y1 - train and test
    ax.plot(
        x, y1["train"], color = "tab:blue",
        label = y1_label + " - Train set"
    )
    ax.plot(
        x, y1["test"], color = "tab:blue",
        label = y1_label + " - Test set",
        linestyle = linestyle
    )

    # Plot y3 - train and test
    ax.plot(
        x, y3["train"], color = "tab:orange",
        label = y3_label + " - Train set"
    )
    ax.plot(
        x, y3["test"], color = "tab:orange",
        label = y3_label + " - Test set",
        linestyle = linestyle
    )
    
    # Plot y2 - train and test
    ax.plot(
        x, y2["train"], color = "tab:green",
        label = y2_label + " - Train set"
    )
    ax.plot(
        x, y2["test"], color = "tab:green",
        label = y2_label + " - Test set",
        linestyle = linestyle
    )
    
    # Plot y4 - train and test
    ax.plot(
        x, y4["train"], color = "tab:red",
        label = y4_label + " - Train set"
    )
    ax.plot(
        x, y4["test"], color = "tab:red",
        label = y4_label + " - Test set",
        linestyle = linestyle
    )
    
    # adding details
    ax.grid(True)
    
    ax.set_xticks([el for el in x], labels=[f'{el}' for el in x], rotation=45)
    ax.set_xlim([x[0], x[-1]])
    
    ax.set_xlabel(x_label, fontsize = fontsize)
    ax.set_ylabel(y_label, fontsize = fontsize)

# Function that retireves, aggregates and defines the data to be included in every single plot
def plot_experiments_results(ds_name, max_depth, min_sup, format = "svg"):
    # load data about accuracy
    with open(f'exp_data/{ds_name}/{ds_name}_{max_depth}_{min_sup}_impurity', 'rb') as f:
        raw_data = pickle.load(f)

    # load cart only pre-aggregated data
    with open(f'exp_data/CART_ONLY/aggregated_data/{ds_name}_{max_depth}_{min_sup}_CART_only', 'rb') as f:
        cart_only_data = pickle.load(f)
    
    # grouping toghether the statistics related to the same number of features
    data_entropy = {}
    data_gini = {}
    
    for fold in raw_data:
        entropy_measures, entropy_thresholds = fold["entropy"]
        gini_measures, gini_thresholds = fold["gini"]
    
        # measures
        for entropy_tuple, gini_tuple in zip(entropy_measures, gini_measures):
            # entropy
            n_features, train_acc, test_acc, runtime = entropy_tuple
            if n_features not in data_entropy:
                data_entropy[n_features] = {
                    "dl85": [[],[],[]],
                    "cart": [[],[],[]]
                }
            data_entropy[n_features]["dl85"][0].append(train_acc["dl85"])
            data_entropy[n_features]["dl85"][1].append(test_acc["dl85"])
            data_entropy[n_features]["dl85"][2].append(runtime["dl85"])
            data_entropy[n_features]["cart"][0].append(train_acc["cart"])
            data_entropy[n_features]["cart"][1].append(test_acc["cart"])
            data_entropy[n_features]["cart"][2].append(runtime["cart"])
    
            # gini
            n_features, train_acc, test_acc, runtime = gini_tuple
            if n_features not in data_gini:
                data_gini[n_features] = {
                    "dl85": [[],[],[]],
                    "cart": [[],[],[]]
                }
            data_gini[n_features]["dl85"][0].append(train_acc["dl85"])
            data_gini[n_features]["dl85"][1].append(test_acc["dl85"])
            data_gini[n_features]["dl85"][2].append(runtime["dl85"])
            data_gini[n_features]["cart"][0].append(train_acc["cart"])
            data_gini[n_features]["cart"][1].append(test_acc["cart"])
            data_gini[n_features]["cart"][2].append(runtime["cart"])
    
    # grouping together the values of the single statistics for the plot
    x = []
    
    entropy_stats = {
        "dl85": {
            "train": [],
            "test": [],
            "runtimes": [],
            "colors": []
        },
        "cart": {
            "train": [],
            "test": [],
            "runtimes": [],
            "colors": []
        }
    }
    gini_stats = {
        "dl85": {
            "train": [],
            "test": [],
            "runtimes": [],
            "colors": []
        },
        "cart": {
            "train": [],
            "test": [],
            "runtimes": [],
            "colors": []
        }
    }
    
    for key in data_entropy.keys():
        x.append(key)
    
        entropy_stats["dl85"]["train"].append(aggr_function(data_entropy[key]["dl85"][0]))
        entropy_stats["dl85"]["test"].append(aggr_function(data_entropy[key]["dl85"][1]))
        entropy_stats["dl85"]["runtimes"].append(aggr_function(data_entropy[key]["dl85"][2]))
        entropy_stats["dl85"]["colors"].append("red" if np.any(np.array(data_entropy[key]["dl85"][2]) > 300) else "green")
        entropy_stats["cart"]["train"].append(aggr_function(data_entropy[key]["cart"][0]))
        entropy_stats["cart"]["test"].append(aggr_function(data_entropy[key]["cart"][1]))
        entropy_stats["cart"]["runtimes"].append(aggr_function(data_entropy[key]["cart"][2]))
        entropy_stats["cart"]["colors"].append("red" if np.any(np.array(data_entropy[key]["cart"][2]) > 300) else "green")
    
        gini_stats["dl85"]["train"].append(aggr_function(data_gini[key]["dl85"][0]))
        gini_stats["dl85"]["test"].append(aggr_function(data_gini[key]["dl85"][1]))
        gini_stats["dl85"]["runtimes"].append(aggr_function(data_gini[key]["dl85"][2]))
        gini_stats["dl85"]["colors"].append("red" if np.any(np.array(data_gini[key]["dl85"][2]) > 300) else "green")
        gini_stats["cart"]["train"].append(aggr_function(data_gini[key]["cart"][0]))
        gini_stats["cart"]["test"].append(aggr_function(data_gini[key]["cart"][1]))
        gini_stats["cart"]["runtimes"].append(aggr_function(data_gini[key]["cart"][2]))
        gini_stats["cart"]["colors"].append("red" if np.any(np.array(data_gini[key]["cart"][2]) > 300) else "green")
    
    entropy_stats["dl85"]["avg_train"] = round(np.mean(entropy_stats["dl85"]["train"]) * 100, 2)
    entropy_stats["dl85"]["avg_test"] = round(np.mean(entropy_stats["dl85"]["test"]) * 100, 2)
    entropy_stats["cart"]["avg_train"] = round(np.mean(entropy_stats["cart"]["train"]) * 100, 2)
    entropy_stats["cart"]["avg_test"] = round(np.mean(entropy_stats["cart"]["test"]) * 100, 2)
    
    gini_stats["dl85"]["avg_train"] = round(np.mean(gini_stats["dl85"]["train"]) * 100, 2)
    gini_stats["dl85"]["avg_test"] = round(np.mean(gini_stats["dl85"]["test"]) * 100, 2)
    gini_stats["cart"]["avg_train"] = round(np.mean(gini_stats["cart"]["train"]) * 100, 2)
    gini_stats["cart"]["avg_test"] = round(np.mean(gini_stats["cart"]["test"]) * 100, 2)
    
    # plotting the data
    if not os.path.exists(f'PAPER_PLOTS/{ds_name}/{max_depth}_{min_sup}'):
        os.makedirs(f'PAPER_PLOTS/{ds_name}/{max_depth}_{min_sup}')

    #------------------#
    # -- ACCURACIES -- #
    #------------------#
    fig, ax = plt.subplots(figsize=(10, 5))
    fill_subplot(
        ax = ax,
        x = x,
        y1 = entropy_stats["dl85"],
        y2 = entropy_stats["cart"],
        y3 = gini_stats["dl85"],
        y4 = gini_stats["cart"],
        y1_label = "DL8.5 (entropy)",
        y2_label = "CART (entropy)",
        y3_label = "DL8.5 (gini)",
        y4_label = "CART (gini)",
        x_label = "N. of features",
        y_label = "Accuracy"
    )

    # adding data about the execution of the cart algorithm alone
    ax.scatter(cart_only_data["entropy"][0], cart_only_data["entropy"][1], s = 20,  c = "black",  label = f"CART only - Train set ({cart_only_data['entropy'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["entropy"][1], cart_only_data["entropy"][1]],
        lw = 0.8,  c = "black", ls = (0, (5, 10))
    )
    ax.scatter(cart_only_data["entropy"][0], cart_only_data["entropy"][2], s = 20,  c = "grey",  label = f"CART only - Test set ({cart_only_data['entropy'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["entropy"][2], cart_only_data["entropy"][2]],
        lw = 0.8,  c = "grey", ls = (0, (5, 10))
    )

    # saving the plot
    ax.legend(fontsize=12)
    fig.tight_layout()
    
    plt.savefig(f'PAPER_PLOTS/{ds_name}/{max_depth}_{min_sup}/accuracies_comparison.{format}', format=format, bbox_inches='tight')
    plt.close()


    #------------------#
    # --  RUNTIMES  -- #
    #------------------#
    fig, ax = plt.subplots(figsize=(10, 5))
    fontsize = 15
    
    ax.plot(x, entropy_stats["dl85"]["runtimes"],  label = "DL8.5 - entropy")
    ax.plot(x, gini_stats["dl85"]["runtimes"],  label = "DL8.5 - gini")
    ax.plot(x, entropy_stats["cart"]["runtimes"],  label = "CART - entropy")
    ax.plot(x, gini_stats["cart"]["runtimes"],  label = "CART - gini")
    
    # adding details
    ax.grid(True)
    
    ax.set_xticks([el for el in x], labels=[f'{el}' for el in x], rotation=45)
    ax.set_xlim([x[0], x[-1]])
    
    ax.set_xlabel("N. of features", fontsize = fontsize)
    ax.set_ylabel("Runtimes (s)", fontsize = fontsize)

    # saving the plot
    ax.legend(fontsize=12)
    fig.tight_layout()
    
    plt.savefig(f'PAPER_PLOTS/{ds_name}/{max_depth}_{min_sup}/runtimes_comparison.{format}', format=format, bbox_inches='tight')
    plt.close()



##########################
# -        CODE        - #
##########################
datasets = [
    # "iris_cleaned",
    # "wine_cleaned",
    # "sonar_cleaned",
    # "penguins_cleaned",
    # "ionosphere_cleaned",
    # "breast_cancer_cleaned",
    # "pima_indians_diabetes",
    "banknote",
    # "yeast_cleaned",
    # "spambase",
    # "musk_cleaned",
    # "pendigits",
    # "letter_recognition",
    # "shuttle",
    # "forest_covtype_cleaned"
]

max_depth_list = [3, 4, 5, 6]
min_sup_list = [5]

for ds_name in datasets:
    print(f"Dataset: {ds_name}")
    for max_depth in max_depth_list:
        for min_sup in min_sup_list:
            plot_experiments_results(ds_name, max_depth, min_sup)

print("\nDone!")