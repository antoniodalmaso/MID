import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def fill_subplot(ax, x, y1, y2, y3, y4, y1_label, y2_label, y3_label, y4_label, x_label, y_label):
    ax.plot(x, y1, label=y1_label)
    ax.plot(x, y2, label=y2_label)
    ax.plot(x, y3, label=y3_label)
    ax.plot(x, y4, label=y4_label)

    ax.grid(True)

    ax.set_xticks([el for el in x], labels=[f'{el}' for el in x], rotation=45)
    ax.set_xlim([x[0], x[-1]])

    ax.set_xlabel(x_label, fontsize = 15)
    ax.set_ylabel(y_label, fontsize = 15)


def plot_results(ds_name, max_depth, min_sup, format = "svg"):
    # retrieving the data previously aggregated
    base_folder = "exp_data/"
    constraints = f"{max_depth}_{min_sup}"

    mid_data_path = base_folder + "MID/" + ds_name + "/" + constraints
    cart_only_data_path = base_folder + "CART_ONLY/aggregated_data/" + f"{ds_name}_{constraints}_CART_only"

    # MID data
    with open(mid_data_path, 'rb') as f:
        mid_data = pickle.load(f)

    dl85_entropy = np.array(mid_data[0])
    cart_entropy = np.array(mid_data[1])
    dl85_gini = np.array(mid_data[2])
    cart_gini = np.array(mid_data[3])

    # CART only data
    with open(cart_only_data_path, 'rb') as f:
        cart_only_data = pickle.load(f)

    # Plotting the data
    plots_folder = f'plots/{ds_name}/{constraints}'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    x = dl85_entropy[:,0].astype(int)

    ########################
    # - TRAIN ACCURACIES - #
    ########################
    fig, ax = plt.subplots(figsize=(10,5))

    # mid data
    y_idx = 1 # train accuracy

    fill_subplot(
        ax = ax,
        x = x,
        y1 = dl85_entropy[:, y_idx],
        y2 = dl85_gini[:, y_idx],
        y3 = cart_entropy[:, y_idx],
        y4 = cart_gini[:, y_idx],
        y1_label = "MID + DL8.5 - entropy",
        y2_label = "MID + DL8.5 - gini",
        y3_label = "MID + CART - entropy",
        y4_label = "MID + CART - gini",
        x_label = "N. input features",
        y_label = "Train accuracies"
    )

    # adding cart only data
    ax.scatter(cart_only_data["entropy"][0], cart_only_data["entropy"][y_idx], s = 20,  c = "black",  label = f"CART ONLY - entropy ({cart_only_data['entropy'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["entropy"][y_idx], cart_only_data["entropy"][y_idx]],
        lw = 0.8,  c = "black", ls = (0, (5, 10))
    )
    ax.scatter(cart_only_data["gini"][0], cart_only_data["gini"][y_idx], s = 20,  c = "grey",  label = f"CART ONLY - gini ({cart_only_data['gini'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["gini"][y_idx], cart_only_data["gini"][y_idx]],
        lw = 0.8,  c = "grey", ls = (0, (5, 10))
    )

    # saving the plot
    ax.legend(fontsize=12)
    fig.tight_layout()

    plt.savefig(f'{plots_folder}/train_accuracies.{format}', format=format, bbox_inches='tight')
    plt.close()

    #######################
    # - TEST ACCURACIES - #
    #######################
    fig, ax = plt.subplots(figsize=(10,5))

    # mid data
    y_idx = 2 # test accuracy

    fill_subplot(
        ax = ax,
        x = x,
        y1 = dl85_entropy[:, y_idx],
        y2 = dl85_gini[:, y_idx],
        y3 = cart_entropy[:, y_idx],
        y4 = cart_gini[:, y_idx],
        y1_label = "MID + DL8.5 - entropy",
        y2_label = "MID + DL8.5 - gini",
        y3_label = "MID + CART - entropy",
        y4_label = "MID + CART - gini",
        x_label = "N. input features",
        y_label = "Test accuracies"
    )

    # adding cart only data
    ax.scatter(cart_only_data["entropy"][0], cart_only_data["entropy"][y_idx], s = 20,  c = "black",  label = f"CART ONLY - entropy ({cart_only_data['entropy'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["entropy"][y_idx], cart_only_data["entropy"][y_idx]],
        lw = 0.8,  c = "black", ls = (0, (5, 10))
    )
    ax.scatter(cart_only_data["gini"][0], cart_only_data["gini"][y_idx], s = 20,  c = "grey",  label = f"CART ONLY - gini ({cart_only_data['gini'][0]} features)")
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["gini"][y_idx], cart_only_data["gini"][y_idx]],
        lw = 0.8,  c = "grey", ls = (0, (5, 10))
    )

    # saving the plot
    ax.legend(fontsize=12)
    fig.tight_layout()

    plt.savefig(f'{plots_folder}/test_accuracies.{format}', format=format, bbox_inches='tight')
    plt.close()

    ########################
    # -     RUNTIMES     - #
    ########################
    fig, ax = plt.subplots(figsize=(10,5))

    # mid data
    y_idx = 3 # runtimes

    fill_subplot(
        ax = ax,
        x = x,
        y1 = dl85_entropy[:, y_idx],
        y2 = dl85_gini[:, y_idx],
        y3 = cart_entropy[:, y_idx],
        y4 = cart_gini[:, y_idx],
        y1_label = "MID + DL8.5 - entropy",
        y2_label = "MID + DL8.5 - gini",
        y3_label = "MID + CART - entropy",
        y4_label = "MID + CART - gini",
        x_label = "N. input features",
        y_label = "Runtimes (s)"
    )

    # saving the plot
    ax.legend(fontsize=12)
    fig.tight_layout()

    plt.savefig(f'{plots_folder}/runtimes.{format}', format=format, bbox_inches='tight')
    plt.close()

    ###############################
    # - INPUT VS. USED FEATURES - #
    ###############################
    fig, ax = plt.subplots(figsize=(10, 5))
    y_idx = 4

    ax.scatter(x, dl85_entropy[:, y_idx], label="MID + DL8.5 - entropy")
    ax.scatter(x, dl85_gini[:, y_idx], label="MID + DL8.5 - gini")

    ax.plot(x, dl85_entropy[:, y_idx], linestyle=(0, (5, 10)), linewidth=0.5)
    ax.plot(x, dl85_gini[:, y_idx], linestyle=(0, (5, 10)), linewidth=0.5)

    # adding cart only data
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["entropy"][0], cart_only_data["entropy"][0]],
        c = "black",
        label = "CART ONLY - entropy"
    )
    ax.plot(
        [x[0], x[-1]],
        [cart_only_data["gini"][0], cart_only_data["gini"][0]],
        c = "grey",
        label = "CART ONLY - gini"
    )

    ax.legend(fontsize = "12")

    ax.set_xticks(x, labels=[f"{v}" for v in x], rotation=45)

    ax.set_xlim([x[0], x[-1]])
    ax.set_xlabel("N. input features", fontsize = 15)
    ax.set_ylabel("N. used features", fontsize = 15)

    ax.grid()
    fig.tight_layout()

    plt.savefig(f'{plots_folder}/input_vs_used_features.{format}', format=format, bbox_inches='tight')
    plt.close()


##########################
# -        CODE        - #
##########################
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
    for max_depth in max_depth_list:
        for min_sup in min_sup_list:
            plot_results(ds_name, max_depth, min_sup)

print("\nDone!")