# Minimum Impurity Discretizer (MID)
This repository contains the code related to the development and testing of the Minimum Impurity Discretizer (MID) algorithm. MID is a new discretization technique developed to enhance the performance of Optimal Decision Tree (ODT) learners when applied to continuous datasets. This technique is designed to be used with [DL8.5](https://ojs.aaai.org/index.php/AAAI/article/view/5711).

The Python implementation of the discretizer can be found in `discretizers/MinimumImpurityDiscretizer.py`.

## How to Run the Experiments
All experiments were conducted on the CECI cluster [Lemaitre4](https://www.ceci-hpc.be/clusters.html#lemaitre4), which uses the Slurm job scheduler. To reproduce the results, execute the bash scripts in the main folder using the command:

```
sbatch script_name.sh
```

Run the scripts in the following order:
1. **_submit_mid.sh_**: Trains DL8.5 and CART on a single dataset discretized using MID, using all numbers of features between 1 and 45. Modify line 13 of the script to specify the dataset, the maximum depth of the trees, and the minimum support of the leaves (in this order).
2. **_submit_mdlp.sh_**: Trains DL8.5 on a single dataset discretized using MDLP. Modify line 13 of the script to specify the dataset.
3. **_submit_equal_freq.sh_**: Trains DL8.5 on a single dataset discretized using 8-bin equal frequency. Modify line 13 of the script to specify the dataset.
4. **_submit_cart_only.sh_**: Trains and tests CART on all original continuous datasets.
5. **_submit_aggregate.sh_**.
6. **_submit_plot.sh_**.

## Credits
The implementation of the MDLP discretization technique used for comparison with MID was developed by Victor Ruiz (vmr11@pitt.edu). The original repository can be found [here](https://github.com/navicto/Discretization-MDLPC.git).

# MID Pseudocode
![pseudocode](https://github.com/user-attachments/assets/8afc36f8-bf98-4bef-94ec-41ae6323c375)
