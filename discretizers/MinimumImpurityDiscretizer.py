from scipy.stats import entropy
from math import inf
import numpy as np
import bisect
    
class MinimumImpurityDiscretizer:
    """
    A class to discretize continuous features using entropy-based thresholds.

    Attributes:
    _max_n_features (int): Maximum number of features that can be returned after the call to the transform method.
    _metric (string): Metric used to compute the impurity of the intervals.
    _sorted_cut_points_per_feature (dict): A dictionary containing lists of cut points divided by feature.
    _thresholds (list): Final result - sorted thresholds for all the features.
    _current_thresholds (dict): Currently selected thresholds for discretization.
    _data_raw (array-like): Raw features of the dataset.
    _class_labels (array-like): Labels of the samples in the dataset.
    _n_samples (int): Number of samples in the dataset.
    _n_features (int): Number of features in the dataset.
    """

    ###########################
    # - INIT                - #
    ###########################
    
    def __init__(self, max_n_features = inf, metric = "entropy"):
        """
        Initialize the EntropyDiscretizer class.

        Args:
        max_n_features (int): Used to stop the training process in case of very large datasets.
        metric (string): Specifies which metric must be used to compute the impurity. Defaults to "entropy".
        """
        if max_n_features < 1:
            raise Exception("A strictly positive value is required for max_n_features")
        self._max_n_features = max_n_features
        
        if metric not in  ["entropy", "gini"]:
            raise Exception("metric can only be 'entropy' or 'gini'")
        self._metric = metric
        
        self._sorted_cut_points_per_feature = {}
        self._thresholds = []
        self._current_thresholds = None
        self._data_raw = None
        self._class_labels = None
        self._n_samples = None
        self._n_features = None

    ###########################
    # - AUXILIARY FUNCTIONS - #
    ###########################

    def compute_impurity(self, labels, base=2):
        """
        Calculate the impurity score of a list of labels.

        Args:
        labels (array-like): The array of labels.
        base (int, optional): The base of the logarithm. Defaults to 2.

        Returns:
        float: Either the entropy value or the gini index, depending on the value of self._metric.
        """
        _, counts = np.unique(labels, return_counts=True)

        if self._metric == "entropy":
            # compute entropy
            impurity = entropy(counts, base=base)
        else:
            # compute gini index
            square_probs = (counts / len(labels)) ** 2
            impurity = 1 - np.sum(square_probs)
        return impurity
            

    def get_index_for_sorted_insertion(self, sorted_list, element):
        """
        Get the index for the sorted insertion of an element in a list.

        Args:
        sorted_list (list): The list where the element will be inserted.
        element: The element to be inserted.

        Returns:
        int: The index for the sorted insertion.
        """
        return bisect.bisect_left(sorted_list, element)

    def threshold_encoding(self, value, thresholds):
        """
        Encode a value based on thresholds.

        Args:
        value: The value to be encoded.
        thresholds (list): List of thresholds (sorted in increasing order).

        Returns:
        array-like: Encoded array based on thresholds. Each element in the array is 0 if value is smaller or equal to the corresponding threshold,
                    otherwise it is 1.
        """
        feat = []
        for i in np.arange(len(thresholds)):
            if value <= thresholds[i]:
                feat.append(0)
            else:
                feat.append(1)
        return np.array(feat)
    
    ###########################
    # - FIT                 - #
    ###########################
    
    def fit(self, X, y):
        """
        Fit the EntropyDiscretizer to the data.

        Args:
        X (array-like): The features of the dataset.
        y (array-like): The labels of the samples in the dataset.
        """
        self._data_raw = X  # make a copy of original input data
        self._class_labels = y.reshape(y.size)  # make sure class labels is a row vector
        self._n_samples, self._n_features = X.shape

        # ------------------------------------------------------- #
        # COMPUTING THE THRESHOLDS INDIVIDUALLY FOR EVERY FEATURE #
        # ------------------------------------------------------- #        
        for col_idx in range(self._n_features):
            # the following steps are repeated for every feature
            idxs = np.argsort(self._data_raw[:, col_idx])
            x_sorted = self._data_raw[idxs, col_idx]
            y_sorted = self._class_labels[idxs]
            
            # -- FINDING THE BOUNDARY POINTS -- (namely, the average values between two consecutive samples of different classes)
            boundary_points, boundary_idxs = self.compute_boudary_points(x_sorted, y_sorted)
            
            # -- SORTING THE CUT POINTS FOR A SINGLE FEATURE USING THE ENTROPY --
            sorted_cut_points, entropies = self.sort_boundary_points_for_single_feature(y_sorted, boundary_points, boundary_idxs)

            # if len(sorted_cut_points) > 0:
            #     # the first threshold is used in any case --> it is immediately added to the result
            #     self._thresholds.append((col_idx, sorted_cut_points[0])) # col_idx allows to assign the threshold to the right feature

            # -- COMPUTING THE ENTROPY GAIN FOR EVERY THRESHOLD --
            for th_idx in range(len(sorted_cut_points)):
                gain = entropies[th_idx] - entropies[th_idx + 1]
                sorted_cut_points[th_idx] = (sorted_cut_points[th_idx], gain) # the gain will be used in the next step to sort thresholds coming from different features
            
            self._sorted_cut_points_per_feature[col_idx] = sorted_cut_points

            # if the n. of thresholds found for a feature is 0, then self._sorted_cut_points_per_feature[col_idx] is empty

        # ------------------------------------------------ #
        # MERGING TOGETHER ALL THE THRESHOLDS              #
        # ------------------------------------------------ #
        self.merge_boundary_points()
        
        return self

    def compute_boudary_points(self, x_sorted, y_sorted):
        """
        Compute boundary points for discretization.

        Args:
        x_sorted (array-like): Sorted values of the feature.
        y_sorted (array-like): Sorted labels corresponding to x_sorted.

        Returns:
        tuple: A tuple containing lists of boundary points and their corresponding indices.
        """
        boundary_points = []
        boundary_idxs = []

        for sample_idx in range(1, self._n_samples):
            if x_sorted[sample_idx] != x_sorted[sample_idx-1]:
                # the current sample and the one preceding it don't have the same value for the current feature
                boundary_point = (x_sorted[sample_idx-1] + x_sorted[sample_idx]) / 2

                if y_sorted[sample_idx] != y_sorted[sample_idx-1]:
                    # if the classes of the two samples are different, this is a boundary point for sure
                    boundary_points.append(boundary_point)
                    boundary_idxs.append(sample_idx)
                else:
                    # otherwise, it could still be a point of interest
                    previous_val_idxs = np.where(x_sorted == x_sorted[sample_idx-1])[0]
                    current_val_idxs = np.where(x_sorted == x_sorted[sample_idx])[0]

                    # a possible cut point is ignored only if ALL the samples having as values "x_sorted[sample_idx-1]" or "x_sorted[sample_idx]"
                    # for the current feature have the same class
                    merged_classes = np.union1d(y_sorted[previous_val_idxs], y_sorted[current_val_idxs])
    
                    if merged_classes.size > 1:
                        boundary_points.append(boundary_point)
                        boundary_idxs.append(sample_idx)

        return boundary_points, boundary_idxs
    
    def sort_boundary_points_for_single_feature(self, labels, boundary_points, boundary_idxs):
        """
        Sort boundary points for a single feature based on entropy.

        Args:
        labels (array-like): Labels of the samples.
        boundary_points (list): List of boundary points.
        boundary_idxs (list): Indices of boundary points.

        Returns:
        tuple: A tuple containing sorted boundary points and corresponding entropies.
        """
        # auxiliary variables
        old_ent = self.compute_impurity(labels)
        old_th_idxs = [] # idxs associated to the thresholds that have already been selected

        sorted_boundary_points = []
        entropies = [old_ent]
        
        # caching variables used to speed up the computations
        parent_entropy_cache = {th: old_ent for th in boundary_points}
        entropy_interval_cache = {} # this is used to avoid recomputing the same entropy value multiple times
        
        N_iterations = min(len(boundary_points), self._max_n_features)
        for i in range(N_iterations):
            # for every iteration, the boundary point leading to the minimum entropy is chosen
            min_ent = inf
            
            for candidate, th_idx in zip(boundary_points, boundary_idxs):
                # adding the candidate to the thresholds that have already been selected (notice that the algorithm works with the indices)
                sorted_ins_idx = self.get_index_for_sorted_insertion(old_th_idxs, th_idx)
                th_idxs = old_th_idxs[:sorted_ins_idx] + [th_idx] + old_th_idxs[sorted_ins_idx:]
            
                # computing the entropy related to the new intervals        
                prev_idx = th_idxs[sorted_ins_idx - 1] if sorted_ins_idx > 0 else 0
                next_idx = th_idxs[sorted_ins_idx + 1] if sorted_ins_idx < (len(th_idxs)-1) else self._n_samples
                
                left_interval = labels[prev_idx:th_idx]
                right_interval = labels[th_idx:next_idx]

                if (prev_idx, th_idx) not in entropy_interval_cache:
                    entropy_left = self.compute_impurity(left_interval) * left_interval.size / self._n_samples
                    entropy_interval_cache[(prev_idx, th_idx)] = entropy_left
                else:
                    entropy_left = entropy_interval_cache[(prev_idx, th_idx)]

                
                if (th_idx, next_idx) not in entropy_interval_cache:
                    entropy_right = self.compute_impurity(right_interval) * right_interval.size / self._n_samples
                    entropy_interval_cache[(th_idx, next_idx)] = entropy_right
                else:
                    entropy_right = entropy_interval_cache[(th_idx, next_idx)]
                
                # computing the entropy related to the current candidate
                ent = old_ent - parent_entropy_cache[candidate] + entropy_left + entropy_right
            
                # checking whether the current threshold is better than the previous alternatives or not
                if min_ent > ent:
                    min_ent = ent
                    best_entropy_left = entropy_left
                    best_entropy_right = entropy_right
                    
                    best_cut = candidate
                    best_idx = th_idx
                    best_th_idxs = th_idxs

                    best_prev_idx = prev_idx
                    best_next_idx = next_idx
        
            # adding the winner to the sorted_boundary_points
            sorted_boundary_points.append(best_cut)
            entropies.append(min_ent)
            
            # removing picked threshold from the candidates
            boundary_points.remove(best_cut)
            boundary_idxs.remove(best_idx)
            
            # updating the cache and the other auxiliary variables
            for th, th_idx in zip(boundary_points, boundary_idxs):
                if (th_idx >= best_prev_idx) and (th_idx < best_idx):
                    parent_entropy_cache[th] = best_entropy_left
                if (th_idx >= best_idx) and (th_idx < best_next_idx):
                    parent_entropy_cache[th] = best_entropy_right
            
            old_th_idxs = best_th_idxs
            old_ent = min_ent

        return sorted_boundary_points, entropies
    
    def merge_boundary_points(self):
        """
        Merge boundary points of all features.
        """
        indexes = [0 for i in range(self._n_features)]
        stop = False
        n_iterations = 0

        while (n_iterations < self._max_n_features) and (not stop):
            # for every iteration, a new threshold is added
            stop = True

            max_gain = -inf
            for col_idx in range(self._n_features):
                if indexes[col_idx] < len(self._sorted_cut_points_per_feature[col_idx]):
                    # there still are thresholds to be considered for the current feature
                    candidate_cut, candidate_gain = self._sorted_cut_points_per_feature[col_idx][indexes[col_idx]]
                    if candidate_gain > max_gain:
                        max_gain = candidate_gain
                        best_cut = candidate_cut
                        best_idx = col_idx
                        stop = False # at least one threshold to apply has been found

            if not stop:
                # a new best threshold has been found
                self._thresholds.append((best_idx, best_cut))
                indexes[best_idx] += 1 # the index related to the selected feature is incremented to consider the following thresholds in the next itereations
                n_iterations += 1

    ###########################
    # - TRANSFORM           - #
    ###########################

    def transform(self, X, n_features):
        """
        Transform the data based on the fitted thresholds.

        Args:
        X (array-like): The features of the dataset.
        n_features (int): The number of features to transform.

        Returns:
        array-like: Transformed dataset.
        """
        # checking the correctness of the parameters - TO BE FIXED SO THAT IT WORKS WITH FEATURES HAVING ONLY ONE VALUE
        # if n_features < self._n_features:
        #     raise Exception("The specified number of features is not suitable: it must be grater than or equal to the original dimension of the dataset.")
        if n_features < 1:
            raise Exception("The specified number of features is not suitable: it must be at least 1")
        if n_features > len(self._thresholds):
            raise Exception(f"The specified number of features is not suitable: given the fitting parameters, it can be at most {len(self._thresholds)}.")

        # add the check for the dimensions of X (n columns)

        # the first n_features thresholds are selected and applied to the data to be transformed
        thresholds_dict = {col_idx: [] for col_idx in range(self._n_features)}
        
        for col_idx, th in self._thresholds[:n_features]:
            thresholds_dict[col_idx].append(th)

        # application of the thresholds
        columns = []
        for col_idx in range(self._n_features):
            thresholds = sorted(thresholds_dict[col_idx])
            thresholds_dict[col_idx] = thresholds

            if len(thresholds) > 0:
                # the features for which no thresholds have been selected are just ignored
                column = []
                for value in X[:, col_idx]:
                    column.append(self.threshold_encoding(value, thresholds))
                columns.append(np.vstack(column))
            # else:
            #     # no thresholds have been found for the current feature --> all values are converted to 1 (zero would be fine as well)
            #     column = np.ones((X[:, col_idx].size, 1), int)
            
            # columns.append(np.vstack(column))

        self._current_thresholds = thresholds_dict
        
        discretized_data = np.hstack(columns)

        return discretized_data
    
    def fit_transform(self, X, y, n_features):
        return self.fit(X=X, y=y).transform(X=X, n_features=n_features)