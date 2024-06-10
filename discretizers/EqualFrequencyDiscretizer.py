from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

class EqualFrequencyDiscretizer(KBinsDiscretizer):
    def __init__(self, n_bins, random_state = None):
        # initialize parent class
        super().__init__(
            n_bins = n_bins,
            strategy = "quantile", # equal frequency discretization
            subsample = None,
            random_state = random_state
        )

    def fit(self, X, y = None, sample_weight = None):
        super().fit(X, y, sample_weight)
        return self
    
    def __threshold_encoding(self, value, thresholds):
        feat = []
        for th in thresholds:
            if value <= th:
                feat.append(0)
            else:
                feat.append(1)
        return np.array(feat)

    def transform(self, X):
        columns = []
        
        for col_idx in range(X.shape[1]):
            thresholds = self.bin_edges_[col_idx][1:-1]

            if thresholds.size > 0:
                # the features for which no thresholds have been selected are just ignored
                column = []
                for value in X[:, col_idx]:
                    column.append(self.__threshold_encoding(value, thresholds))
                columns.append(np.vstack(column))
        
        X_discretized = np.hstack(columns)

        return X_discretized

    def fit_transform(self, X, y = None, sample_weight = None):
        return self.fit(X, y, sample_weight).transform(X)