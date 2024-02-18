import numpy as np
import itertools


class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_=0.1, feat_map_deg=2, num_iterations=1000, is_polynomial=True):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.feat_map_deg = feat_map_deg
        self.is_polynomial = is_polynomial

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Condition on feature mapping
        X = self.map_features(X) if self.is_polynomial else X
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)

            # Condition on feature regularization
            if self.is_polynomial:
                w_gradient = (np.dot(X.T, (h - y)) +
                              self.lambda_ * self.weights) / num_samples
                b_gradient = self.lambda_ * self.bias / num_samples
            else:
                w_gradient = np.dot(X.T, (h - y)) / num_samples
                b_gradient = np.sum(h-y) / num_samples

            self.weights -= self.learning_rate * w_gradient
            self.bias -= self.learning_rate * b_gradient

    def predict(self, X, threshold=0.5):
        # Condition on feature mapping
        X = self.map_features(X) if self.is_polynomial else X
        prob = self.sigmoid(np.dot(X, self.weights))
        return prob >= threshold

    def map_features(self, X):
        """
        Takes a set of features and creates new combinations of those features. 
        It starts with a set of features that contains just a column of ones.
        It iteratively combines different sets of original features to generate new features. 
        """
        # Initialize the output array with a column of ones
        out = np.ones(X.shape[0]).reshape(-1, 1)

        # Iterate over feature degrees from 1 up to feat_map_deg
        for d in range(1, self.feat_map_deg + 1):
            # Generate combinations of features with replacement
            for comb in itertools.combinations_with_replacement(range(X.shape[1]), d):
                # Compute the product of selected features and concatenate it to the output array
                out = np.hstack(
                    (out, np.prod(X[:, comb], axis=1).reshape(-1, 1)))

        return out
