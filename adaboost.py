import sys
import os
import random
import numpy as np
import random

sys.path.append(os.path.abspath("../Regression-Trees---From-Scratch"))
from regression_trees import RegressionTree

class AdaBoost:
    def __init__(self, X, y, max_number_of_trees = 10):
        self.X = X
        self.y = y
        self.max_number_of_trees = max_number_of_trees
    def fit(self):
        self._trees = []
        self._model_weights = []

        number_of_observations = np.shape(self.X)[0]
        weights = np.ones(number_of_observations)/number_of_observations

        # Initialize L
        L = 0
        number_of_trees = 0
        while L <= 0.5 and number_of_trees<=self.max_number_of_trees: 
            number_of_trees += 1
            sample_indexes = random.choices(range(number_of_observations), weights=weights, k=number_of_observations)
            X_tmp = self.X[sample_indexes,:]
            y_tmp = self.y[sample_indexes]

            # Train the model
            model = RegressionTree(X_tmp, y_tmp, max_depth=2)
            model.fit()
            self._trees.append(model)

            # Get the model's predictions
            pred = model.predict(X_tmp)

            # Calculate L (weighted error) 
            D = np.max(np.abs(pred-y_tmp))
            L_n = np.abs(pred-y_tmp) / D
            L = np.sum(L_n * weights)

            # Calculate the model's weight (beta)
            beta = L / (1-L)

            # Update weights for the next iteration
            Z = weights * beta**(1-L_n)
            weights = Z / np.sum(Z)

            # Save model weight (log scale)
            self._model_weights.append(np.log(1/beta))
    def predict(self, X):
        # Collect predictions from all trees
        tree_preds = np.zeros((len(self._trees),len(X)))
        for i, tree in enumerate(self._trees):
            tree_preds[i, :] = tree.predict(X)

        # Stack model weights to the predictions
        tree_preds = np.column_stack((tree_preds, self._model_weights))
        
        
        pred = []
        for i in range(np.shape(tree_preds)[1]):
            # Sorting predictions for each sample
            sorted_preds = tree_preds[np.argsort(tree_preds[:, i]), :]

            # Calculating the cumulative sum of weights
            weight_cumsum = np.cumsum(sorted_preds[:,-1])

            # Adding the cumulative sum of weights to the results of sorted predictions
            sorted_preds = np.column_stack((sorted_preds, weight_cumsum))

            # Calculating the weight median
            pred.append(sorted_preds[np.where(sorted_preds[:,-1] > sorted_preds[-1,-1]/2)[0][0],i])
        
        return pred[:-1]

        