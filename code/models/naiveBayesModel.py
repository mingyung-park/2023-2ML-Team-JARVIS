from modelClass import ModelClass
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np


class NaiveBayesModel(ModelClass):
    name = "NaiveBayes"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param_grid = {"var_smoothing": np.logspace(0, -9, num=100)}
        self.model = GridSearchCV(
            GaussianNB(),
            param_grid,
            verbose=1,
            refit=True,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp
