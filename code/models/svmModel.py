from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from modelClass import ModelClass


class SVMModel(ModelClass):
    name = "SVM"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "C": [0.1, 1, 10],
            "gamma": [1, 0.1, 0.01],
            "kernel": ["rbf"],
        }
        self.model = GridSearchCV(
            SVC(),
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp
