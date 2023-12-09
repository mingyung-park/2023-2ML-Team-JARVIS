from sklearn.ensemble import (
    # AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    # GradientBoostingClassifier,
    # VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from modelClass import ModelClass


class SimpleRandomForest(ModelClass):
    name = "RandomForest"

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, criterion="gini")


class RandomForestModel(ModelClass):
    name = "RandomForest"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "n_estimators": [10, 50, 100],
            "criterion": ["gini", "entropy"],
        }
        self.model = GridSearchCV(
            RandomForestClassifier(),
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp


class BaggingClassifierModel(ModelClass):
    name = "BaggingClassifier"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "n_estimators": [10, 50, 100],
        }

        estimator = DecisionTreeClassifier()
        self.model = GridSearchCV(
            BaggingClassifier(estimator),
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp


class RandomForestWithSeq(ModelClass):
    name = "RandomForest-SEQ"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "n_estimators": [100],
            "criterion": ["gini", "entropy"],
        }
        self.model = GridSearchCV(
            RandomForestClassifier(),
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp


class BaggingClassifierWithSeq(ModelClass):
    name = "BaggingClassifier-SEQ"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "n_estimators": [100],
        }

        estimator = DecisionTreeClassifier()
        self.model = GridSearchCV(
            BaggingClassifier(estimator),
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp
