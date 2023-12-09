from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from modelClass import ModelClass


class KNNModel(ModelClass):
    name = "KNN-with-StandardScaler"

    def __init__(self, prefix="") -> None:
        self.name = prefix + self.name
        param = {
            "knn__n_neighbors": list(range(3,7)),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan", "minkowski"],
        }

        model = Pipeline(
            [
                ("ss", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        self.model = GridSearchCV(
            model,
            param,
            refit=True,
            verbose=1,
            cv=5,
        )

    def fit(self, x, y):
        temp = super().fit(x, y)
        print(f"best param: {self.model.best_params_}")
        return temp
