from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from modelClass import ModelClass


class RandomForest(ModelClass):
    model = RandomForestClassifier(n_estimators=5, criterion="gini")


class RandomForestNoSeq(ModelClass):
    model = RandomForestClassifier(n_estimators=5, criterion="gini")
    config_path = "./code/config/only-categorical.json"
