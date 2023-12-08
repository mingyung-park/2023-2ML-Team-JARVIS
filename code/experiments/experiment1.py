"""
# Experiments 1
[target World] - only closed world

[Features] - code/config/experiment1.json
only with categorical features.

- total count
- incoming count
- outgoing count
- incoming rate
- outgoing rate


[Models]
- KNN with standard scaler (models.knnModel.KNNModel)
- NaiveBayes (models.naiveBayesModel.NaiveBayesModel)
- SVM (models.svmModel.SVMModel)
- Random forest (models.treeModel.RandomForestModel)
- Bagging classifier with DecisionTreeClassifier as estimator (models.treeModel.BaggingClassifierModel)
"""

import preprocessing

# import models
import models.knnModel
import models.naiveBayesModel
import models.svmModel
import models.treeModel

import evaluation
import experiments.common

CONFIG_PATH = "code/config/experiment1.json"
DATASET_PATH = "data/experiment1.pkl"


def do_experiment(base_path, result_dir="../result/experiment1"):
    print(f"processing data with {base_path}/{CONFIG_PATH}")
    df = preprocessing.process_data(
        config_path=f"{base_path}/{CONFIG_PATH}",
        path_to_save=f"{base_path}/{DATASET_PATH}",
    )
    data = preprocessing.filter_dataset(df=df, mode="CM")
    # data = preprocessing.filter_dataset(
    #     data_path=f"{base_path}/{DATASET_PATH}", mode="CM"
    # )

    model_list = [
        models.knnModel.KNNModel(),
        models.naiveBayesModel.NaiveBayesModel(),
        models.svmModel.SVMModel(),
        models.treeModel.RandomForestModel(),
        models.treeModel.BaggingClassifierModel(),
    ]

    results = [
        experiments.common.train_analyze_model(
            model,
            data,
            model_save_path=f"{result_dir}/models/{model.name}.pkl",
            save_fig_path=f"{result_dir}/confusion-matrix/{model.name}",
        )
        for model in model_list
    ]

    evaluation.compare_results_from_prediction(
        "Experiment1",
        results,
        save_fig_path=f"{result_dir}/bar_chart",
    )
