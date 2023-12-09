"""
# Experiments 4
[target World] - both open, closed world

[Features] - code/config/experiment4.json
- total count
- incoming count
  - of total packet
  - of first 10, 20, 30, 40, 50 packets
- outgoing count
  - of total packet
  - of first 10, 20, 30, 40, 50 packets
- incoming rate
  - of total packet
  - of first 10, 20, 30, 40, 50 packets
- outgoing rate
  - of total packet
  - of first 10, 20, 30, 40, 50 packets

- first 50 packets of sequential data
  - direction
  - burst direction
  - cumulative direction


[Models]
- KNN with standard scaler (models.knnModel.KNNwithSeq)
- Random forest (models.treeModel.RandomForestWithSeq)
- Bagging classifier with DecisionTreeClassifier as estimator (models.treeModel.BaggingClassifierWithSeq)
"""

import preprocessing

# import models
import models.knnModel
import models.naiveBayesModel
import models.svmModel
import models.treeModel

import evaluation
import experiments.common

CONFIG_PATH = "code/config/experiment4.json"
DATASET_PATH = "data/experiment4.pkl"


def do_experiment(base_path, result_dir="../result/experiment4"):
    print(f"processing data with {base_path}/{CONFIG_PATH}")
    df = preprocessing.process_data(
        config_path=f"{base_path}/{CONFIG_PATH}",
        path_to_save=f"{base_path}/{DATASET_PATH}",
    )

    modes = ["CM", "OM", "OB"]
    model_list = []

    for model in [
        models.knnModel.KNNwithSeq,
        models.treeModel.RandomForestWithSeq,
        models.treeModel.BaggingClassifierWithSeq,
    ]:
        for m in modes:
            model_list.append(model(f"{m}_"))

    data = [preprocessing.filter_dataset(df=df, mode=m, test_size=0.25) for m in modes]

    results = [
        experiments.common.train_analyze_model(
            model,
            data[idx % 3],
            model_save_path=f"{result_dir}/models/{model.name}.pkl",
            save_fig_path=f"{result_dir}/confusion-matrix/{model.name}",
        )
        for idx, model in enumerate(model_list)
    ]

    evaluation.compare_results_from_prediction(
        "Experiment4",
        results,
        save_fig_path=f"{result_dir}/bar_chart",
    )
