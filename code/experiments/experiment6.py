"""
# Experiments 6
[target World] - both open, closed world

[Features] - code/config/experiment6.json
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

- first 20 packets of sequential data
  - direction
  - burst direction
  - cumulative direction


[Models]
- Random forest (models.treeModel.RandomForestWithSeq)
"""

import preprocessing

# import models
import models.knnModel
import models.naiveBayesModel
import models.svmModel
import models.treeModel

import evaluation
import experiments.common

CONFIG_PATH = "code/config/experiment6.json"
DATASET_PATH = "data/experiment6.pkl"


def do_experiment(base_path, result_dir="../result/experiment6"):
    print(f"processing data with {base_path}/{CONFIG_PATH}")
    df = preprocessing.process_data(
        config_path=f"{base_path}/{CONFIG_PATH}",
        path_to_save=f"{base_path}/{DATASET_PATH}",
    )

    modes = ["CM", "OM", "OB"]

    model = models.treeModel.RandomForestWithSeq
    model_list = [model(f"{m}_") for m in modes]

    data = [preprocessing.filter_dataset(df=df, mode=m, test_size=0.25) for m in modes]

    results = [
        experiments.common.train_analyze_model(
            model,
            data[idx],
            model_save_path=f"{result_dir}/models/{model.name}.pkl",
            save_fig_path=f"{result_dir}/confusion-matrix/{model.name}",
        )
        for idx, model in enumerate(model_list)
    ]

    evaluation.compare_results_from_prediction(
        "Experiment6",
        results,
        save_fig_path=f"{result_dir}/bar_chart",
    )
