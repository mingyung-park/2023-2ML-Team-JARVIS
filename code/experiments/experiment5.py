"""
# Experiments 5
[target World] - closed multi world

[Features] - code/config/experiment5.json
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

- first N packets of sequential data
  - direction
  - burst direction
  - cumulative direction


[Model]
- Random forest (models.treeModel.SimpleRandomForest)
"""

import preprocessing, evaluation

# import models
import models.knnModel
import models.naiveBayesModel
import models.svmModel
import models.treeModel

import json
import matplotlib.pyplot as plt

CONFIG_PATH = "code/config/experiment5.json"
DATASET_PATH = "data/experiment5.pkl"


def seq_length_test(title, dataset, path):
    SIZE = 200

    accuracy = []
    precision = []
    recall = []
    f1 = []

    result_list = []
    train_x, test_x = dataset[0], dataset[1]

    for cnt in range(SIZE, 0, -1):
        model = models.treeModel.SimpleRandomForest()
        model.fit(train_x, dataset[2])
        y_pred = model.predict(test_x)
        result_list.append(evaluation.compare_models(f"{cnt}", dataset[3], y_pred))

        train_x = train_x.drop(
            columns=[
                f"{prefix}_{cnt-1}"
                for prefix in ["direction", "burst_dir", "cumulative_dir"]
            ]
        )
        test_x = test_x.drop(
            columns=[
                f"{prefix}_{cnt-1}"
                for prefix in ["direction", "burst_dir", "cumulative_dir"]
            ]
        )

    result_list.reverse()
    with open(f"{path}/result.json", "w") as f:
        f.write(
            json.dumps(
                result_list,
                indent=4,
                separators=(",", ": "),
            )
        )

    accuracy = [result["accuracy"] for result in result_list]
    precision = [result["precision"] for result in result_list]
    recall = [result["recall"] for result in result_list]
    f1 = [result["f1"] for result in result_list]

    plt.figure(figsize=(20, 8))
    plt.plot(accuracy, label="accuracy")
    plt.plot(precision, label="precision")
    plt.plot(recall, label="recall")
    plt.plot(f1, label="f1")
    plt.xticks(range(0, SIZE + 1, 10))

    plt.title(title)
    plt.legend()
    plt.savefig(f"{path}/plot")


def do_experiment(base_path, result_dir="../result/experiment5"):
    df = preprocessing.process_data(
        config_path=f"{base_path}/{CONFIG_PATH}",
        path_to_save=f"{base_path}/{DATASET_PATH}",
    )

    for mode in ["CM", "OM", "OB"]:
        data = preprocessing.filter_dataset(df=df, mode=mode, test_size=0.25)
        seq_length_test(
            f"size of sequential data - {mode}",
            data,
            f"{base_path}/{result_dir}/{mode}",
        )
