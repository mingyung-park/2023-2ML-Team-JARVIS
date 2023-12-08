import pickle, json, os
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
import commonUtils

PACKET_SIZE = 512


def put_multi_label(mon_data, unmon_data):
    """put multi label from index of each data

    monitored data: 0 ~ SITE_CNT (=95)
    unmonitored data: -1

    Args:
        mon_data (list): monitored data
        unmon_data (list): _description_

    Returns:
        (list, list): dataset, label
    """
    SITE_CNT = 95
    URL_PER_SITE = 10

    print("parsing dataset...")
    dataset = []
    label = []

    label = [i for i in range(95) for _ in range(200)]

    for i in range(SITE_CNT):
        temp = []
        for j in range(URL_PER_SITE):
            temp.append(mon_data[i * URL_PER_SITE + j])
        dataset.append(list(chain.from_iterable(temp)))

    label.extend([-1] * 10000)
    dataset.append(unmon_data)

    dataset = list(chain.from_iterable(dataset))
    print("Done.\n")
    return dataset, label


def pad_zero_right(arr, size):
    """to make list specific length

    if length of arr is already bigger of equal to size, just return original array

    Args:
        arr (list): list to pad
        size (int): length of the list to be

    Returns:
        list: padded list
    """
    if len(arr) >= size:
        return arr[:size]
    return list(np.pad(arr, (0, max(0, size - len(arr))), mode="constant"))


def calculate_burst_pattern(arr):
    """transform packet list to burst pattern

    Args:
        arr (list): list of packet size | packet direction...

    Returns:
        list: burst pattern
    """
    if not arr:
        return []
    prev = arr[0]
    result = []

    for dir in arr[1:]:
        if (prev * dir) > 0:
            prev += dir
        else:
            result.append(prev)
            prev = dir

    result.append(prev)
    return result


def unfold_array(arr, size, col_name):
    """spread sequential data

    ex) timestamp = [0, 1.3, 2.4]
    => timestamp_0 = 0
       timestamp_1 = 1.3
       timestamp_2 - 2.4

    Args:
        arr (list): column consist of array
        size (int): limit size to spread
        col_name (_type_): name for label

    Returns:
        Dataframe: dataframe with multiple columns
    """
    return pd.DataFrame(
        map(lambda x: pad_zero_right(x, size), arr),
        columns=[f"{col_name}_{i}" for i in range(size)],
    )


def analyze_direction(direction_arr, prefix):
    """analyze direction list - incoming count/rate, outgoing count/rate

    Args:
        direction_arr (list): column consist of array
        prefix (str): label prefix

    Returns:
        Dataframe: datafram with 4 columns(incoming count/rate, outgoing count/rate)
    """
    df = pd.DataFrame()

    length = list(map(len, direction_arr))

    df[f"{prefix}_incoming_count"] = list(map(lambda x: x.count(-1), direction_arr))
    df[f"{prefix}_incoming_rate"] = np.array(df[f"{prefix}_incoming_count"]) / length
    df[f"{prefix}_outgoing_count"] = length - np.array(df[f"{prefix}_incoming_count"])
    df[f"{prefix}_outgoing_rate"] = np.array(df[f"{prefix}_outgoing_count"]) / length
    return df


def extract_features(direction_arr, timestamps_arr, config):
    """extract features from original data

    - spreaded timestamp
    - spreaded direction
    - spreaded packetsize
    - spreaded burst pattern
    - spreaded cumulative packet size
    - analysis for initial packets

    Args:
        direction_arr (list): direction column
        timestamps_arr (list): timestamp column

    Returns:
        Dataframe: columns with extracted featrues
    """
    df = pd.DataFrame()

    columns = []

    if config["FT_TIMESTAMPS"]:
        columns.append(
            unfold_array(timestamps_arr, config["FT_TIMESTAMPS"], "timestamp")
        )
        columns[0] = columns[0].drop(columns=["timestamp_0"])

    if config["FT_DIRECTION"]:
        columns.append(unfold_array(direction_arr, config["FT_DIRECTION"], "direction"))

    if config["FT_PACKET_SIZE"]:
        packet_size = list(map(lambda x: np.array(x) * PACKET_SIZE, direction_arr))
        columns.append(
            unfold_array(packet_size, config["FT_PACKET_SIZE"], "packet_size")
        )

    if config["FT_BURST_DIR"] or config["FT_BURST_SIZE"]:
        burst = list(map(calculate_burst_pattern, direction_arr))
        if config["FT_BURST_DIR"]:
            columns.append(unfold_array(burst, config["FT_BURST_DIR"], "burst_dir"))
        if config["FT_BURST_SIZE"]:
            burst = list(map(lambda x: np.array(x) * PACKET_SIZE, burst))
            columns.append(unfold_array(burst, config["FT_BURST_SIZE"], "burst_size"))

    if config["FT_CUMULATIVE_DIR"] or config["FT_CUMULATIVE_SIZE"]:
        cumulative = list(map(lambda x: np.cumsum(x), direction_arr))
        if config["FT_CUMULATIVE_DIR"]:
            columns.append(
                unfold_array(
                    cumulative,
                    config["FT_CUMULATIVE_DIR"],
                    "cumulative_dir",
                )
            )
        if config["FT_CUMULATIVE_SIZE"]:
            cumulative = list(map(lambda x: x * PACKET_SIZE, cumulative))
            columns.append(
                unfold_array(
                    cumulative,
                    config["FT_CUMULATIVE_SIZE"],
                    "cumulative_size",
                )
            )

    if config["FT_ANALYSIS"]:
        df["count"] = list(map(lambda x: len(x), direction_arr))
        columns.append(analyze_direction(direction_arr, ""))

        for i in config["FT_ANALYSIS_SIZE"]:
            columns.append(
                analyze_direction(
                    list(map(lambda x: pad_zero_right(x, i), direction_arr)),
                    f"initial{i}_",
                )
            )

    return pd.concat([df, *columns], axis=1)


def parse_raw_data(monitored_path, unmonitored_path, dest="./data/original"):
    """
    Args:
        monitored_path (str): file path for monitored pkl file
        unmonitored_path (str): file path for unmonitored pkl file

    """
    mon_data = commonUtils.load_pickle_file(monitored_path)
    unmon_data = commonUtils.load_pickle_file(unmonitored_path)

    dataset, label = put_multi_label(mon_data, unmon_data)
    del mon_data, unmon_data

    timestamps = []
    direction = []

    for data in dataset:
        timestamps.append(abs(np.array(data)))
        direction.append(list(map(lambda x: 1 if x > 0 else -1, data)))

    del dataset

    if not (os.path.exists(dest)):
        os.makedirs(dest)

    commonUtils.save_pickle(timestamps, f"{dest}/timestamps.pkl")
    commonUtils.save_pickle(direction, f"{dest}/directions.pkl")
    commonUtils.save_pickle(label, f"{dest}/label.pkl")

    return


def process_data(config_path, path_to_save=None):
    """data processing

    Args:
        path_to_save (str)=None: path to save entire dataset. if None, don't save dataset

    Returns:
        dataframe, dataframe: train_set, test_set
    """
    with open(config_path) as f:
        config = json.load(f)

    try:
        timestamps = commonUtils.load_pickle_file("./data/original/timestamps.pkl")
        direction = commonUtils.load_pickle_file("./data/original/directions.pkl")
        label = commonUtils.load_pickle_file("./data/original/label.pkl")
    except:
        FileNotFoundError("parse data first with preprocessing.parse_raw_data")

    print("extracting features...")
    df = extract_features(direction, timestamps, config)
    print("Done.\n")

    del direction, timestamps

    df["label"] = label

    if path_to_save:
        commonUtils.save_pickle(df, path_to_save)

    return df


def filter_dataset(df=None, data_path=None, mode="OM", test_size=0.2, random_state=0):
    if data_path != None:
        df = commonUtils.load_pickle_file(data_path)
    elif df.empty:
        raise ValueError("data not given")

    data = None

    if mode == "OM":
        data = df
    elif mode == "CM":
        data = df[df["label"] >= 0]
    elif mode == "OB":
        df["label"] = df["label"].map(lambda x: 1 if x >= 0 else -1)
        data = df
    else:
        raise ValueError("mode should be one of 'OM', 'CM', or 'OB'")

    return train_test_split(
        data.drop(columns=["label"]),
        data["label"],
        test_size=test_size,
        random_state=random_state,
    )
