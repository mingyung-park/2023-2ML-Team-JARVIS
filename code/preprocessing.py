import pickle
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split


#####################################################
SITE_CNT = 95
URL_PER_SITE = 10
CNT_PER_URL = 20

# data 선택
FT_UNMON = True
FT_MON = True

# label type - False 인 경우 monitored(1), unmonitored(-1)
FT_MULTI_LABEL = True

PACKET_SIZE = 512

# analysis
FT_ANALYSIS = True
FT_ANALYSIS_SIZE = range(
    10, 60, 10
)  # 10부터 10씩 늘려감, 60보다 작을 때까지. -> (10, 20, 30, 40, 50)

# feature 선택
FT_SEQ_SIZE = 50
# continuous
FT_TIMESTAMPS = True

FT_DIRECTION = True  # 방향만 표기 [1. -1, 1, 1, ...]
FT_PACKET_SIZE = False  # packet의 size로 표기 [512, -512, 512, 512, ...]

# busrt와 cumulative의 경우, Packet size를 반영하여 계산할 수도 있지만 Tor에서는 모든 패킷이 512 단위이므로, 그냥 scale을 줄여서 사용할 수도 있을거 같습니다.
# (저희에게 주어진 데이터의 경우 모두 512이므로 그냥 방향만으로 계산한다고 생각해도 무방)
FT_BURST_DIR = True  # [1, -1, 1, 2, ...]
FT_BURST_SIZE = False  # burst_dir에 packet size를 반영한 것 (burst_dir * 512)
FT_CUMULATIVE_DIR = True  # [1, 0, 1, 2, ...]
FT_CUMULATIVE_SIZE = False  # cumulative dir에 packet size를 반영한 것 (cumulative_dir * 512)

seqeunce_features_size = {
    "timestamps": FT_SEQ_SIZE,
    "direction": FT_SEQ_SIZE,
    "packet_size": FT_SEQ_SIZE,
    "burst_dir": FT_SEQ_SIZE,
    "burst_size": FT_SEQ_SIZE,
    "cumulative_dir": FT_SEQ_SIZE,
    "cumulative_size": FT_SEQ_SIZE,
}
##########################################################


def load_pickle_file(file_path):
    """load pickle file

    Args:
        file_path (string): absolute path of pickle file

    Returns:
        list: data from pickle
    """
    print("Loading datafile...")
    with open(file_path, "rb") as fi:
        data = pickle.load(fi)

    print("Done.")
    return data


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
    print("parsing dataset...")
    dataset = []
    label = []

    if mon_data:
        if FT_MULTI_LABEL:
            label = [i for i in range(95) for _ in range(200)]
        else:
            label = [1] * SITE_CNT * URL_PER_SITE * CNT_PER_URL

        for i in range(SITE_CNT):
            temp = []
            for j in range(URL_PER_SITE):
                temp.append(mon_data[i * URL_PER_SITE + j])
            dataset.append(list(chain.from_iterable(temp)))

    if unmon_data:
        label.extend([-1] * 10000)
        dataset.append(unmon_data)

    dataset = list(chain.from_iterable(dataset))
    print("Done.")
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


def extract_features(direction_arr, timestamps_arr):
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

    if FT_TIMESTAMPS:
        columns.append(
            unfold_array(
                timestamps_arr, seqeunce_features_size["timestamps"], "timestamp"
            )
        )
        columns[0] = columns[0].drop(columns=["timestamp_0"])

    if FT_DIRECTION:
        columns.append(
            unfold_array(
                direction_arr, seqeunce_features_size["direction"], "direction"
            )
        )

    if FT_PACKET_SIZE:
        packet_size = list(map(lambda x: np.array(x) * PACKET_SIZE, direction_arr))
        columns.append(
            unfold_array(
                packet_size, seqeunce_features_size["packet_size"], "packet_size"
            )
        )

    if FT_BURST_DIR or FT_BURST_SIZE:
        burst = list(map(calculate_burst_pattern, direction_arr))
        if FT_BURST_DIR:
            columns.append(
                unfold_array(burst, seqeunce_features_size["burst_dir"], "burst_dir")
            )
        if FT_BURST_SIZE:
            burst = list(map(lambda x: np.array(x) * PACKET_SIZE, burst))
            columns.append(
                unfold_array(burst, seqeunce_features_size["burst_size"], "burst_size")
            )

    if FT_CUMULATIVE_DIR or FT_CUMULATIVE_SIZE:
        cumulative = list(map(lambda x: np.cumsum(x), direction_arr))
        if FT_CUMULATIVE_DIR:
            columns.append(
                unfold_array(
                    cumulative,
                    seqeunce_features_size["cumulative_dir"],
                    "cumulative_dir",
                )
            )
        if FT_CUMULATIVE_SIZE:
            cumulative = list(map(lambda x: x * PACKET_SIZE, cumulative))
            columns.append(
                unfold_array(
                    cumulative,
                    seqeunce_features_size["cumulative_size"],
                    "cumulative_size",
                )
            )

    if FT_ANALYSIS:
        df["count"] = list(map(lambda x: len(x), direction_arr))
        columns.append(analyze_direction(direction_arr, ""))

        for i in FT_ANALYSIS_SIZE:
            columns.append(
                analyze_direction(
                    list(map(lambda x: pad_zero_right(x, i), direction_arr)),
                    f"initial{i}_",
                )
            )

    return pd.concat([df, *columns], axis=1)


def save_dataset(data, file_path):
    """save as pikle"""

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return


def process_raw_data(monitored_path, unmonitored_path, save_data):
    """data processing

    Args:
        monitored_path (str): file path for monitored pkl file
        unmonitored_path (str): file path for unmonitored pkl file
        save_data (bool): if False, return dataset which contains monitored and unmonitored data with multi label

    Returns:
        dataframe, dataframe: train_set, test_set
    """
    if FT_MON:
        mon_data = load_pickle_file(monitored_path)
    else:
        mon_data = None
    if FT_UNMON:
        unmon_data = load_pickle_file(unmonitored_path)
    else:
        unmon_data = None

    dataset, label = put_multi_label(mon_data, unmon_data)
    del mon_data, unmon_data

    timestamps = []
    direction = []

    for data in dataset:
        timestamps.append(abs(np.array(data)))
        direction.append(list(map(lambda x: 1 if x > 0 else -1, data)))

    del dataset

    df = extract_features(direction, timestamps)
    del direction, timestamps

    df["label"] = label

    if save_data:
        save_dataset(df, "./data/dataset.pkl")
    return df


def split_dataset(df):
    """
    /data
    ㄴ train
        ㄴ closed_multi.pkl
        ㄴ open_binary.pkl
        ㄴ open_multi.pkl
    ㄴ test
        ㄴ closed_multi.pkl
        ㄴ open_binary.pkl
        ㄴ open_multi.pkl

        f"./data/train/closed_multi.pkl"
    """
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)

    prefix = "./data"
    save_dataset(train_set, f"{prefix}/train/open_multi.pkl")
    save_dataset(train_set[train_set["label"] >= 0], f"{prefix}/train/closed_multi.pkl")
    train_set["label"] = train_set["label"].map(lambda x: 1 if x >= 0 else -1)
    save_dataset(train_set, f"{prefix}/train/open_binary.pkl")

    save_dataset(test_set, f"{prefix}/test/open_multi.pkl")
    save_dataset(test_set[test_set["label"] >= 0], f"{prefix}/test/closed_multi.pkl")
    test_set["label"] = test_set["label"].map(lambda x: 1 if x >= 0 else -1)
    save_dataset(test_set, f"{prefix}/test/open_binary.pkl")

    return
