import pickle


def load_pickle_file(file_path):
    """load pickle file

    Args:
        file_path (string): absolute path of pickle file

    Returns:
        list: data from pickle
    """
    print(f"Loading datafile... : {file_path}")
    with open(file_path, "rb") as fi:
        data = pickle.load(fi)

    print("Done.\n")
    return data


def save_pickle(data, file_path):
    """save as pikle"""

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return
