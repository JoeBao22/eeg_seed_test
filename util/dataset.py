import numpy as np
import os


def load_mat(dirname):
    feature_arr = np.load(os.path.join(dirname, "feature.npy"))
    label_arr = np.load(os.path.join(dirname, "label.npy"))
    cumulative_arr = np.load(os.path.join(dirname, "cumulative.npy"))
    return feature_arr, label_arr, cumulative_arr
    

def train_test_split(feature_arr, label_arr, cumulative_arr, train_sample=9):
    train_trials = train_sample * 3 * 15
    num_of_train = cumulative_arr[train_trials]
    train_arr = feature_arr[:num_of_train]
    train_label = label_arr[:num_of_train]
    test_arr = feature_arr[num_of_train:]
    test_label = label_arr[num_of_train:]
    return train_arr, train_label, test_arr, test_label
    
if __name__ == "__main__":
    temp_dir = "../dataset/SEED/ExtractedFeatures/de_LDS"
    feature, label, cumulative = load_mat(temp_dir)
    train_arr, train_label, test_arr, test_label = train_test_split(feature, label, cumulative)
    np.save(os.path.join(temp_dir, "train_feature.npy"), train_arr)
    np.save(os.path.join(temp_dir, "train_label.npy"), train_label)
    np.save(os.path.join(temp_dir, "test_feature.npy"), test_arr)
    np.save(os.path.join(temp_dir, "test_label.npy"), test_label)
