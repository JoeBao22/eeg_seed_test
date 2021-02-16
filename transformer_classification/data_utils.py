from typing import Iterable, Union, List
import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
from sklearn import preprocessing
import random
    

def create_examples(args,
                    mode: str = 'train') -> Iterable[Union[List, dict]]:
    dataset_dir = args.dataset_dir
    features, labels, cumulative = load_mat(dataset_dir)
    
    labels += 1  # deal with offset (original labels: -1, 0, 1 => new labels: 0, 1, 2)

    train_arr, train_label, test_arr, test_label = train_test_split(
        features, labels, cumulative, args.train_percentage, args.sample_per_input, args.normalization)
    
    train_tensor = torch.tensor(train_arr, dtype=torch.float)
    train_label = torch.tensor(train_label, dtype=torch.int64).squeeze_(1)
    test_tensor = torch.tensor(test_arr, dtype=torch.float)
    test_label = torch.tensor(test_label, dtype=torch.int64).squeeze_(1)
    
    train_dataset = TensorDataset(train_tensor, train_label)
    test_dataset = TensorDataset(test_tensor, test_label)

    return train_dataset, test_dataset


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
     

def load_mat(dirname):
    feature_arr = np.load(os.path.join(dirname, "feature.npy"))
    label_arr = np.load(os.path.join(dirname, "label.npy"))
    cumulative_arr = np.load(os.path.join(dirname, "cumulative.npy"))
    return feature_arr, label_arr, cumulative_arr
    

def train_test_split(feature_arr, label_arr, cumulative_arr, train_percentage, group_by, normalization=False):
    seq_len = feature_arr.shape[1]
    train_feature = None
    train_label = None
    test_feature = None
    test_label = None
    for person_index in range(15):
        for day_index in range(3):
            train_trial_num = int(train_percentage * 15) 
            for trial_index in range(15):
                start_index = cumulative_arr[(person_index * 3 + day_index) * 15 + trial_index]
                end_index = cumulative_arr[(person_index * 3 + day_index) * 15 + trial_index + 1]
                remainder = (end_index - start_index) % group_by
                end_index -=  remainder  # truncate
                num_of_groups = (end_index - start_index) // group_by
                temp_arr = feature_arr[start_index: end_index].reshape(num_of_groups, group_by, seq_len)
                temp_label = label_arr[start_index: end_index: group_by]
                if trial_index < train_trial_num:
                    if train_feature is None:
                        train_feature = temp_arr
                        train_label = temp_label
                    else:
                        train_feature = np.vstack((train_feature, temp_arr))
                        train_label = np.vstack((train_label, temp_label))
                else:
                    if test_feature is None:
                        test_feature = temp_arr
                        test_label = temp_label
                    else:
                        test_feature = np.vstack((test_feature, temp_arr))
                        test_label = np.vstack((test_label, temp_label))
    vector_len = train_feature.shape[-1]
    train_feature_2d = train_feature.reshape(-1, vector_len)
    test_feature_2d = test_feature.reshape(-1, vector_len)
    stdScale = preprocessing.StandardScaler().fit(train_feature_2d)
    train_feature_2d = stdScale.transform(train_feature_2d)
    test_feature_2d = stdScale.transform(test_feature_2d)
    train_feature = train_feature_2d.reshape((-1, group_by, vector_len))
    test_feature = test_feature_2d.reshape(-1, group_by, vector_len)
    return train_feature, train_label, test_feature, test_label