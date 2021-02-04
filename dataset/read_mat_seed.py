import scipy.io as sio 
import numpy as np
import os


dir_path = "./SEED/ExtractedFeatures"
feature_types = ["de", "psd", "dasm", "rasm", "asm", "dcau"]
smooth_method_types = ["movingAve", "LDS"]
# get labels:
label_path = os.path.join(dir_path, "label.mat")
labels = sio.loadmat(label_path)["label"][0]

num_of_people = 15
num_of_experiment = 15
for feature_type in feature_types:
    for smooth_method_type in smooth_method_types:
        folder_name = os.path.join(dir_path, feature_type +"_" + smooth_method_type)
        print("folder name: ", folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        stacked_arr = None
        stacked_label = None
        cumulative_samples = [0]
        for i in range(num_of_people):
            print("resolving person {} / {}".format(i+1, num_of_people))
            for trial_path in os.listdir(dir_path):
                if trial_path.startswith(str(i+1) + "_"):  # trial record for the person
                    feature2dict= sio.loadmat(os.path.join(dir_path, trial_path))
                    for experiment_index in range(num_of_experiment):
                        k = feature_type + "_" + smooth_method_type + str(experiment_index+1)
                        v = feature2dict[k]
                        # print(v.shape)  # (62, 235, 5) for example
                        temp_arr = np.swapaxes(v, 0, 1).reshape(v.shape[1], -1)    
                        num_of_samples = temp_arr.shape[0]   
                        cumulative_samples.append(cumulative_samples[-1] + num_of_samples)
                        temp_labels = np.ones((num_of_samples, 1)) * labels[experiment_index]          
                        # print(temp_arr.shape)  # (235, 310) for example
                        if stacked_arr is None:
                            stacked_arr = temp_arr.copy()
                            stacked_label = temp_labels.copy()
                        else:
                            stacked_arr = np.vstack((stacked_arr, temp_arr))  # vertically stack arrays
                            stacked_label = np.vstack((stacked_label, temp_labels))
        print("feature shape:", stacked_arr.shape)
        print("label shape: ", stacked_label.shape)
        cumulative_sample_arr = np.array(cumulative_samples)
        print("cumulative sample shape: ", cumulative_sample_arr.shape)
        feature_path = os.path.join(folder_name, "feature.npy")
        label_path = os.path.join(folder_name, "label.npy")
        cumulative_samples_path = os.path.join(folder_name, "cumulative.npy")
        print("saving feature.npy and label.npy to folder {}".format(folder_name))
        np.save(feature_path, stacked_arr)
        np.save(label_path, stacked_label)
        np.save(cumulative_samples_path, cumulative_sample_arr)
