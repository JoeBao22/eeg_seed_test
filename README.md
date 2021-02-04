# EEG SEED TEST

The repository contains trials that apply model transformer to the emotion recognition (classification) task based on electroencephalography(EEG).

## experiment 1: transformer for DE feature

### dataset
We use the differential entropy(DE) feature from [SEED Dataset](http://bcmi.sjtu.edu.cn/home/seed/seed.html) for the first phase of test. In the data preprocessing part, we generate a feature array(feature.npy) and its corresponding label array(label.npy) for each pair of (chosen feature,  average method). The splitting points are written to cumulative array (cumulative.npy).

The Encoder part of transformer is used for classification. We remove the word embedding layer, because our input is already 310 * samples_per_input.
To do: We'll adjust the hyper-parameters of transformer for better classification results, and we'll also modify the data_util part, so that the task is subject-dependent.
