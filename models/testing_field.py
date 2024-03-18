from Datasets.cascade_preprocessor_seismic_survival import \
    uniformed_cascades_to_proper_survival_model_input_test_split_linear_diminishing as \
    cascades_to_proper_survival_model_input_test_split, get_one_hot_peak_times_from_labels_with_linear_time_diminishing
import numpy as np

index_file_addr = '../Datasets/index.csv'
data_file_addr = '../Datasets/data.csv'
peak_bins_num = 20
max_cascade_in_each_bin=1500
time_bin_len=400
max_non_burst_twice_the_max_burst_in_bin=True
prediction_time_windows_num = 20



train_cascades, test_cascades, train_spike_labels, \
test_spike_labels, train_burst_labels, test_burst_labels = cascades_to_proper_survival_model_input_test_split(
    index_file_addr=index_file_addr, data_file_addr=data_file_addr,
    peak_bins_num=peak_bins_num,
    max_cascade_in_each_bin=max_cascade_in_each_bin,
    time_bin_len=time_bin_len,
    max_non_burst_twice_the_max_burst_in_bin=max_non_burst_twice_the_max_burst_in_bin)


train_spike_times = get_one_hot_peak_times_from_labels_with_linear_time_diminishing(
    train_spike_labels,
    prediction_time_windows=prediction_time_windows_num)

print(train_spike_times[0])
#print percentage of burst vs non-burst
print("train_cascades shape:", train_cascades.shape)
print("test_cascades shape:", test_cascades.shape)
print("train_spike_labels shape:", train_spike_labels.shape)
print("test_spike_labels shape:", test_spike_labels.shape)
print("train_burst_labels shape:", train_burst_labels.shape)
print("test_burst_labels shape:", test_burst_labels.shape)

#print percentage of burst labels
print("train burst percentage:", np.sum(train_burst_labels) / len(train_burst_labels))
print("test burst percentage:", np.sum(test_burst_labels) / len(test_burst_labels))
