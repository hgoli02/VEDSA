import argparse
from Datasets.new2_cascade_preprocessor_seismic_survival import \
    uniformed_cascades_to_proper_survival_model_input_test_split_linear_diminishing as \
    cascades_to_proper_survival_model_input_test_split, get_one_hot_burst_times_from_labels_with_linear_time_diminishing
from models.recurrent_model import RecurrentModel, loss_function
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from models.predictor import predictor

from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

time_bin_len = 400
bins_num = 10
max_cascade_in_each_bin=100000
max_non_burst_twice_the_max_burst_in_bin=True
test_size = 0.2
index_file_addr = 'Datasets/index.csv'
data_file_addr = 'Datasets/data.csv'


parser = argparse.ArgumentParser(description='Process some integers.')

#dataset
parser.add_argument('--dataset', type=str, default='twitter', help='twitter | dig | weibo')
parser.add_argument('--hours', type=int, default=1, help='hours')
parser.add_argument('--burstminlen', type=int, default=800, help='bminlen')
parser.add_argument('--nonburstmaxlen', type=int, default=600, help='bmaxlen')

dataset = parser.parse_args().dataset
hours = parser.parse_args().hours
burst_min_len = parser.parse_args().burstminlen
non_burst_max_len = parser.parse_args().nonburstmaxlen

if dataset == 'twitter':
    flag = 1
    CTIME = 24 * 7
elif dataset == 'dig':
    flag = 2
    CTIME = 24 * 30
elif dataset == 'weibo':
    flag = 3
    CTIME = 24 * 60

DATA_PERCENTAGE = (hours) / CTIME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_cascades_read, test_cascades_read, train_spike_labels, \
test_spike_labels, train_burst_labels, test_burst_labels = cascades_to_proper_survival_model_input_test_split(
    index_file_addr=index_file_addr, data_file_addr=data_file_addr,
    burst_bins_num=bins_num,
    max_cascade_in_each_bin=max_cascade_in_each_bin,
    time_bin_len=time_bin_len,
    max_non_burst_twice_the_max_burst_in_bin=max_non_burst_twice_the_max_burst_in_bin,
    burst_min_len = burst_min_len,
    non_burst_max_len = non_burst_max_len,
    test_size = 0.2,
    dataset_flag = flag)


train_cascades = train_cascades_read[:, :int(train_cascades_read.shape[1] * DATA_PERCENTAGE)]
test_cascades = test_cascades_read[:, :int(test_cascades_read.shape[1] * DATA_PERCENTAGE)]
max_len = len(train_cascades_read[0])
print(f"max len = {max_len}")

def get_dataloader(cascades, spike_labels, burst_labels, batch_size=128):
    dataset = TensorDataset(torch.from_numpy(cascades).float(),
                            torch.from_numpy(spike_labels).float(),
                            torch.from_numpy(burst_labels).float())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def to_weibull_survival_function(lambdas, k, times):
    return np.exp(-np.power(times / lambdas, k))

def to_weibull_survival_function_tensor(lambdas, k, times):
    return torch.exp(-torch.pow(times / lambdas, k))



dataloader = get_dataloader(train_cascades, train_spike_labels, train_burst_labels)
model = RecurrentModel(1, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = loss_function

model.train()
for epoch in range(800):
    total_loss = 0

    for cascades, spike_labels, burst_labels in dataloader:
        cascades = cascades.to(device)
        spike_labels = spike_labels.to(device)
        burst_labels = burst_labels.to(device)
        hidden = model.init_hidden(cascades.shape[0], device)
        optimizer.zero_grad()
        lambdas, k = model(cascades, hidden)
        times = torch.arange(1, max_len + 1).float().to(device)
        loss = criterion(lambdas, k, spike_labels, times)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch: {}, Loss: {}".format(epoch, total_loss))


model.eval()
second_model = predictor(max_len).to(device)
second_optimizer = torch.optim.Adam(second_model.parameters(), lr=0.0001)
second_criterion = torch.nn.BCEWithLogitsLoss()

second_model.eval()
test_case = train_cascades
label_train_case = train_burst_labels
test_case = torch.from_numpy(test_case).float().to(device)
with torch.no_grad():
    hidden = model.init_hidden(test_case.shape[0], device)
    lambdas, k = model(test_case, hidden)
    times = torch.arange(1, max_len + 1).float().to(device)
    # convert to numpy
    lambdas = lambdas.cpu().numpy()
    k = k.cpu().numpy()
    times = times.cpu().numpy()
    test_case = test_case.detach().cpu().numpy()
    survival_function_train = []
    for i in range(len(lambdas)):
        survival_function_train.append(to_weibull_survival_function(lambdas[i], k[i], times))

survival_function_train = np.array(survival_function_train)
survival_function_train = torch.Tensor(survival_function_train)

train_spike_times = get_one_hot_burst_times_from_labels_with_linear_time_diminishing(
            train_spike_labels,
            prediction_time_windows=bins_num)

test_spike_times = get_one_hot_burst_times_from_labels_with_linear_time_diminishing(
            test_spike_labels,
            prediction_time_windows=bins_num)

second_model = predictor(max_len, 1).to(device)
second_optimizer = torch.optim.Adam(second_model.parameters(), lr=0.0001)
second_criterion = torch.nn.BCEWithLogitsLoss()
dataset = TensorDataset(survival_function_train, torch.from_numpy(train_spike_times).float(), torch.from_numpy(train_burst_labels).float())
second_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


second_model.train()
best_model = None
max_burst_correct = 0
best_acc = 0
for epoch in range(100):
    total_loss = 0
    total_count = 0
    total_correct = 0
    burst_correct = 0
    total_bursts = 0
    total_non_bursts = 0
    for survival_function, train_spike_time, burst_label in second_dataloader:
        survival_function = survival_function.to(device)
        train_spike_time = train_spike_time.to(device)
        burst_label = burst_label.to(device)
        second_optimizer.zero_grad()
        survival_function = survival_function.view(survival_function.shape[0], 1, -1)
        output = second_model(survival_function)
        total_correct += torch.count_nonzero((output > 0) == burst_label.view(-1, 1))
        loss_second = second_criterion(output,burst_label.view(-1, 1))
        loss_second.backward()
        second_optimizer.step()
        total_loss += loss_second.item()
        total_count += len(burst_label)
        total_bursts += (burst_label == 1).sum().item()
        total_non_bursts += (burst_label == 0).sum().item()
        #print(torch.count_nonzero((output > 0) == burst_label.view(-1, 1)))
    if (total_correct / total_count > best_acc):
        print(f"updating best model from {best_acc} to {total_correct / total_count}")
        best_model = second_model.state_dict()
        best_acc = total_correct / total_count
    print("Epoch: {}, Loss: {}".format(epoch, total_loss))
    print("Accuracy on bursts: {}".format(total_correct / total_count))
    print("Total bursts: {}, Total non bursts: {}".format(total_bursts, total_non_bursts))


second_model.eval()
test_case = test_cascades
label_train_case = test_burst_labels
test_case = torch.from_numpy(test_case).float().to(device)

with torch.no_grad():
    hidden = model.init_hidden(test_case.shape[0], device)
    lambdas, k = model(test_case, hidden)
    times = torch.arange(1, max_len + 1).float().to(device)
    # convert to numpy
    lambdas = lambdas.cpu().numpy()
    k = k.cpu().numpy()
    times = times.cpu().numpy()
    test_case = test_case.detach().cpu().numpy()
    survival_function_test = []
    for i in range(len(lambdas)):
        survival_function_test.append(to_weibull_survival_function(lambdas[i], k[i], times))

survival_function_test = torch.Tensor(np.array(survival_function_test))
survival_function_test = survival_function_test.view(survival_function_test.shape[0], 1, -1)
survival_function_test = survival_function_test.to(device)

torch.save(best_model, "second_model_state_dict.pt")
second_model.load_state_dict(torch.load("second_model_state_dict.pt"))
output = second_model(survival_function_test)

output_t_f = (output > 0).cpu().numpy()
print("Accuracy Burst: {}".format(accuracy_score(test_burst_labels, output_t_f)))
print("F1 score Burst: {}".format(f1_score(test_burst_labels, output_t_f)))
print("Precison - Recall - Fscore")
print(precision_recall_fscore_support(test_burst_labels, output_t_f, average='binary'))
print(classification_report(test_burst_labels, output_t_f, target_names=['non burst', 'burst'], digits=4))
