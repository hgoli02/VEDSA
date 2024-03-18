import os

import seaborn
from Datasets.cascade_preprocessor_seismic import *
import matplotlib.pyplot as plt
import sys



def read_flickr_cascades(verbose=False,
                       min_cascade_size=10):
    
    # This file contain a list of all of the favorite markings by users from the list above.  
    # The data includes the (anonymized) user who marked the photo as a favorite, the (anonymized) photo identifier, and the time of the favorite marking.

    # Format:  Gzipped ASCII.  Each line contains information about one photo being marked as a favorite, 
    # with fields separated by a tab. The data is (anonymized) user who marked the photo as a favorite,
    # (anonymized) photo identifier, and the (anonymized) time of the favorite marking.

    with open('./Datasets/flickr-all-photo-favorite-markings.txt', 'r') as cascades_file:

        all_tweets = {}

        for line in cascades_file:

            user, id, time = line.split()  # deleting enter from end of each line
            time = int(time)
            id = int(id)
            if id not in all_tweets:
                all_tweets[id] = []

            all_tweets[id].append(time)

        for id in list(all_tweets):
            if len(all_tweets[id]) >= min_cascade_size:
                all_tweets[id] = sorted(all_tweets[id])
            else:
                del all_tweets[id]

        for id in all_tweets:
            min = all_tweets[id][0]
            for i in range(len(all_tweets[id])):
                all_tweets[id][i] -= min

        cascades = []
        for id in all_tweets:
            tweets = []
            for time in all_tweets[id]:
                tweets.append(Tweet(id, time))
            cascades.append(Cascade(tweets))

        max = 0
        for key, value in all_tweets.items():
            if value[-1] - value[0]:
                max = value[-1] - value[0]

    if verbose:
        print('cascades num', len(cascades))

    return cascades


def read_digg_cascades(cascades_file_addr='../Datasets/digg_votes1.csv', verbose=False,
                       min_cascade_size=10):
    with open('./Datasets/digg_votes1.csv', 'r') as cascades_file:

        all_tweets = {}
        debug_counter = 0
        counter = 0
        for line in cascades_file:

            time, _, id = line.split(',')  # deleting enter from end of each line
            time = int(time[1:-1])
            id = int(id[1:-2])
            if id not in all_tweets:
                all_tweets[id] = []

            all_tweets[id].append(time)

        for id in all_tweets:
            if len(all_tweets[id]) >= min_cascade_size:
                all_tweets[id] = sorted(all_tweets[id])
            else:
                del all_tweets[id]

        for id in all_tweets:
            min = all_tweets[id][0]
            for i in range(len(all_tweets[id])):
                all_tweets[id][i] -= min

        cascades = []
        for id in all_tweets:
            tweets = []
            for time in all_tweets[id]:
                tweets.append(Tweet(id, time))
            cascades.append(Cascade(tweets))

        max = 0
        for key, value in all_tweets.items():
            if value[-1] - value[0]:
                max = value[-1] - value[0]

    if verbose:
        print('cascades num', len(cascades))

    return cascades

def read_wcascades(cascades_file_addr='datasets/weibo_cascades.txt'):
    with open(cascades_file_addr, 'r') as cascades_file:

        cascades = []

        min_cascade_size = 50

        counter = 0
        for line in cascades_file:
            counter = counter + 1

            line = line[0:len(line) - 1]  # deleting enter from end of each line
            split_line = line.split(' ')

            if len(split_line) < min_cascade_size:
                continue

            tweets = []
            for tweet in split_line:
                split_tweet = tweet.split(',')
                twitter = int(split_tweet[0])
                time = int(split_tweet[1])
                tweets.append(Tweet(twitter, time))

            tweets = sorted(tweets, key=lambda tweet: tweet.time)

            retweets_start = tweets[0].time
            for tweet in tweets:  # calculating relational times and normalizing time
                tweet.time -= retweets_start
            #     tweet.time /= 100  # sina weibo post propagate much slower than tweet times
            # cascades.append(Cascade(tweets))

            # important_tweets = []
            # retweets_start = tweets[0].time
            # for tweet in tweets:
            #     tweet.time -= retweets_start  # calculating relational times from start
            #     if tweet.time < 1000000:  # only considering tweets in about 10 days from start of propagation
            #         important_tweets.append(tweet)
            #
            # cascades.append(Cascade(important_tweets))

            cascade = Cascade(tweets)
            # if tweets[-1].time - tweets[0].time < 30 * 24 * 3600:
            #     # only adding cascades with duration of less than a month
            if tweets[-1].time - tweets[0].time < 60 * 24 * 3600:
                cascades.append(cascade)

    print('num cascade:', len(cascades))

    return cascades

def read_new_weibo_cascades(cascades_file_addr='../Datasets/', verbose=False,
                       min_cascade_size=10):
    with open('./Datasets/weibo_cascades.txt', 'r') as cascades_file:

        cascades = []

        counter = 0
        for line in cascades_file:
            counter = counter + 1

            line = line[0:len(line) - 1]  # deleting enter from end of each line
            split_line = line.split(' ')

            if len(split_line) < min_cascade_size:
                continue

            tweets = []
            for tweet in split_line:
                split_tweet = tweet.split(',')
                twitter = int(split_tweet[0])
                time = int(split_tweet[1])
                tweets.append(Tweet(twitter, time))

            tweets = sorted(tweets, key=lambda tweet: tweet.time)

            retweets_start = tweets[0].time
            for tweet in tweets:  # calculating relational times and normalizing time
                tweet.time -= retweets_start
            #     tweet.time /= 100  # sina weibo post propagate much slower than tweet times
            # cascades.append(Cascade(tweets))

            # important_tweets = []
            # retweets_start = tweets[0].time
            # for tweet in tweets:
            #     tweet.time -= retweets_start  # calculating relational times from start
            #     if tweet.time < 1000000:  # only considering tweets in about 10 days from start of propagation
            #         important_tweets.append(tweet)
            #
            # cascades.append(Cascade(important_tweets))

            cascade = Cascade(tweets)
            # if tweets[-1].time - tweets[0].time < 30 * 24 * 3600:
            #     # only adding cascades with duration of less than a month
            if tweets[-1].time - tweets[0].time < 60 * 24 * 3600:
                cascades.append(cascade)

    print('num cascade:', len(cascades))

    return cascades



def get_one_hot_burst_times_from_labels_with_linear_time_diminishing(cascade_burst_labels,
                                                                     prediction_time_windows=20,
                                                                     # first_bin_len=20):
                                                                     first_bin_len=50):
    len_burst_labels = len(cascade_burst_labels[0])
    num_burst_time_windows = prediction_time_windows - 1

    # f = first_bin_len, n = num_burst_time_windows, m = len_burst_labels => f + (f+x) + (f+2x) + ... + (f+(n-1)x) = m
    # => nf + [n(n-1)/2]x = m => x = (m-nf) / [n(n-1)/2]

    windows_time_difference = np.floor((len_burst_labels - num_burst_time_windows * first_bin_len)
                                       / (num_burst_time_windows * (num_burst_time_windows - 1) / 2))

    time_windows_lengths = []
    for i in range(num_burst_time_windows):
        time_windows_lengths.append(first_bin_len + i * windows_time_difference)

    divide_points = []
    for i in range(num_burst_time_windows):
        divide_points.append(sum(time_windows_lengths[:i]))

    divide_points.append(len_burst_labels)

    # print(len_burst_labels, num_burst_time_windows, windows_time_difference)
    # print(time_windows_lengths)
    # print(divide_points)

    # output:
    # 6048 19 34.0h
    # [6.0, 40.0, 74.0, 108.0, 142.0, 176.0, 210.0, 244.0, 278.0, 312.0, 346.0, 380.0, 414.0, 448.0, 482.0, 516.0,
    # 550.0, 584.0, 618.0]
    # [0, 6.0, 46.0, 120.0, 228.0, 370.0, 546.0, 756.0, 1000.0, 1278.0, 1590.0, 1936.0, 2316.0, 2730.0, 3178.0, 3660.0,
    # 4176.0, 4726.0, 5310.0, 6048]

    one_hots_times = []
    for burst_labels in cascade_burst_labels:
        one_hot_burst_time = np.zeros(prediction_time_windows)

        sum_burst_labels = sum(burst_labels)
        burst_time = len(burst_labels) - sum_burst_labels
        diminished_burst_time = -1
        for i in range(prediction_time_windows - 1):
            if divide_points[i] <= burst_time <= divide_points[i + 1]:
                diminished_burst_time = i
                break

        if sum_burst_labels == 0:  # non_burst cascade without burst time
            one_hot_burst_time[-1] = 1
        else:
            one_hot_burst_time[diminished_burst_time] = 1

        one_hots_times.append(one_hot_burst_time)

    return np.array(one_hots_times)


def get_burst_labels(windowed_cascades, burst_labels, burst_times):
    cascades_labels = []

    counter = 0
    for cascade in windowed_cascades:
        if burst_labels[counter]:
            cascades_labels.append(
                np.append(np.zeros(burst_times[counter]),
                          np.ones(len(cascade) - burst_times[counter]))
            )

        else:
            cascades_labels.append(np.zeros(len(cascade)))
        counter += 1

    return cascades_labels


def cascades_to_proper_survival_model_input_lite(burst_min_len=1000, non_burst_max_len=900, cascades_min_len=50,
                                                 time_bin_len=100,
                                                 index_file_addr='../seismic_dataset/index.csv',
                                                 data_file_addr='../seismic_dataset/data.csv',
                                                 dataset_flag=1
                                                 ):
    if dataset_flag == 1:
        cascades = read_cascades(index_file_addr=index_file_addr, data_file_addr=data_file_addr)
    elif dataset_flag == 2:
        cascades = read_digg_cascades(verbose=False,
                                      min_cascade_size=10)
    elif dataset_flag == 3:
        cascades = read_flickr_cascades(verbose=False,
                                      min_cascade_size=10)
    elif dataset_flag == 4:
        cascades = read_wcascades()

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))
    print('selected cascades num', len(selected_cascades))

    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    # windowed_cascades = create_time_windows(selected_cascades)
    windowed_cascades = create_time_windows(selected_cascades, time_window_len=time_bin_len)
    burst_times = get_burst_time(windowed_cascades, burst_min_len)
    # peak_times = get_peak_windows(windowed_cascades)
    # peak_times = get_burst_threshold_times(windowed_cascades)

    max_len = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)  # bin:10s -> answer: 60480

    # burst_max_len = 0
    # for windowed_cascade in create_time_windows(burst_cascades):
    #     if len(windowed_cascade) > burst_max_len:
    #         burst_max_len = len(windowed_cascade)
    #
    # print('burst_max_len is ', burst_max_len)

    for i in range(len(windowed_cascades)):
        windowed_cascades[i] += np.zeros(max_len - len(windowed_cascades[i])).tolist()
        windowed_cascades[i] = np.array(windowed_cascades[i])

    cascade_burst_labels = get_burst_labels(windowed_cascades, labels, burst_times)

    return windowed_cascades, cascade_burst_labels, labels


def make_cascades_burst_location_distribution_uniform_linear_diminishing(burst_min_len=1000, non_burst_max_len=900,
                                                                         cascades_min_len=50, burst_bins_num=20,
                                                                         max_cascade_in_each_bin=500,
                                                                         time_bin_len=100,
                                                                         max_non_burst_twice_the_max_burst_in_bin=True,
                                                                         index_file_addr='../seismic_dataset/index.csv',
                                                                         data_file_addr='../seismic_dataset/data.csv',
                                                                         dataset_flag = 1
                                                                         ):
    windowed_cascades, cascade_burst_labels, labels = cascades_to_proper_survival_model_input_lite(burst_min_len,
                                                                                                   non_burst_max_len,
                                                                                                   cascades_min_len,
                                                                                                   time_bin_len,
                                                                                                   index_file_addr=
                                                                                                   index_file_addr,
                                                                                                   data_file_addr=
                                                                                                   data_file_addr,
                                                                                                   dataset_flag=dataset_flag)

    windowed_cascades_burst = []
    windowed_cascades_non_burst = []
    cascade_burst_labels_burst = []
    cascade_burst_labels_non_burst = []
    labels_burst = []
    labels_non_burst = []

    for i in range(len(labels)):
        if labels[i] == 1:
            windowed_cascades_burst.append(windowed_cascades[i])
            cascade_burst_labels_burst.append(cascade_burst_labels[i])
            labels_burst.append(labels[i])
        else:
            windowed_cascades_non_burst.append(windowed_cascades[i])
            cascade_burst_labels_non_burst.append(cascade_burst_labels[i])
            labels_non_burst.append(labels[i])

    windowed_cascades_burst_new = []
    cascade_burst_labels_burst_new = []
    labels_burst_new = []

    simplified_cascade_burst_labels_burst = np.argmax(
        get_one_hot_burst_times_from_labels_with_linear_time_diminishing
        (cascade_burst_labels=cascade_burst_labels_burst, prediction_time_windows=burst_bins_num), 1)

    burst_dist = np.zeros(burst_bins_num)
    i = 0
    for burst_bin in simplified_cascade_burst_labels_burst:
        # print('bin -> ', burst_bin)

        if burst_dist[burst_bin] < max_cascade_in_each_bin:
            windowed_cascades_burst_new.append(windowed_cascades_burst[i])
            cascade_burst_labels_burst_new.append(cascade_burst_labels_burst[i])
            labels_burst_new.append(labels_burst[i])
            burst_dist[burst_bin] += 1

        i += 1

    for j in range(burst_bins_num):
        print("bin: " + str(j) + ", " + "number_of_burst_time_in_this_bin: " + str(burst_dist[j]))
    # plot_number_of_burst_time_in_each_bin(burst_dist)
    # print('burst_dist after uniformization:', list(burst_dist))
    # remaining_burst_cascades_num = len(windowed_cascades_burst_new)
    # windowed_cascades_non_burst = windowed_cascades_non_burst[:remaining_burst_cascades_num]
    # cascade_burst_labels_non_burst = cascade_burst_labels_non_burst[:remaining_burst_cascades_num]
    # labels_non_burst = labels_non_burst[:remaining_burst_cascades_num]

    non_burst_length = len(windowed_cascades_burst_new)
    windowed_cascades_non_burst = windowed_cascades_non_burst[:non_burst_length]
    cascade_burst_labels_non_burst = cascade_burst_labels_non_burst[:non_burst_length]
    labels_non_burst = labels_non_burst[:non_burst_length]
    # if max_non_burst_twice_the_max_burst_in_bin:
    #     windowed_cascades_non_burst = windowed_cascades_non_burst[:2 * max_cascade_in_each_bin]
    #     cascade_burst_labels_non_burst = cascade_burst_labels_non_burst[:2 * max_cascade_in_each_bin]
    #     labels_non_burst = labels_non_burst[:2 * max_cascade_in_each_bin]
    # else:
    #     windowed_cascades_non_burst = windowed_cascades_non_burst[:max_cascade_in_each_bin]
    #     cascade_burst_labels_non_burst = cascade_burst_labels_non_burst[:max_cascade_in_each_bin]
    #     labels_non_burst = labels_non_burst[:max_cascade_in_each_bin]

    windowed_cascades_new = np.array(windowed_cascades_burst_new + windowed_cascades_non_burst)
    cascade_burst_labels_new = np.array(cascade_burst_labels_burst_new + cascade_burst_labels_non_burst)
    labels_new = np.array(labels_burst_new + labels_non_burst)

    windowed_cascades_new, cascade_burst_labels_new, labels_new = shuffle(windowed_cascades_new,
                                                                          cascade_burst_labels_new, labels_new,
                                                                          random_state=0)
    # todo: remove random_state=0

    return windowed_cascades_new, cascade_burst_labels_new, labels_new


def uniformed_cascades_to_proper_survival_model_input_test_split_linear_diminishing(
        burst_min_len=200,
        non_burst_max_len=100,
        cascades_min_len=50,
        burst_bins_num=20,
        max_cascade_in_each_bin=500,
        test_size=0.2,
        time_bin_len=100,
        max_non_burst_twice_the_max_burst_in_bin=True,
        index_file_addr='../seismic_dataset/index.csv',
        data_file_addr='../seismic_dataset/data.csv',
        dataset_flag = 3):
    windowed_cascades, cascade_burst_labels, labels = make_cascades_burst_location_distribution_uniform_linear_diminishing(
        burst_min_len,
        non_burst_max_len,
        cascades_min_len,
        burst_bins_num,
        max_cascade_in_each_bin,
        time_bin_len,
        max_non_burst_twice_the_max_burst_in_bin=max_non_burst_twice_the_max_burst_in_bin,
        index_file_addr=index_file_addr,
        data_file_addr=data_file_addr, 
        dataset_flag=dataset_flag)

    print('uniformed_cascades_to_proper_survival_model_input_test_split')
    print(len(windowed_cascades), 'cascades')

    return train_test_split(windowed_cascades, cascade_burst_labels, labels, test_size=test_size, random_state=0,
                            shuffle=False)  # todo: remove this after finishing working with seed 0


def create_edrn_input(selected_cascades):
    input = []
    for cascade in selected_cascades:
        tweet_intervals = []
        for i, tweet in enumerate(cascade.tweets):
            if i + 1 == len(cascade.tweets):
                next = tweet
            else:
                next = cascade.tweets[i+1]
            
            interval = next.time - tweet.time

            tweet_intervals.append(interval)

        
        input.append(tweet_intervals)

    return np.array(input, dtype=object)

def EDRN_preproccess(
        burst_min_len=200,
        non_burst_max_len=100,
        cascades_min_len=50,
        burst_bins_num=20,
        max_cascade_in_each_bin=500,
        test_size=0.2,
        time_bin_len=100,
        max_non_burst_twice_the_max_burst_in_bin=True,
        index_file_addr='../seismic_dataset/index.csv',
        data_file_addr='../seismic_dataset/data.csv',
        dataset_flag = 1):
    

    if dataset_flag == 1:
        cascades = read_cascades(index_file_addr=index_file_addr, data_file_addr=data_file_addr)
    elif dataset_flag == 2:
        cascades = read_digg_cascades(verbose=False,
                                      min_cascade_size=10)
    elif dataset_flag == 3:
        cascades = read_flickr_cascades(verbose=False,
                                      min_cascade_size=10)

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))
    print('selected cascades num', len(selected_cascades))

    windowed_cascades = create_edrn_input(selected_cascades) 

    max_len = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)  # bin:10s -> answer: 60480


    return train_test_split(windowed_cascades, labels, labels, test_size=test_size, random_state=0,
                            shuffle=False)  # todo: remove this after finishing working with seed 0

