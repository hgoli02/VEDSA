# this code's task is to provide the information and pre processes needed for the deep model

import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import ceil
import matplotlib.pyplot as plt


class Tweet:
    def __init__(self, followers_num, time):
        self.followers_num = followers_num
        self.time = time

    def print(self):
        print(self.time, self.followers_num, sep=',', end=' ')

    def write_in_file(self, file):
        file.write(str(self.followers_num) + ',' + str(self.time) + ' ')


class Cascade:
    def __init__(self, tweets):
        self.tweets = tweets

    def sort_tweets(self):
        self.tweets = sorted(self.tweets, key=lambda tweet: tweet.time)

    def get_tweet_times(self):
        l = []
        for tweet in self.tweets[0:len(self.tweets)]:
            l.append(tweet.time)
        return l

    def print(self):
        for tweet in self.tweets:
            tweet.print()
        print()

    def write_in_file(self, file):
        for tweet in self.tweets:
            tweet.write_in_file(file)
        file.write('\n')


def read_cascades(index_file_addr='index.csv',
                  data_file_addr='data.csv', debug=None):
    cascades_info_addresses = []
    cascades = []

    with open(index_file_addr) as cascades_info_file:
        info_reader = csv.reader(cascades_info_file)

        counter = 0
        for line in info_reader:
            if counter == 0:  # skipping first line of file
                counter += 1
                continue

            cascades_info_addresses.append(
                [int(x) for x in line[2:]]
            )

            counter += 1

    with open(data_file_addr) as cascades_file:
        reader = csv.reader(cascades_file)

        counter = 0
        cascade_index = 0
        tweets = []
        for line in reader:
            if counter == 0:  # skipping first line of file
                counter += 1
                continue

            if counter > cascades_info_addresses[cascade_index][1]:
                cascade_index += 1
                cascades.append(Cascade(tweets))
                tweets = []

                # print('->', counter)

            try:
                time, followers_num = [float(x) for x in line]
                tweets.append(Tweet(followers_num, time))
            except ValueError as e:
                print(e)
                print(line)
                print(counter)
            # print(time, followers_num)

            counter += 1
            if debug is not None and counter > debug:
                break
            # print(counter, len(cascades))
            # if len(cascades) > 1000:
            #     break

    # print('cas indx', cascade_index)
    print('num of all cascades: ', len(cascades))
    total_len = 0
    for cas in cascades:
        total_len += len(cas.get_tweet_times())
    print(total_len)

    return cascades


def read_weibo_cascades(cascades_file_addr='../weibo_preprocess/weibo_cascades.txt'):
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


def create_time_windows(cascades, time_window_len=10):
    windowed_cascades = []

    c = 0
    for cascade in cascades:
        if cascade.get_tweet_times()[-1] == 0:
            bins = 1
        else:
            bins = ceil(cascade.get_tweet_times()[-1] / time_window_len)
        # print('bins:', bins)
        hist_y, hist_x = np.histogram(cascade.get_tweet_times(), bins=bins)
        hist_y = hist_y.tolist()
        hist_x = hist_x.tolist()

        #print(len(hist_y))
        #c += 1
        #print(c)

        windowed_cascades.append(hist_y)
        # print(hist_y[0:100])
        # print(hist_x[0:100])

    return windowed_cascades


def create_slope_series(windowed_cascades):
    slope_series = []
    for cascade in windowed_cascades:
        cascade_slopes = [cascade[0]]
        for i in range(len(cascade) - 1):
            cascade_slopes.append(
                cascade[i + 1] - cascade[i]
            )
        slope_series.append(cascade_slopes)

    return np.array(slope_series)


def get_observed_window_of_cascades(cascades, observation_time):
    observed_cascades = []

    for cascade in cascades:
        observed_tweets_num = 0
        for tweet in cascade.tweets:
            if tweet.time > observation_time:
                break
            observed_tweets_num += 1

        observed_cascades.append(Cascade(cascade.tweets[0:observed_tweets_num]))

    return observed_cascades


def get_peak_windows(windowed_cascades):
    peak_times = []
    for cascade in windowed_cascades:
        peak_times.append(np.argmax(cascade))
        # print("-->", np.argmax(cascade))
    return np.array(peak_times)


def normalize_cascades(cascades):
    scaler = MinMaxScaler()
    scaler = scaler.fit(cascades)
    return scaler.transform(cascades)


def normalize_peak_times(peak_times):  # changing duration of bins from 10s to 1h (new test: 10h)
    # return np.array([np.floor(peak_time / 360) for peak_time in peak_times])
    return np.array([np.floor(peak_time / 3600) for peak_time in peak_times])


def cascades_to_proper_burst_pred_input(max_observation_time=3600,
                                        burst_min_len=400, non_burst_max_len=100,
                                        max_burst_cascades_num=1000000,
                                        index_file_addr='../seismic_dataset/index.csv',
                                        data_file_addr='../seismic_dataset/data.csv'
                                        ):
    cascades = read_cascades(index_file_addr=index_file_addr,
                             data_file_addr=data_file_addr)

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    burst_cascades = burst_cascades[:max_burst_cascades_num]

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))

    # selected_non_burst_cascades = non_burst_cascades[0:2*len(burst_cascades)]
    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    # selected_cascades, labels = shuffle(selected_cascades, labels)
    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    observed_cascades = get_observed_window_of_cascades(selected_cascades, max_observation_time)

    # print('burst num:', len(burst_cascades), 'non_burst num:', len(non_burst_cascades))
    # print(len(selected_cascades))

    windowed_cascades = create_time_windows(observed_cascades)  # -> time series (not list of Cascades)

    max_len = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)

    for i in range(len(windowed_cascades)):
        windowed_cascades[i] += np.zeros(max_len - len(windowed_cascades[i])).tolist()
        windowed_cascades[i] = np.array(windowed_cascades[i])

    # print(np.array(windowed_cascades).shape)
    # print(np.array(labels).shape)

    # return normalize_cascades(np.array(windowed_cascades)), np.array(labels)
    return np.array(windowed_cascades), np.array(labels)


def weibo_cascades_to_proper_burst_pred_input(max_observation_time=3600,
                                              burst_min_len=400, non_burst_max_len=100,
                                              max_burst_cascades_num=1000000,
                                              cascades_file_addr='../weibo_preprocess/weibo_cascades.txt'
                                              ):
    cascades = read_weibo_cascades(cascades_file_addr=cascades_file_addr)

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    burst_cascades = burst_cascades[:max_burst_cascades_num]

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))

    # selected_non_burst_cascades = non_burst_cascades[0:2*len(burst_cascades)]
    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    # selected_cascades, labels = shuffle(selected_cascades, labels)
    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    observed_cascades = get_observed_window_of_cascades(selected_cascades, max_observation_time)

    # print('burst num:', len(burst_cascades), 'non_burst num:', len(non_burst_cascades))
    # print(len(selected_cascades))

    windowed_cascades = create_time_windows(observed_cascades)  # -> time series (not list of Cascades)

    max_len = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)

    for i in range(len(windowed_cascades)):
        windowed_cascades[i] += np.zeros(max_len - len(windowed_cascades[i])).tolist()
        windowed_cascades[i] = np.array(windowed_cascades[i])

    # print(np.array(windowed_cascades).shape)
    # print(np.array(labels).shape)

    # return normalize_cascades(np.array(windowed_cascades)), np.array(labels)
    return np.array(windowed_cascades), np.array(labels)


def cascades_to_proper_peak_pred_input(max_observation_time=3600,
                                       burst_min_len=1000, non_burst_max_len=900, cascades_min_len=900):
    cascades = read_cascades()

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    # selected_cascades, labels = shuffle(selected_cascades, labels)
    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    observed_cascades = get_observed_window_of_cascades(selected_cascades, max_observation_time)

    observed_windowed_cascades = create_time_windows(observed_cascades)  # -> time series (not list of Cascades)
    windowed_cascades = create_time_windows(selected_cascades)  # -> time series (not list of Cascades)

    max_len = 0
    for windowed_cascade in observed_windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)

    # max_real_len = 0  # max len when cascades are not clipped
    # for windowed_cascade in windowed_cascades:
    #     if len(windowed_cascade) > max_len:
    #         max_real_len = len(windowed_cascade)
    #
    # print('max_actual_len is ', max_real_len)

    # answer: max_actual_len is 46249

    for i in range(len(observed_windowed_cascades)):
        observed_windowed_cascades[i] += np.zeros(max_len - len(observed_windowed_cascades[i])).tolist()
        observed_windowed_cascades[i] = np.array(observed_windowed_cascades[i])

    return np.array(observed_windowed_cascades), np.array(labels), \
           normalize_peak_times(get_peak_windows(windowed_cascades))


def cascades_to_cheng_plus_spike_pred_input(max_observation_time=3600,
                                            burst_min_len=1000, non_burst_max_len=900, cascades_min_len=900,
                                            index_file_addr='../seismic_dataset/index.csv',
                                            data_file_addr='../seismic_dataset/data.csv'
                                            ):
    cascades = read_cascades(index_file_addr=index_file_addr, data_file_addr=data_file_addr)

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    # selected_cascades, labels = shuffle(selected_cascades, labels)
    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    observed_cascades = get_observed_window_of_cascades(selected_cascades, max_observation_time)

    observed_windowed_cascades = create_time_windows(observed_cascades)  # -> time series (not list of Cascades)
    # observed_windowed_cascades = create_time_windows(observed_cascades,
    #                                                  time_window_len=100)  # todo: don't forget that this is not the original prop of paper

    # windowed_cascades = create_time_windows(selected_cascades)  # -> time series (not list of Cascades)
    windowed_cascades = create_time_windows(selected_cascades,
                                            time_window_len=100)  # todo: don't forget that this is not the original prop of paper
    peak_times = get_peak_windows(windowed_cascades)

    max_len = 0
    for windowed_cascade in observed_windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)

    max_len_2 = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len_2:
            max_len_2 = len(windowed_cascade)

    print('max_len_2 is ', max_len_2)

    # max_real_len = 0  # max len when cascades are not clipped
    # for windowed_cascade in windowed_cascades:
    #     if len(windowed_cascade) > max_len:
    #         max_real_len = len(windowed_cascade)
    #
    # print('max_actual_len is ', max_real_len)

    # answer: max_actual_len is 46249

    for i in range(len(observed_windowed_cascades)):
        observed_windowed_cascades[i] += np.zeros(max_len - len(observed_windowed_cascades[i])).tolist()
        observed_windowed_cascades[i] = np.array(observed_windowed_cascades[i])

    for i in range(len(windowed_cascades)):
        windowed_cascades[i] += np.zeros(max_len_2 - len(windowed_cascades[i])).tolist()
        windowed_cascades[i] = np.array(windowed_cascades[i])

    # cascades_peak_labels = get_peak_labels(windowed_cascades, labels, peak_times)
    cascades_peak_labels = []

    counter = 0
    for cascade in windowed_cascades:
        if labels[counter]:
            cascades_peak_labels.append(
                np.append(np.zeros(peak_times[counter]),
                          np.ones(len(cascade) - peak_times[counter]))
            )

        else:
            cascades_peak_labels.append(np.zeros(len(cascade)))
        counter += 1

    return np.array(observed_windowed_cascades), np.array(cascades_peak_labels), np.array(labels)


def weibo_cascades_to_cheng_plus_spike_pred_input(max_observation_time=3600,
                                                  burst_min_len=1000, non_burst_max_len=900, cascades_min_len=900,
                                                  time_bin_len=800,
                                                  cascades_file_addr='../weibo_preprocess/weibo_cascades.txt'
                                                  ):
    cascades = read_weibo_cascades(cascades_file_addr=cascades_file_addr)

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    print('bursts num', len(burst_cascades))
    print('non bursts num', len(non_burst_cascades))

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),
                             np.zeros(len(selected_non_burst_cascades))))

    # selected_cascades, labels = shuffle(selected_cascades, labels)
    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    observed_cascades = get_observed_window_of_cascades(selected_cascades, max_observation_time)

    observed_windowed_cascades = create_time_windows(observed_cascades)  # -> time series (not list of Cascades)
    windowed_cascades = create_time_windows(selected_cascades,
                                            time_window_len=time_bin_len)
    peak_times = get_peak_windows(windowed_cascades)

    max_len = 0
    for windowed_cascade in observed_windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    print('max_len is ', max_len)

    max_len_2 = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len_2:
            max_len_2 = len(windowed_cascade)

    print('max_len_2 is ', max_len_2)

    # max_real_len = 0  # max len when cascades are not clipped
    # for windowed_cascade in windowed_cascades:
    #     if len(windowed_cascade) > max_len:
    #         max_real_len = len(windowed_cascade)
    #
    # print('max_actual_len is ', max_real_len)

    # answer: max_actual_len is 46249

    for i in range(len(observed_windowed_cascades)):
        observed_windowed_cascades[i] += np.zeros(max_len - len(observed_windowed_cascades[i])).tolist()
        observed_windowed_cascades[i] = np.array(observed_windowed_cascades[i])

    for i in range(len(windowed_cascades)):
        windowed_cascades[i] += np.zeros(max_len_2 - len(windowed_cascades[i])).tolist()
        windowed_cascades[i] = np.array(windowed_cascades[i])

    # cascades_peak_labels = get_peak_labels(windowed_cascades, labels, peak_times)
    cascades_peak_labels = []

    counter = 0
    for cascade in windowed_cascades:
        if labels[counter]:
            cascades_peak_labels.append(
                np.append(np.zeros(peak_times[counter]),
                          np.ones(len(cascade) - peak_times[counter]))
            )

        else:
            cascades_peak_labels.append(np.zeros(len(cascade)))
        counter += 1

    return np.array(observed_windowed_cascades), np.array(cascades_peak_labels), np.array(labels)


def cascades_to_proper_burst_pred_input_test_split(max_observation_time=3600,
                                                   burst_min_len=400, non_burst_max_len=100, test_size=0.2):
    cascades, labels = cascades_to_proper_burst_pred_input(
        max_observation_time, burst_min_len, non_burst_max_len
    )

    return train_test_split(cascades, labels, test_size=test_size)


def cascades_to_proper_peak_pred_input_test_split(max_observation_time=3600, burst_min_len=1000,
                                                  non_burst_max_len=900, cascades_min_len=900, test_size=0.2):
    cascades, labels, peak_times = cascades_to_proper_peak_pred_input(
        max_observation_time, burst_min_len, non_burst_max_len, cascades_min_len
    )

    return train_test_split(cascades, labels, peak_times, test_size=test_size)


def cascades_to_burst_pred_multi_representation_test_split(max_observation_time=3600,
                                                           burst_min_len=400, non_burst_max_len=100, test_size=0.2,
                                                           max_burst_cascades_num=1000000,
                                                           index_file_addr='../seismic_dataset/index.csv',
                                                           data_file_addr='../seismic_dataset/data.csv'
                                                           ):
    windowed_cascades, labels = cascades_to_proper_burst_pred_input(
        max_observation_time, burst_min_len, non_burst_max_len,
        max_burst_cascades_num=max_burst_cascades_num,
        index_file_addr=index_file_addr, data_file_addr=data_file_addr
    )

    slope_series = create_slope_series(windowed_cascades)

    cascades = []
    for i in range(len(windowed_cascades)):
        cascade_representation = np.array([
            windowed_cascades[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    return train_test_split(cascades, labels, test_size=test_size)


def weibo_cascades_to_burst_pred_multi_representation_test_split(max_observation_time=3600,
                                                                 burst_min_len=400, non_burst_max_len=100,
                                                                 test_size=0.2,
                                                                 max_burst_cascades_num=1000000,
                                                                 cascades_file_addr=
                                                                 '../weibo_preprocess/weibo_cascades.txt'
                                                                 ):
    windowed_cascades, labels = weibo_cascades_to_proper_burst_pred_input(
        max_observation_time, burst_min_len, non_burst_max_len,
        max_burst_cascades_num=max_burst_cascades_num,
        cascades_file_addr=cascades_file_addr
    )

    slope_series = create_slope_series(windowed_cascades)

    cascades = []
    for i in range(len(windowed_cascades)):
        cascade_representation = np.array([
            windowed_cascades[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    return train_test_split(cascades, labels, test_size=test_size)


def cascades_to_cheng_multi_representation_uniformed_linear_diminishing_test_split(  # todo: make sure works correctly
        max_observation_time=3600,
        burst_min_len=200, non_burst_max_len=100,
        cascades_min_len=50,
        peak_bins_num=20,
        max_cascade_in_each_bin=1500,
        max_non_burst_twice_the_max_burst_in_bin=True,
        test_size=0.2,
        index_file_addr='../seismic_dataset/index.csv',
        data_file_addr='../seismic_dataset/data.csv'):
    # many parts copied from new_uniformed_twitter_with_linear_diminishing functions in
    # cascade_preprocessor_seismic_survival file

    windowed_cascades, cascade_peak_labels, labels = cascades_to_cheng_plus_spike_pred_input(
        max_observation_time,
        burst_min_len,
        non_burst_max_len,
        cascades_min_len,
        index_file_addr=index_file_addr,
        data_file_addr=data_file_addr)

    print('init shapes:', windowed_cascades.shape, cascade_peak_labels.shape, labels.shape)

    windowed_cascades_burst = []
    windowed_cascades_non_burst = []
    cascade_peak_labels_burst = []
    cascade_peak_labels_non_burst = []
    labels_burst = []
    labels_non_burst = []

    for i in range(len(labels)):
        if labels[i] == 1:
            windowed_cascades_burst.append(windowed_cascades[i])
            cascade_peak_labels_burst.append(cascade_peak_labels[i])
            labels_burst.append(labels[i])
        else:
            windowed_cascades_non_burst.append(windowed_cascades[i])
            cascade_peak_labels_non_burst.append(cascade_peak_labels[i])
            labels_non_burst.append(labels[i])

    windowed_cascades_burst_new = []
    cascade_peak_labels_burst_new = []
    labels_burst_new = []

    print('shapes after burst and non burst separation:', np.array(windowed_cascades_burst).shape,
          np.array(cascade_peak_labels_burst).shape, np.array(labels_burst).shape,
          np.array(windowed_cascades_non_burst).shape, np.array(cascade_peak_labels_non_burst).shape,
          np.array(labels_non_burst).shape)

    # from deep_burst_detection.cascade_preprocessor_seismic_survival import \
    #     get_one_hot_peak_times_from_labels_with_linear_time_diminishing

    def get_one_hot_peak_times_from_labels_with_linear_time_diminishing_copy(cascade_peak_labels_,
                                                                             prediction_time_windows=20,
                                                                             first_bin_len=6):
        len_peak_labels = len(cascade_peak_labels_[0])
        num_burst_time_windows = prediction_time_windows - 1

        # f = first_bin_len, n = num_burst_time_windows,
        # m = len_peak_labels => f + (f+x) + (f+2x) + ... + (f+(n-1)x) = m
        # => nf + [n(n-1)/2]x = m => x = (m-nf) / [n(n-1)/2]

        windows_time_difference = np.floor((len_peak_labels - num_burst_time_windows * first_bin_len)
                                           / (num_burst_time_windows * (num_burst_time_windows - 1) / 2))

        time_windows_lengths = []
        for i in range(num_burst_time_windows):
            time_windows_lengths.append(first_bin_len + i * windows_time_difference)

        divide_points = []
        for i in range(num_burst_time_windows):
            divide_points.append(sum(time_windows_lengths[:i]))

        divide_points.append(len_peak_labels)

        # print(len_peak_labels, num_burst_time_windows, windows_time_difference)
        # print(time_windows_lengths)
        # print(divide_points)

        # output:
        # 6048 19 34.0
        # [6.0, 40.0, 74.0, 108.0, 142.0, 176.0, 210.0, 244.0, 278.0, 312.0, 346.0, 380.0, 414.0, 448.0, 482.0, 516.0,
        # 550.0, 584.0, 618.0]
        # [0, 6.0, 46.0, 120.0, 228.0, 370.0, 546.0, 756.0, 1000.0, 1278.0, 1590.0, 1936.0, 2316.0, 2730.0, 3178.0,
        # 3660.0,
        # 4176.0, 4726.0, 5310.0, 6048]

        one_hots_times = []
        for peak_labels in cascade_peak_labels_:
            one_hot_peak_time = np.zeros(prediction_time_windows)

            sum_peak_labels = sum(peak_labels)
            peak_time = len(peak_labels) - sum_peak_labels
            diminished_peak_time = -1
            for i in range(prediction_time_windows - 1):
                if divide_points[i] <= peak_time <= divide_points[i + 1]:
                    diminished_peak_time = i
                    break

            if sum_peak_labels == 0:  # non_burst cascade without peak time
                one_hot_peak_time[-1] = 1
            else:
                one_hot_peak_time[diminished_peak_time] = 1

            one_hots_times.append(one_hot_peak_time)

        return np.array(one_hots_times)

    simplified_cascade_peak_labels_burst = np.argmax(
        get_one_hot_peak_times_from_labels_with_linear_time_diminishing_copy(cascade_peak_labels_=
                                                                             cascade_peak_labels_burst,
                                                                             prediction_time_windows=peak_bins_num), 1)

    peak_dist = np.zeros(peak_bins_num)
    i = 0
    for peak_bin in simplified_cascade_peak_labels_burst:
        # print('bin -> ', peak_bin)

        if peak_dist[peak_bin] < max_cascade_in_each_bin:
            windowed_cascades_burst_new.append(windowed_cascades_burst[i])
            cascade_peak_labels_burst_new.append(cascade_peak_labels_burst[i])
            labels_burst_new.append(labels_burst[i])
            peak_dist[peak_bin] += 1

        i += 1

        # print('peak_dist after uniformization:', list(peak_dist))

    # remaining_burst_cascades_num = len(windowed_cascades_burst_new)
    # windowed_cascades_non_burst = windowed_cascades_non_burst[:remaining_burst_cascades_num]
    # cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:remaining_burst_cascades_num]
    # labels_non_burst = labels_non_burst[:remaining_burst_cascades_num]

    print('shapes after uniformization:', np.array(windowed_cascades_burst_new).shape,
          np.array(cascade_peak_labels_burst_new).shape, np.array(labels_burst_new).shape)

    if max_non_burst_twice_the_max_burst_in_bin:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:2 * max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:2 * max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:2 * max_cascade_in_each_bin]
    else:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:max_cascade_in_each_bin]

    windowed_cascades_new = np.array(windowed_cascades_burst_new + windowed_cascades_non_burst)
    cascade_peak_labels_new = np.array(cascade_peak_labels_burst_new + cascade_peak_labels_non_burst)
    labels_new = np.array(labels_burst_new + labels_non_burst)

    slope_series = create_slope_series(windowed_cascades_new)

    cascades = []
    for i in range(len(windowed_cascades_new)):
        cascade_representation = np.array([
            windowed_cascades_new[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    print('final shapes:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    cascades, cascade_peak_labels_new, labels_new = shuffle(cascades,
                                                            cascade_peak_labels_new, labels_new,
                                                            random_state=0)
    # todo: remove random_state=0

    # return windowed_cascades_new, cascade_peak_labels_new, labels_new

    print('cascades_to_cheng_multi_representation_uniformed_linear_diminishing_test_split')
    print(len(cascades), 'cascades')

    print('final shapes after shuffling:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    return train_test_split(cascades, cascade_peak_labels_new, labels_new, test_size=test_size,
                            shuffle=False)  # todo: remove this after finishing working with seed 0


def cascades_to_cheng_multi_representation_uniformed_test_split(
        max_observation_time=3600,
        burst_min_len=200, non_burst_max_len=100,
        cascades_min_len=50,
        peak_bins_num=20,
        max_cascade_in_each_bin=1500,
        max_non_burst_twice_the_max_burst_in_bin=True,
        test_size=0.2,
        index_file_addr='../seismic_dataset/index.csv',
        data_file_addr='../seismic_dataset/data.csv'):
    windowed_cascades, cascade_peak_labels, labels = cascades_to_cheng_plus_spike_pred_input(
        max_observation_time,
        burst_min_len,
        non_burst_max_len,
        cascades_min_len,
        index_file_addr=index_file_addr,
        data_file_addr=data_file_addr)

    print('init shapes:', windowed_cascades.shape, cascade_peak_labels.shape, labels.shape)

    windowed_cascades_burst = []
    windowed_cascades_non_burst = []
    cascade_peak_labels_burst = []
    cascade_peak_labels_non_burst = []
    labels_burst = []
    labels_non_burst = []

    for i in range(len(labels)):
        if labels[i] == 1:
            windowed_cascades_burst.append(windowed_cascades[i])
            cascade_peak_labels_burst.append(cascade_peak_labels[i])
            labels_burst.append(labels[i])
        else:
            windowed_cascades_non_burst.append(windowed_cascades[i])
            cascade_peak_labels_non_burst.append(cascade_peak_labels[i])
            labels_non_burst.append(labels[i])

    windowed_cascades_burst_new = []
    cascade_peak_labels_burst_new = []
    labels_burst_new = []

    print('shapes after burst and non burst separation:', np.array(windowed_cascades_burst).shape,
          np.array(cascade_peak_labels_burst).shape, np.array(labels_burst).shape,
          np.array(windowed_cascades_non_burst).shape, np.array(cascade_peak_labels_non_burst).shape,
          np.array(labels_non_burst).shape)

    def get_one_hot_peak_times_from_labels_copy(cascade_peak_labels_, diminish_scale=10):
        one_hots_times = []
        for peak_labels in cascade_peak_labels_:
            # one_hot_peak_time = np.zeros(int(len(peak_labels) / diminish_scale) + 1)
            one_hot_peak_time = np.zeros(int(np.ceil(len(peak_labels) / diminish_scale)))
            # todo: make sure this change doesn't damage any part of the code

            sum_peak_labels = sum(peak_labels)
            peak_time = len(peak_labels) - sum_peak_labels
            if sum_peak_labels == 0:  # non_burst cascade without peak time
                one_hot_peak_time[-1] = 1
            else:
                one_hot_peak_time[int(peak_time / diminish_scale)] = 1

            one_hots_times.append(one_hot_peak_time)

        return np.array(one_hots_times)

    simplified_cascade_peak_labels_burst = np.argmax(
        get_one_hot_peak_times_from_labels_copy(cascade_peak_labels_=cascade_peak_labels_burst, diminish_scale=1), 1)

    peak_dist = np.zeros(peak_bins_num)
    denominator = np.ceil(len(cascade_peak_labels_burst[0]) / peak_bins_num)
    # print('denom', len(cascade_peak_labels_burst[0]),  peak_bins_num, denominator)
    for i in range(len(labels_burst)):
        peak_bin = int(simplified_cascade_peak_labels_burst[i] / denominator)
        # print('bin -> ', simplified_cascade_peak_labels_burst[i], peak_bin)

        if peak_dist[peak_bin] < max_cascade_in_each_bin:
            windowed_cascades_burst_new.append(windowed_cascades_burst[i])
            cascade_peak_labels_burst_new.append(cascade_peak_labels_burst[i])
            labels_burst_new.append(labels_burst[i])
            peak_dist[peak_bin] += 1

        # print('peak_dist after uniformization:', list(peak_dist))

    # remaining_burst_cascades_num = len(windowed_cascades_burst_new)
    # windowed_cascades_non_burst = windowed_cascades_non_burst[:remaining_burst_cascades_num]
    # cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:remaining_burst_cascades_num]
    # labels_non_burst = labels_non_burst[:remaining_burst_cascades_num]

    print('shapes after uniformization:', np.array(windowed_cascades_burst_new).shape,
          np.array(cascade_peak_labels_burst_new).shape, np.array(labels_burst_new).shape)

    if max_non_burst_twice_the_max_burst_in_bin:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:2 * max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:2 * max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:2 * max_cascade_in_each_bin]
    else:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:max_cascade_in_each_bin]

    windowed_cascades_new = np.array(windowed_cascades_burst_new + windowed_cascades_non_burst)
    cascade_peak_labels_new = np.array(cascade_peak_labels_burst_new + cascade_peak_labels_non_burst)
    labels_new = np.array(labels_burst_new + labels_non_burst)

    slope_series = create_slope_series(windowed_cascades_new)

    cascades = []
    for i in range(len(windowed_cascades_new)):
        cascade_representation = np.array([
            windowed_cascades_new[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    print('final shapes:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    cascades, cascade_peak_labels_new, labels_new = shuffle(cascades,
                                                            cascade_peak_labels_new, labels_new,
                                                            random_state=0)
    # todo: remove random_state=0

    # return windowed_cascades_new, cascade_peak_labels_new, labels_new

    print('cascades_to_cheng_multi_representation_uniformed_test_split')
    print(len(cascades), 'cascades')

    print('final shapes after shuffling:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    return train_test_split(cascades, cascade_peak_labels_new, labels_new, test_size=test_size,
                            shuffle=False)  # todo: remove this after finishing working with seed 0


def weibo_cascades_to_cheng_multi_representation_uniformed_test_split(
        max_observation_time=3600,
        burst_min_len=200, non_burst_max_len=100,
        cascades_min_len=50,
        peak_bins_num=20,
        max_cascade_in_each_bin=1500,
        max_non_burst_twice_the_max_burst_in_bin=True,
        test_size=0.2,
        time_bin_len=800,
        cascades_file_addr='../weibo_preprocess/weibo_cascades.txt'):
    windowed_cascades, cascade_peak_labels, labels = weibo_cascades_to_cheng_plus_spike_pred_input(
        max_observation_time,
        burst_min_len,
        non_burst_max_len,
        cascades_min_len,
        time_bin_len=time_bin_len,
        cascades_file_addr=cascades_file_addr)

    print('init shapes:', windowed_cascades.shape, cascade_peak_labels.shape, labels.shape)

    windowed_cascades_burst = []
    windowed_cascades_non_burst = []
    cascade_peak_labels_burst = []
    cascade_peak_labels_non_burst = []
    labels_burst = []
    labels_non_burst = []

    for i in range(len(labels)):
        if labels[i] == 1:
            windowed_cascades_burst.append(windowed_cascades[i])
            cascade_peak_labels_burst.append(cascade_peak_labels[i])
            labels_burst.append(labels[i])
        else:
            windowed_cascades_non_burst.append(windowed_cascades[i])
            cascade_peak_labels_non_burst.append(cascade_peak_labels[i])
            labels_non_burst.append(labels[i])

    windowed_cascades_burst_new = []
    cascade_peak_labels_burst_new = []
    labels_burst_new = []

    print('shapes after burst and non burst separation:', np.array(windowed_cascades_burst).shape,
          np.array(cascade_peak_labels_burst).shape, np.array(labels_burst).shape,
          np.array(windowed_cascades_non_burst).shape, np.array(cascade_peak_labels_non_burst).shape,
          np.array(labels_non_burst).shape)

    def get_one_hot_peak_times_from_labels_copy(cascade_peak_labels_, diminish_scale=10):
        one_hots_times = []
        for peak_labels in cascade_peak_labels_:
            # one_hot_peak_time = np.zeros(int(len(peak_labels) / diminish_scale) + 1)
            one_hot_peak_time = np.zeros(int(np.ceil(len(peak_labels) / diminish_scale)))
            # todo: make sure this change doesn't damage any part of the code

            sum_peak_labels = sum(peak_labels)
            peak_time = len(peak_labels) - sum_peak_labels
            if sum_peak_labels == 0:  # non_burst cascade without peak time
                one_hot_peak_time[-1] = 1
            else:
                one_hot_peak_time[int(peak_time / diminish_scale)] = 1

            one_hots_times.append(one_hot_peak_time)

        return np.array(one_hots_times)

    simplified_cascade_peak_labels_burst = np.argmax(
        get_one_hot_peak_times_from_labels_copy(cascade_peak_labels_=cascade_peak_labels_burst, diminish_scale=1), 1)

    peak_dist = np.zeros(peak_bins_num)
    denominator = np.ceil(len(cascade_peak_labels_burst[0]) / peak_bins_num)
    # print('denom', len(cascade_peak_labels_burst[0]),  peak_bins_num, denominator)
    for i in range(len(labels_burst)):
        peak_bin = int(simplified_cascade_peak_labels_burst[i] / denominator)
        # print('bin -> ', simplified_cascade_peak_labels_burst[i], peak_bin)

        if peak_dist[peak_bin] < max_cascade_in_each_bin:
            windowed_cascades_burst_new.append(windowed_cascades_burst[i])
            cascade_peak_labels_burst_new.append(cascade_peak_labels_burst[i])
            labels_burst_new.append(labels_burst[i])
            peak_dist[peak_bin] += 1

        # print('peak_dist after uniformization:', list(peak_dist))

    # remaining_burst_cascades_num = len(windowed_cascades_burst_new)
    # windowed_cascades_non_burst = windowed_cascades_non_burst[:remaining_burst_cascades_num]
    # cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:remaining_burst_cascades_num]
    # labels_non_burst = labels_non_burst[:remaining_burst_cascades_num]

    print('shapes after uniformization:', np.array(windowed_cascades_burst_new).shape,
          np.array(cascade_peak_labels_burst_new).shape, np.array(labels_burst_new).shape)

    if max_non_burst_twice_the_max_burst_in_bin:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:2 * max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:2 * max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:2 * max_cascade_in_each_bin]
    else:
        windowed_cascades_non_burst = windowed_cascades_non_burst[:max_cascade_in_each_bin]
        cascade_peak_labels_non_burst = cascade_peak_labels_non_burst[:max_cascade_in_each_bin]
        labels_non_burst = labels_non_burst[:max_cascade_in_each_bin]

    windowed_cascades_new = np.array(windowed_cascades_burst_new + windowed_cascades_non_burst)
    cascade_peak_labels_new = np.array(cascade_peak_labels_burst_new + cascade_peak_labels_non_burst)
    labels_new = np.array(labels_burst_new + labels_non_burst)

    slope_series = create_slope_series(windowed_cascades_new)

    cascades = []
    for i in range(len(windowed_cascades_new)):
        cascade_representation = np.array([
            windowed_cascades_new[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    print('final shapes:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    cascades, cascade_peak_labels_new, labels_new = shuffle(cascades, cascade_peak_labels_new, labels_new,
                                                            random_state=0)
    # todo: remove random_state=0

    # return windowed_cascades_new, cascade_peak_labels_new, labels_new

    print('cascades_to_cheng_multi_representation_uniformed_test_split')
    print(len(cascades), 'cascades')

    print('final shapes after shuffling:', cascades.shape, cascade_peak_labels_new.shape, labels_new.shape)

    return train_test_split(cascades, cascade_peak_labels_new, labels_new, test_size=test_size,
                            shuffle=False)  # todo: remove this after finishing working with seed 0


def cascades_to_cheng_model_plus_pred_input_test_split(max_observation_time=3600,
                                                       burst_min_len=1000,
                                                       non_burst_max_len=900,
                                                       cascades_min_len=50,
                                                       test_size=0.2
                                                       ):
    # this function returns cascades with dimension (batch_size, max_series_len, 2) -> cascades have 2 representation

    windowed_cascades, cascade_peak_labels, labels = cascades_to_cheng_plus_spike_pred_input(max_observation_time,
                                                                                             burst_min_len,
                                                                                             non_burst_max_len,
                                                                                             cascades_min_len)

    slope_series = create_slope_series(windowed_cascades)

    cascades = []
    for i in range(len(windowed_cascades)):
        cascade_representation = np.array([
            windowed_cascades[i], slope_series[i]
        ])

        cascades.append(cascade_representation)

    cascades = np.array(cascades)
    cascades = np.reshape(cascades, newshape=[
        cascades.shape[0], cascades.shape[2], -1
    ])

    return train_test_split(cascades, cascade_peak_labels, labels, test_size=test_size,
                            shuffle=False)  # todo: remove this after finishing working with seed 0


def spike_hist_drawer(fig_name='peaK_distribution_non_zero.png'):
    _, _, _, _, train_peak_times, test_peak_times = cascades_to_proper_peak_pred_input_test_split(
        cascades_min_len=800
    )

    # hist_y, hist_x = np.histogram(train_peak_times.tolist(), bins=24)
    # plt.plot(hist_x[0:(len(hist_x)-1)], hist_y, 'ro')
    # plt.savefig(fig_name)

    l = []
    for peak_time in train_peak_times:
        if peak_time != 0:
            l.append(peak_time)

    hist_y, hist_x = np.histogram(l, bins=24)
    plt.plot(hist_x[0:(len(hist_x) - 1)], hist_y, 'ro')
    plt.savefig(fig_name)

# spike_hist_drawer()
# cascades_to_cheng_multi_representation_uniformed_linear_diminishing_test_split()


def get_burst_time(windowed_cascades, burst_min_len):
    burst_times = []
    for cascade in windowed_cascades:
        if sum(cascade) > burst_min_len:
            cumulative_sum = np.cumsum(cascade)
            burst_times.append(np.argwhere(cumulative_sum > burst_min_len)[0][0])
        else:
            burst_times.append(-1)
    return np.array(burst_times)

