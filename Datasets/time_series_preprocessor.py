import numpy as np
from sklearn.utils import shuffle

from Datasets.BasicObjects import Tweet, Cascade, create_time_windows, get_peak_windows, get_peak_labels

"""
Total Cascades: 527847
"""


def read_cascades(cascades_file_addr='../Datasets/NBA-cascade.txt', debug_maximum=None, verbose=False):
    with open(cascades_file_addr, 'r') as cascades_file:

        cascades = []

        min_cascade_size = 10
        debug_counter = 0
        counter = 0
        for line in cascades_file:
            if counter <= 2056:  # skipping redundant lines
                counter += 1
                continue

            line = line[0:len(line) - 1]  # deleting enter from end of each line
            split_line = line.split(';')

            if len(split_line) < min_cascade_size:
                continue

            tweets = []
            for tweet in split_line:
                split_tweet = tweet.split(',')
                twitter = int(split_tweet[0])
                time = int(float(split_tweet[1]) * 10000)  # todo: is this scaling good?
                tweets.append(Tweet(twitter, time))
            cascades.append(Cascade(tweets))
            debug_counter += 1

            if debug_maximum is not None and debug_counter >= debug_maximum:
                return cascades

            if debug_counter % 1000 == 0:
                print(debug_counter)
    if verbose:
        print('cascades num', len(cascades))

    return cascades


def cascades_to_proper_survival_model_input(burst_min_len=100, non_burst_max_len=50, cascades_min_len=10, verbose=True,
                                            debug_maximum=None):
    cascades = read_cascades(debug_maximum=debug_maximum, verbose=verbose)  # TODO: REDO

    burst_cascades = []
    non_burst_cascades = []

    for windowed_cascade in cascades:  # TODO: I dont get it
        if len(windowed_cascade.get_tweet_times()) > burst_min_len:
            burst_cascades.append(windowed_cascade)

        if cascades_min_len < len(windowed_cascade.get_tweet_times()) < non_burst_max_len:
            non_burst_cascades.append(windowed_cascade)

    selected_non_burst_cascades = non_burst_cascades[0:len(burst_cascades)]
    selected_cascades = np.array(burst_cascades + selected_non_burst_cascades)
    labels = np.concatenate((np.ones(len(burst_cascades)),  # TODO: Where did the labels come from?
                             np.zeros(len(selected_non_burst_cascades))))

    if verbose:
        print('bursts num', len(burst_cascades))
        print('non bursts num', len(non_burst_cascades))
        print('selected cascades num', len(selected_cascades))

    selected_cascades, labels = shuffle(selected_cascades, labels,
                                        random_state=0)  # todo: remove this after finishing working with seed 0
    # windowed_cascades = create_time_windows(selected_cascades)
    # windowed_cascades = create_time_windows(selected_cascades, time_window_len=100)  # todo: unTOF
    windowed_cascades = create_time_windows(selected_cascades, time_window_len=2160000)  # todo: unTOF
    peak_times = get_peak_windows(windowed_cascades)
    # peak_times = get_burst_threshold_times(windowed_cascades)  # todo: undo this test

    max_len = 0
    for windowed_cascade in windowed_cascades:
        if len(windowed_cascade) > max_len:
            max_len = len(windowed_cascade)

    if verbose:
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

    cascade_peak_labels = get_peak_labels(windowed_cascades, labels, peak_times)

    return np.array(windowed_cascades), np.array(cascade_peak_labels), np.array(labels)

# def cascades_to_proper_survival_model_input_test_split(burst_min_len=100,
#                                                        non_burst_max_len=50,
#                                                        cascades_min_len=10,
#                                                        test_size=0.2
#                                                        ):
# cascades, peak_labels, labels = cascades_to_proper_survival_model_input()
# print("cascades shape:", cascades.shape)
# plt.plot(cascades[2])
# plt.show()
# print(np.max(cascades))

# a_cascade = cascades[0]
# a_cascade.set_delta_times()
# print(a_cascade.get_tweet_times())
# a_cascade.print_delta_times()
# a_cascade.plot_delta_times_distribution()
#
# #plot get_tweet_times
# tweet_times = a_cascade.get_tweet_times()
# sns.displot(tweet_times, kind="kde")
# plt.show()
