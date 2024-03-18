import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


class Tweet:
    def __init__(self, twitter, time):
        self.twitter = twitter
        self.time = time

    def print(self):
        print(self.time, self.twitter, sep=',', end=' ')


class Cascade:
    def __init__(self, tweets):
        self.tweets = tweets
        self.delta_times = []

    def sort_tweets(self):
        self.tweets = sorted(self.tweets, key=lambda tweet: tweet.time)

    def set_delta_times(self):
        self.sort_tweets()
        for i in range(1, len(self.tweets)):
            self.delta_times.append(self.tweets[i].time - self.tweets[i - 1].time)

        self.delta_times = self.delta_times[1:len(self.delta_times)]

    def get_tweet_times(self):
        l = []
        for tweet in self.tweets[1:len(self.tweets) - 1]:
            l.append(tweet.time)
        return l

    def print(self):
        for tweet in self.tweets:
            tweet.print()
        print()

    def print_delta_times(self):
        for t in self.delta_times:
            print(t, end=' ')
        print()

    def plot_delta_times_distribution(self):
        sns.displot(self.delta_times, kind="kde")
        plt.show()


def create_time_windows(cascades, time_window_len=100):
    windowed_cascades = []

    for cascade in cascades:
        if cascade.get_tweet_times()[-1] == 0:
            bins = 1
        else:
            bins = ceil(cascade.get_tweet_times()[-1] / time_window_len)
        # print('bins:', bins)
        hist_y, hist_x = np.histogram(cascade.get_tweet_times(), bins=bins)
        hist_y = hist_y.tolist()
        hist_x = hist_x.tolist()
        #plt.plot(hist_y)
        #plt.show()
        windowed_cascades.append(hist_y)
        # print(hist_y[0:100])
        # print(hist_x[0:100])

    return windowed_cascades


def get_peak_windows(windowed_cascades):
    peak_times = []
    for cascade in windowed_cascades:
        peak_times.append(np.argmax(cascade))
        # print("-->", np.argmax(cascade))
    return np.array(peak_times)


def get_peak_labels(windowed_cascades, burst_labels, peak_times):
    cascades_labels = []

    counter = 0
    for cascade in windowed_cascades:
        if burst_labels[counter]:
            cascades_labels.append(
                np.append(np.zeros(peak_times[counter]),
                          np.ones(len(cascade) - peak_times[counter]))
            )

        else:
            cascades_labels.append(np.zeros(len(cascade)))
        counter += 1

    return cascades_labels
