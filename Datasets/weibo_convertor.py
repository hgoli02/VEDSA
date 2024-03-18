#i have a weibo_netwrok.txt file read the first line of it and print it

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# weibo_network.txt

# This file describes the Sina Weibo sub-network at the very first timestamp.
# First line consists of two integers, representing the number of users N and number of the follow relationships M respectively.
# In the following N lines, each line starts with an integer v1_id, representing the user_id of user v1, followed by another integer k.
# And the following 2k numbers describes the users "FOLLOWED" by v1, each represented by a user id v2_id and a number indicating the type of relationship.
# In weibo data set, "1" indicates that user v1 and v2 have a reciprocal FOLLOW relationship, while "0" indicates that their relationship is not reciprocal.

#create the folloinwg and save it as weibo-procesed.txt
#date,story_id,follower_count

#weibo_cascades.txt
#This file describes the Sina Weibo cascades.
#Each line represents a cascade, starting with a user id, followed by a timestamp which are separated by a space.

#create a dictionary with key as user_id and value as number of followers from weibo_network.txt
#then create a file and save it as csv each entry is a cascade with timestamp,follower_count, and story_id

#weibo_network.txt
weibo_network = open("weibo_network.txt", "r")
n, m = map(int, weibo_network.readline().split())
print(n,m)

user_follower = {}
for i in tqdm(range(n)):
    line = weibo_network.readline().split()
    user_follower[int(line[0])] = int(line[1])
weibo_network.close()

print('Finished reading weibo_network.txt')

#weibo_cascades.txt
weibo_cascades = open("Datasets\weibo_cascades.txt", "r")
#i want output as csv
weibo_cascades_processed = open("Datasets\weibo_cascades_processed.csv", "w")
weibo_cascades_processed.write("vote_date,story_id,follower_count\n")
for i,line in tqdm(enumerate(weibo_cascades)):
    line = line.split()
    story_id = i
    timestamp = line
    for entry in timestamp:
        id,timestamp = entry.split(',')
        follower_count = user_follower[int(id)]
        weibo_cascades_processed.write(str(timestamp)+","+str(story_id)+","+str(follower_count)+"\n")
weibo_cascades_processed.close()
weibo_cascades.close()
