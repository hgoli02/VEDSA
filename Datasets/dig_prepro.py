import numpy as np
import csv
import pandas as pd

user_follower_count = {}
#read digg_friends.csv
df = pd.read_csv('Datasets/digg_friends.csv', header=None)
df.columns = ['mutual','friend_date','user_id', 'friend_id']
df = df.drop(['friend_date'], axis=1)

# #caclulate follower count
for index, row in df.iterrows():
    if row['user_id'] in user_follower_count:
        #check if mutal
        if row['mutual'] == 1:
            if row['friend_id'] in user_follower_count:
                user_follower_count[row['friend_id']] += 1
            else:
                user_follower_count[row['friend_id']] = 1
        user_follower_count[row['user_id']] += 1
    else:
        if row['mutual'] == 1:
            if row['friend_id'] in user_follower_count:
                user_follower_count[row['friend_id']] += 1
            else:
                user_follower_count[row['friend_id']] = 1
        user_follower_count[row['user_id']] = 1

print("finished calculating follower count")

#read digg_votes.csv
df = pd.read_csv('Datasets/digg_votes1.csv', header=None)
df.columns = ['vote_date', 'voter_id', 'story_id']
#add new column called follower count
df['follower_count'] = 0

print("finished reading digg_votes1.csv")

print(df.shape)
print(df.head())
i = 0
for index, row in df.iterrows():
    i += 1
    if row['voter_id'] in user_follower_count:
        #add follower count to df
        df.at[index, 'follower_count'] = user_follower_count[row['voter_id']]
    if i % 100000 == 0:
        print(i)

#drop voter_id
df = df.drop(['voter_id'], axis=1)

#write to csv
df.to_csv('digg_seisimic.csv', index=False)
