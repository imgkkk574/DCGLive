# -*- coding: utf-8 -*

'''
This is a supporting library for the loading the data.

Reference Paper: 
Kumar, S., Zhang, X., & Leskovec, J. (2019, July). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1269-1278).
Li, X., Zhang, M., Wu, S., Liu, Z., Wang, L., & Yu, P. S. (2020, November). Dynamic graph collaborative filtering. In 2020 IEEE international conference on data mining (ICDM) (pp. 322-331). IEEE.
'''

from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
# import cPickle
import argparse
from sklearn.preprocessing import scale


# LOAD THE NETWORK
def load_network(args, time_scaling=True):

    network = args.network
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    streamer_sequence = []
    item_feature_sequence = []
    streamer_feature_sequence = []


    timestamp_sequence = []
    end_time_sequence = []
    start_timestamp = None
    y_true_labels = []
    streamer_interaction_label = [] 

    print ("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()

    for cnt, l in enumerate(f):
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[2])
        streamer_sequence.append(ls[1])

        if start_timestamp is None:
            start_timestamp = float(ls[3])
        timestamp_sequence.append(float(ls[3]) - start_timestamp) 
        end_time_sequence.append(float(ls[4]) - start_timestamp)

        y_true_labels.append(int(ls[5])) # label = 1 at state change, 0 otherwise

        item_feature_sequence.append([float(ls[6])])
        streamer_feature_sequence.append(list(map(float, ls[7:])))  
        if all(int(x) == 0 for x in ls[7:]):
            streamer_interaction_label.append(0)
        else:
            streamer_interaction_label.append(1)
    f.close()

    user_sequence = np.array(user_sequence)            # user
    item_sequence = np.array(item_sequence)            # item
    streamer_sequence = np.array(streamer_sequence)    # streamer
    timestamp_sequence = np.array(timestamp_sequence)  

    print ("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])  
        item_current_timestamp[item] = timestamp         
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]      

    print ("Formating streamer sequence")
    nodeid = 0
    streamer2id = {}
    streamer_timedifference_sequence = []
    streamer_current_timestamp = defaultdict(float)
    streamer_current_timediff = defaultdict(float)

    for cnt, streamer in enumerate(streamer_sequence):
        if streamer not in streamer2id:
            streamer2id[streamer] = nodeid
            nodeid += 1
        if streamer_interaction_label[cnt] == 0:
            last_timediff = streamer_current_timediff[streamer] 
            streamer_timedifference_sequence.append(last_timediff) 

        else: 
            timestamp = timestamp_sequence[cnt] 
            timediff_now = timestamp - streamer_current_timestamp[streamer] 
            streamer_timedifference_sequence.append(timediff_now) 
            streamer_current_timestamp[streamer] = timestamp 
            streamer_current_timediff[streamer] = timediff_now 

    num_streamers = len(streamer2id)
    streamer_sequence_id = [streamer2id[streamer] for streamer in streamer_sequence]     

    print ("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = [] 
    user_current_timestamp = defaultdict(float) 

    user_timedifference_sequence_streamer = [] 
    user_current_timestamp_streamer = defaultdict(float) 

    user_previous_itemid_sequence = []
    user_previous_streamerid_sequence = [] 
    user_latest_itemid = defaultdict(lambda: num_items)  
    user_latest_streamerid = defaultdict(lambda: num_streamers)  
    first_label = 0
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:       
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt] 
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user]) 
        user_current_timestamp[user] = timestamp                                      
        user_previous_itemid_sequence.append(user_latest_itemid[user])                
        user_latest_itemid[user] = item2id[item_sequence[cnt]]                       
        if streamer_interaction_label[cnt] == 1:
            if first_label == 0:
                user_timedifference_sequence_streamer.append(0)
            else:
                user_timedifference_sequence_streamer.append(timestamp - user_current_timestamp_streamer[user]) 
            user_current_timestamp_streamer[user] = timestamp 
            user_previous_streamerid_sequence.append(user_latest_streamerid[user])       
            user_latest_streamerid[user] = streamer2id[streamer_sequence[cnt]]           
            first_label = 1
        else:
            if first_label == 1:
                last_user_timedifference_streamer = user_timedifference_sequence_streamer[-1]
                user_timedifference_sequence_streamer.append(last_user_timedifference_streamer) 
                last_user_precious_streamerid = user_previous_streamerid_sequence[-1]
                user_previous_streamerid_sequence.append(last_user_precious_streamerid)
            else: 
                user_timedifference_sequence_streamer.append(0)
                user_previous_streamerid_sequence.append(num_streamers)

    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print ("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)
        streamer_timedifference_sequence = scale(np.array(streamer_timedifference_sequence) + 1)
        user_timedifference_sequence_streamer = scale(np.array(user_timedifference_sequence_streamer) + 1)

    print ("*** Network loading completed ***\n\n")

    return [ user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, user_timedifference_sequence_streamer, user_previous_streamerid_sequence,
    item2id, item_sequence_id, item_timedifference_sequence, 
    streamer2id, streamer_sequence_id, streamer_timedifference_sequence, 
    timestamp_sequence, end_time_sequence, 
    item_feature_sequence, streamer_feature_sequence,
    y_true_labels, streamer_interaction_label]
