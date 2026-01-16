# -*- coding: utf-8 -*
'''
This is a supporting library with the code of the model.

Reference Paper: 
Kumar, S., Zhang, X., & Leskovec, J. (2019, July). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1269-1278).
Li, X., Zhang, M., Wu, S., Liu, Z., Wang, L., & Yu, P. S. (2020, November). Dynamic graph collaborative filtering. In 2020 IEEE international conference on data mining (ICDM) (pp. 322-331). IEEE.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import time

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

def map_tensor(input_tensor, alpha):
    if input_tensor.shape[1] != 1:
        raise ValueError("the tensor needs to be (batch_size, 1)")
    output_tensor = (1 - alpha) * input_tensor + alpha
    return output_tensor


# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

# THE DCGLive MODULE
class DCGLive(nn.Module):
    def __init__(self, args, num_features, num_users, num_items, num_streamers, num_streamer_features):
        super(DCGLive,self).__init__()

        print ("*** Initializing the DCGLive model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.num_streamers = num_streamers
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.streamer_static_embedding_size = num_streamers
        self.method = args.method

        print ("Initializing user, item and streamer embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_streamer_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))


        rnn_input_size_items = rnn_input_size_users = 2 * self.embedding_dim + 1 + num_features
        rnn_input_size_streamers = 2 * self.embedding_dim + 1 + num_streamer_features
        rnn_input_size_user_with_streamers = 2 * self.embedding_dim + 1 + num_streamer_features



        print ("Initializing user, room, and streamer RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)
        self.streamer_rnn = nn.RNNCell(rnn_input_size_streamers, self.embedding_dim)
        self.user_rnn_with_streamer = nn.RNNCell(rnn_input_size_user_with_streamers, self.embedding_dim) 

        print ("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.prediction_layer_streamer = nn.Linear(self.user_static_embedding_size + self.streamer_static_embedding_size + self.embedding_dim * 2, self.streamer_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim) 
        print ("*** DCGLive initialization complete ***\n\n")

        print("Initializing aggregate layers")
        if self.method == 'mean' or self.method == 'attention':
            self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)

            self.weigh_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_user_with_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.linear_three_with_streamer = nn.Linear(self.embedding_dim, 1, bias=False)
        elif self.method == 'gat':
            self.weigh_item = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_user = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.linear_three1 = nn.Linear(self.embedding_dim, 1, bias=False)
            self.linear_three2 = nn.Linear(self.embedding_dim, 1, bias=False)

            self.weigh_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_user_with_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.linear_three1_with_streamer = nn.Linear(self.embedding_dim, 1, bias=False)
            self.linear_three2_with_streamer = nn.Linear(self.embedding_dim, 1, bias=False)
        elif self.method == 'lstm':
            self.item_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
            self.user_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
            self.weigh_cen = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_adj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

            self.streamer_lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
            self.user_lstm_with_streamer = nn.LSTM(self.embedding_dim, self.embedding_dim, 1, batch_first=True)
            self.weigh_cen_with_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.weigh_adj_with_streamer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    
    def forward(self, user_embeddings, item_embeddings, streamer_embeddings,
    timediffs=None,  features=None, adj_embeddings=None, 
    select=None, CIs = None, alpha = 0.9):
        if CIs != None:
            CIs = map_tensor(CIs, alpha)
            
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features, adj_embeddings], dim=1)
            input1 = input1 * CIs
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features, adj_embeddings], dim=1)
            input2 = input2 * CIs
            user_embedding_output = self.user_rnn(input2, user_embeddings)

            return F.normalize(user_embedding_output)


        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding

        elif select == 'streamer_update':
            input3 = torch.cat([user_embeddings, timediffs, features, adj_embeddings], dim=1)
            streamer_embedding_output = self.streamer_rnn(input3, streamer_embeddings)

            return F.normalize(streamer_embedding_output)

        elif select == 'user_update_with_streamer': 
            input4 = torch.cat([streamer_embeddings, timediffs, features, adj_embeddings], dim=1)
            user_embedding_output = self.user_rnn_with_streamer(input4, user_embeddings)

            return F.normalize(user_embedding_output)



    def aggregate_attention(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_update':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
            alpha = self.linear_three(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        elif select == 'item_update':
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
            alpha = self.linear_three(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        elif select == 'user_update_with_streamer':
            user_em = self.weigh_user_with_streamer(center_embedding)
            item_em = self.weigh_streamer(embeddings)   
            alpha = self.linear_three_with_streamer(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        elif select == 'streamer_update':
            user_em = self.weigh_streamer(center_embedding)
            item_em = self.weigh_user_with_streamer(embeddings)
            alpha = self.linear_three_with_streamer(torch.sigmoid(item_em + user_em.unsqueeze(1)))
        else:
            raise ValueError(f"Invalid select value: {select}")

        fin_em = torch.sum(alpha*embeddings*mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_gat(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_upate':
            user_em = self.weigh_user(center_embedding)
            item_em = self.weigh_item(embeddings)
        elif select == 'item_update':
            user_em = self.weigh_item(center_embedding)
            item_em = self.weigh_user(embeddings)
        alpha = torch.nn.LeakyReLU()(self.linear_three1(item_em).squeeze(-1) + self.linear_three2(user_em))
        zero_vec = -9e15 * torch.ones_like(mask).float().cuda()
        attention = torch.softmax(torch.where(mask.cuda()>0, alpha, zero_vec), dim=1).unsqueeze(-1)
        fin_em = torch.sum(attention * embeddings * mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return fin_em

    def aggregate_lstm(self, embeddings, length_mask, max_length, center_embedding, select=None, train=True):
        if select == 'user_upate':
            out, _ = self.user_lstm(embeddings)
        elif select == 'item_update':
            out, _ = self.item_lstm(embeddings)
        lstm_em = out[torch.arange(embeddings.shape[0]), length_mask-1, :]
        fin_em = self.weigh_cen(center_embedding) + self.weigh_adj(lstm_em)
        return fin_em

    def aggregate_mean(self, embeddings, length_mask, max_length, center_embedding, select=None):
        mask = torch.arange(max_length)[None, :] < length_mask[:, None]
        if select == 'user_upate':
            em = self.weigh_item(embeddings)
        elif select == 'item_update':
            em = self.weigh_user(embeddings)
        em_mean = torch.div(torch.sum(em.mul(mask.unsqueeze(2).float().cuda()), 1) + center_embedding, length_mask.unsqueeze(1).float().cuda()+1)
        return em_mean


    def context_convert(self, embeddings, timediffs, features):
        time_embed = self.embedding_layer(timediffs)  
        new_embeddings = embeddings * (1 + time_embed)
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

    def predict_streamer_embedding(self, user_embeddings):
        X_out = self.prediction_layer_streamer(user_embeddings)
        return X_out        

def normalize_sequence(sequence):
    min_val = 0
    max_val = max(sequence)
    normalized_sequence = [(x - min_val) / (max_val - min_val) for x in sequence]
    return normalized_sequence

def exponent_f(x):
    return math.exp(-x)

def kappa_f(x_list, timestamp):
    all_less_than_timestamp = all(x <= timestamp for x in x_list)
    exp_values = [exponent_f(timestamp - x) for x in x_list]
    sum_exp = sum(exp_values)
    result_list = [value / sum_exp for value in exp_values]
    return result_list


def jaccard_similarity(A, B):
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    return intersection / union if union != 0 else 0

def get_latest_timestamps(adj_set, adj_timestamp_set, adj_dict, b, state, max_length):
    adj_list = []
    timestamp_list = []
    end_timestamp_list = []
    adj_adj = []

    adj_timestamp_list = list(adj_timestamp_set)
    sorted_adj_timestamp_list = sorted(adj_timestamp_list, key=lambda x: x[1], reverse=True)
    latest = sorted_adj_timestamp_list[:max_length]
    adj_list = [item[0] for item in latest]
    timestamp_list = [item[1] for item in latest]
    adj_adj = [adj_dict[id_] for id_ in adj_list]
    if state:
        end_timestamp_list = [item[2] for item in latest]

    return adj_list, timestamp_list, adj_adj, end_timestamp_list

def min_max_scale(data):
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0 for _ in data]
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data

def calculate_overlap(start_list, end_list, start, end):
    overlap_list = []
    for s, e in zip(start_list, end_list):
        overlap_start = max(s, start)
        overlap_end = min(e, end)
        if overlap_start < overlap_end:
            overlap_list.append(overlap_end - overlap_start)
        else:
            overlap_list.append(0)
    return min_max_scale(overlap_list)

def LiveCIU(a, a_adj, a_adj_timestamp, b, b_adj, timestamp, adj_dict, max_length):
    tbatch_size = len(a)
    CIs = []

    for i in range(tbatch_size):
        _a = a[i]
        _b = b[i]
        _timestamp = timestamp[i]
        _a_adj, _a_adj_timestamp, _a_adj_adj, _ = get_latest_timestamps(a_adj[i], a_adj_timestamp[i],  adj_dict, _b, 0, max_length)
        if len(_a_adj) == 0:
            CIs.append(1)
            continue        
        _b_adj = b_adj[i]
        kappa_list = kappa_f(_a_adj_timestamp, _timestamp)

        CI = 0.0
        b_adj_set = set(_b_adj)
        b_adj_set.add(_b)

        a_adj_adj_sets = [set(adj) for adj in _a_adj_adj]

        for l, adj_timestamp_element in enumerate(_a_adj_timestamp):
            kappa = kappa_list[l]
            adj_adj_set = a_adj_adj_sets[l]
            adj_adj_set.add(_a_adj[l])
            simi = jaccard_similarity(adj_adj_set, b_adj_set)
            CI += kappa * simi
        CIs.append(CI)

    return CIs

def LiveCIR(a, a_adj, a_adj_timestamp, b, b_adj, timestamp, end_timestamp, adj_dict, max_length):
    tbatch_size = len(a)
    CIs = []

    for i in range(tbatch_size):
        _a = a[i]
        _b = b[i]
        _start_timestamp = timestamp[i]
        _end_timestamp = end_timestamp[i]
        _a_adj, _a_adj_timestamp, _a_adj_adj, _a_adj_endtimestamp = get_latest_timestamps(a_adj[i], a_adj_timestamp[i],  adj_dict, _b, 1, max_length)
        if len(_a_adj) == 0:
            CIs.append(1)
            continue
        owt_list = calculate_overlap(_a_adj_timestamp, _a_adj_endtimestamp, _start_timestamp, _end_timestamp)
        _b_adj = b_adj[i]
        kappa_list = kappa_f(_a_adj_timestamp, _start_timestamp)

        CI = 0.0
        b_adj_set = set(_b_adj)
        b_adj_set.add(_b)

        a_adj_adj_sets = [set(adj) for adj in _a_adj_adj]

        for l, adj_timestamp_element in enumerate(_a_adj_timestamp):
            kappa = kappa_list[l]
            adj_adj_set = a_adj_adj_sets[l]
            adj_adj_set.add(_a_adj[l])
            simi = jaccard_similarity(adj_adj_set, b_adj_set)
            owt = owt_list[l]
            CI += kappa * simi * (1 + owt)

        CIs.append(CI)

    return CIs


def adj_pad(adj_seq):
    adjs = []
    length = [len(seq) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(list(seq) + (max_length - len(seq))*[0])
    return adjs, length, max_length

def adj_sample(adj_seq, sam_l):
    adjs = []
    length = [len(seq[:sam_l]) for seq in adj_seq]
    max_length = max(length)
    for seq in adj_seq:
        adjs.append(seq[::-1][:sam_l] + (max_length - len(seq[:sam_l]))*[0])
    return adjs, length, max_length



# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_endtimestamp, current_tbatches_item_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs
    global current_tbatches_user_adj_item, current_tbatches_item_adj_user  
    global current_tbatches_streamer, current_tbatches_streamer_feature, current_tbatches_streamer_interaction_label, current_tbatches_previous_streamer, current_tbatches_streamer_timediffs
    global current_tbatches_streamer_rooms 
    global current_tbatches_user_adj_streamer, current_tbatches_streamer_adj_user
    global current_tbatches_user_timediffs_streamer 
    global current_tbatches_user_adj_timestamp, current_tbatches_item_adj_timestamp

    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list) 
    current_tbatches_item = defaultdict(list) 
    current_tbatches_timestamp = defaultdict(list) 
    current_tbatches_endtimestamp = defaultdict(list)
    current_tbatches_item_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_streamer = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_adj_timestamp = defaultdict(list)
    current_tbatches_item_adj_timestamp = defaultdict(list)

    current_tbatches_user_adj_item = defaultdict(list)
    current_tbatches_item_adj_user = defaultdict(list)

    # streamer
    current_tbatches_streamer = defaultdict(list)
    current_tbatches_streamer_feature = defaultdict(list)
    current_tbatches_streamer_interaction_label = defaultdict(list)
    current_tbatches_previous_streamer = defaultdict(list)
    current_tbatches_streamer_timediffs = defaultdict(list)
    current_tbatches_streamer_rooms = defaultdict(list)

    # adj of streamer
    current_tbatches_user_adj_streamer = defaultdict(list)
    current_tbatches_streamer_adj_user = defaultdict(list)



    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer_UR, optimizer_US, args, epoch, 
                user_embeddings, item_embeddings, streamer_embeddings, train_end_idx, 
                user_adj, item_adj, user_adj_streamer, streamer_adj,
                user_adj_timestamp, item_adj_timestamp, 
                user_embeddings_time_series=None, item_embeddings_time_series=None, streamer_embeddings_time_series=None,
                path=PATH):
    print ("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'streamer_embeddings': streamer_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_UR' : optimizer_UR.state_dict(),
            'optimizer_US' : optimizer_US.state_dict(),
            'train_end_idx': train_end_idx,
            'user_adj': user_adj,
            'item_adj': item_adj,
            'user_adj_streamer': user_adj_streamer,
            'streamer_adj': streamer_adj,
            'user_adj_timestamp': user_adj_timestamp,
            'item_adj_timestamp': item_adj_timestamp
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()
    directory = os.path.join(path, 'saved_models/%s/' % args.network)

    if not os.path.exists(directory):
        os.makedirs(directory)
    if args.l2u == 1.0 and args.l2i == 1.0:
        filename = os.path.join(directory,
                                "adj_checkpoint.%s.%s.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (
                                args.model, args.method, epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length))
    else:
        filename = os.path.join(directory,
                                "adj_checkpoint.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (args.model, args.method, args.l2u, args.l2i,  epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length))

    torch.save(state, filename)
    print ("*** Saved embeddings and model to file: %s ***\n\n" % filename)


def check_ep(model, args, epoch):
    modelname = args.model
    dic = 'saved_models/'

    if args.l2u == 1.0 and args.l2i == 1.0:
        filename = PATH + dic +"%s/adj_checkpoint.%s.%s.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (
            args.network, modelname, model.method, epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length)
    else:
        filename = PATH + dic + "%s/adj_checkpoint.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (
        args.network, modelname, model.method, args.l2u, args.l2i, epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length)

    return os.path.isfile(filename)

# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer_UR, optimizer_US, args, epoch):
    modelname = args.model
    dic = 'saved_models/'

    if args.l2u == 1.0 and args.l2i == 1.0:
        filename = PATH + dic +"%s/adj_checkpoint.%s.%s.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (
            args.network, modelname, model.method, epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length)
    else:
        filename = PATH + dic + "%s/adj_checkpoint.%s.%s.user%.1f.item%.1f.ep%d.tp%.1f.al%.1f.CIm%d.pth.tar" % (
        args.network, modelname, model.method, args.l2u, args.l2i, epoch, args.train_proportion, args.alpha_LiveCI, args.CI_max_length)
    checkpoint = torch.load(filename)
    print ("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    streamer_embeddings = Variable(torch.from_numpy(checkpoint['streamer_embeddings']).cuda())
    user_adj = checkpoint['user_adj']
    item_adj = checkpoint['item_adj']
    user_adj_streamer = checkpoint['user_adj_streamer']
    streamer_adj = checkpoint['streamer_adj']
    user_adj_timestamp = checkpoint['user_adj_timestamp']
    item_adj_timestamp = checkpoint['item_adj_timestamp']

    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
        streamer_embeddings_time_series = Variable(torch.from_numpy(checkpoint['streamer_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None
        streamer_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer_UR.load_state_dict(checkpoint['optimizer_UR'])
    optimizer_US.load_state_dict(checkpoint['optimizer_US'])

    return [model, optimizer_UR, optimizer_US, user_embeddings, item_embeddings, streamer_embeddings,
            user_adj, item_adj, user_adj_streamer, streamer_adj,
            user_adj_timestamp, item_adj_timestamp,
            user_embeddings_time_series, item_embeddings_time_series, streamer_embeddings_time_series,
            train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_free_gpu(seed = 1754066736):
    current_time = int(time.time())
    # seed = current_time
    seed = 1758720383
    set_seed(seed)
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpus = gpu_stats.gpus  
    if not gpus:
        return None  
    
    mem = []
    for gpu in gpus:
        mem.append(gpu.memory_used)
    
    selected_index = np.argmin(mem)
    print("-----", gpus)
    print("-----", selected_index)
    print("-----seed:", seed)
    return str(gpus[selected_index].index), seed