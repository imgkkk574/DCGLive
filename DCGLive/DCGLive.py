# -*- coding: utf-8 -*
'''
This code trains the DCGLive model for the kuailive dataset.
The task is: next room prediction.

How to run: 
$ python DCGLive.py --network kuailive5_lr_30 --model DCGLive --epochs 50 --method attention --embedding_dim 128 --alpha_LiveCI 0.9

Reference Paper: 
Kumar, S., Zhang, X., & Leskovec, J. (2019, July). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1269-1278).
Li, X., Zhang, M., Wu, S., Liu, Z., Wang, L., & Yu, P. S. (2020, November). Dynamic graph collaborative filtering. In 2020 IEEE international conference on data mining (ICDM) (pp. 322-331). IEEE.
'''
from library_data import *
import library_models as lib
from library_models import *
from IPython import embed
import time
import datetime

torch.autograd.set_detect_anomaly(True)

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="DCGLive", help='Model name to save output in file')
parser.add_argument('--method', default="attention", help='The way of aggregate adj')
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--l2us', type=float, default=1.0, help='regular coefficient of user with streamer')
parser.add_argument('--l2s', type=float, default=1.0, help='regular coefficient of streamer')
parser.add_argument('--l2uu', type=float, default=1.0, help='regular coefficient of user after streamer')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--lr_US', default=5e-4, type=float, help='Learning rate for optimizing US')
parser.add_argument('--lr_UR', default=1e-3, type=float, help='Learning rate for optimizing UR')
parser.add_argument('--alpha_LiveCI', default=0.9, type=float, help='Weight of LiveCI')
parser.add_argument('--CI_max_length', default=30, type=int, help='The max number of neighbors to compute LiveCI')
args = parser.parse_args()
print(args)

args.datapath = "data/%s.csv" % args.network 
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
if args.gpu == -1:
    args.gpu, seed = select_free_gpu() 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, user_timediffs_sequence_streamer, user_previous_streamerid_sequence,
    item2id, item_sequence_id, item_timediffs_sequence, 
    streamer2id, streamer_sequence_id, streamer_timediffs_sequence, 
    timestamp_sequence, endtime_sequence,
    item_feature_sequence, streamer_feature_sequence, y_true, streamer_interaction_label_sequence] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_streamers = len(streamer2id) + 1 # one extra item for "none-of-these"
normalized_timestamp_sequence = normalize_sequence(timestamp_sequence)
normalized_end_timestamp_sequence = normalize_sequence(endtime_sequence)


num_features = len(item_feature_sequence[0])
num_streamer_features = len(streamer_feature_sequence[0])


id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}
id2streamer = {v: k for k, v in streamer2id.items()}


true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error.

embed() 
print(f"*** Network statistics:\n  {num_users} users\n  {num_items} items\n  {num_interactions} interactions\n  {sum(y_true)}/{len(y_true)} true labels ***\n")
# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
timespan = timestamp_sequence[-1] - timestamp_sequence[0]   
tbatch_timespan = timespan / args.span_num                  

# INITIALIZE MODEL AND PARAMETERS
model = DCGLive(args, num_features, num_users, num_items, num_streamers, num_streamer_features).cuda()
weight = torch.Tensor([1, true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
initial_streamer_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))

model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding
model.initial_streamer_embedding = initial_streamer_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
streamer_embeddings = initial_streamer_embedding.repeat(num_streamers, 1) # initialize all streamers to the same embedding

item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings
streamer_embedding_static = Variable(torch.eye(num_streamers).cuda()) # one-hot vectors for static embeddings

# INITIALIZE MODEL


UR_para = []
US_para = []

for name, param in model.named_parameters():
    # print(f"Parameter name: {name}")
    if 'streamer' in name:
        print(name, param.shape)
        US_para.append(param)
    else:
        print("---", name, param.shape)
        UR_para.append(param)




print(len(list(model.parameters())))
print(len(UR_para))
print(len(US_para))

learning_rate_r = args.lr_UR
learning_rate_s = args.lr_US
CI_max_length = args.CI_max_length
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.l2)

optimizer_UR = optim.Adam(UR_para, lr=learning_rate_r, weight_decay=args.l2)
optimizer_US = optim.Adam(US_para, lr=learning_rate_s, weight_decay=args.l2)


# RUN THE DCGLive MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, DCGLive USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the DCGLive model for %d epochs ***" % args.epochs)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
# with trange(args.epochs) as progress_bar1:


user_adj = None
item_adj = None
user_adj_timestamp = None
item_adj_timestamp = None
streamer_adj = None
user_adj_streamer = None 

seen_streamer = None
seen_item = None
streamer_rooms = None
start = 0

for ep in range(args.epochs): 
    if check_ep(model, args, ep):
        continue
    else:
        print(ep, "!")
        start = ep
        if ep == 0:
            break
        # LOAD THE MODEL
        model, optimizer_UR, optimizer_US, \
        user_embeddings_dystat, item_embeddings_dystat, streamer_embeddings_dystat, \
        user_adj, item_adj, user_adj_streamer, streamer_adj, \
        user_adj_timestamp, item_adj_timestamp, \
        user_embeddings_timeseries, item_embeddings_timeseries, streamer_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer_UR, optimizer_US, args, ep - 1)
        # LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
        item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
        item_embeddings = item_embeddings.clone()
        item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
        item_embeddings_static = item_embeddings_static.clone()

        user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
        user_embeddings = user_embeddings.clone()
        user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
        user_embeddings_static = user_embeddings_static.clone()

        streamer_embeddings = streamer_embeddings_dystat[:, :args.embedding_dim]
        streamer_embeddings = streamer_embeddings.clone()
        streamer_embeddings_static = streamer_embeddings_dystat[:, args.embedding_dim:]
        streamer_embeddings_static = streamer_embeddings_static.clone()
        break


for ep in range(start, args.epochs): 

    seen_streamer = set() 
    seen_item = set() 
    streamer_rooms = defaultdict(list) 

    #progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

    # INITIALIZE EMBEDDING TRAJECTORY STORAGE
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    streamer_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

    optimizer_US.zero_grad()
    optimizer_UR.zero_grad()
    reinitialize_tbatches() 
    total_loss, loss, total_interaction_count = 0, 0, 0
    US_loss, total_US_loss = 0, 0 

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False
    user_adj = defaultdict(set)  
    item_adj = defaultdict(set)  
    user_adj_timestamp = defaultdict(set)
    item_adj_timestamp = defaultdict(set)
    user_adj_streamer = defaultdict(set) 
    streamer_adj = defaultdict(set) 


    # TRAIN TILL THE END OF TRAINING INTERACTION IDX
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(ep, formatted_time)

    for j in range(train_end_idx): 

        # READ INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        item_feature = item_feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        # streamer
        streamerid = streamer_sequence_id[j] 
        streamer_feature = streamer_feature_sequence[j]
        streamer_timediff = streamer_timediffs_sequence[j]
        streamer_interaction_label = streamer_interaction_label_sequence[j]

        user_timediff_streamer = user_timediffs_sequence_streamer[j] # user和streamer的timediff
        current_timestamp = normalized_timestamp_sequence[j]
        end_timestamp = normalized_end_timestamp_sequence[j]


        user_adj[userid].add(itemid)  
        item_adj[itemid].add(userid)
        user_adj_timestamp[userid].add((itemid, current_timestamp))
        item_adj_timestamp[itemid].add((userid, current_timestamp, end_timestamp))
        if streamer_interaction_label == 1:
            user_adj_streamer[userid].add(streamerid)
            streamer_adj[streamerid].add(userid)



        # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[streamerid]) + 1   
        lib.tbatchid_user[userid] = tbatch_to_insert                                       
        lib.tbatchid_item[streamerid] = tbatch_to_insert
        lib.current_tbatches_user[tbatch_to_insert].append(userid)    
        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
        lib.current_tbatches_item_feature[tbatch_to_insert].append(item_feature)
        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
        # streamer
        lib.current_tbatches_streamer[tbatch_to_insert].append(streamerid)
        lib.current_tbatches_streamer_feature[tbatch_to_insert].append(streamer_feature)
        lib.current_tbatches_streamer_interaction_label[tbatch_to_insert].append(streamer_interaction_label)
        lib.current_tbatches_streamer_timediffs[tbatch_to_insert].append(streamer_timediff)
        lib.current_tbatches_previous_streamer[tbatch_to_insert].append(user_previous_streamerid_sequence[j])
        lib.current_tbatches_user_timediffs_streamer[tbatch_to_insert].append(user_timediff_streamer) 
        lib.current_tbatches_timestamp[tbatch_to_insert].append(normalized_timestamp_sequence[j])
        lib.current_tbatches_endtimestamp[tbatch_to_insert].append(normalized_end_timestamp_sequence[j])


        lib.current_tbatches_user_adj_item[tbatch_to_insert].append(frozenset(user_adj[userid]))  # item
        lib.current_tbatches_item_adj_user[tbatch_to_insert].append(frozenset(item_adj[itemid]))  # user
        lib.current_tbatches_user_adj_timestamp[tbatch_to_insert].append(frozenset(user_adj_timestamp[userid]))  # item
        lib.current_tbatches_item_adj_timestamp[tbatch_to_insert].append(frozenset(item_adj_timestamp[itemid]))  # user
        lib.current_tbatches_user_adj_streamer[tbatch_to_insert].append(frozenset(user_adj_streamer[userid])) 
        lib.current_tbatches_streamer_adj_user[tbatch_to_insert].append(frozenset(streamer_adj[streamerid])) 


        timestamp = timestamp_sequence[j]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES,
        # FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
        # after all interactions in the timespan are converted to t-batchs,
        # forward pass to crate embedding trajectories and calculate prediction loss


        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # ITERATE OVER ALL T-BATCHES
            for i in range(len(lib.current_tbatches_user)):

                item_CIs = LiveCIR(lib.current_tbatches_item[i], lib.current_tbatches_item_adj_user[i], lib.current_tbatches_item_adj_timestamp[i], lib.current_tbatches_user[i], lib.current_tbatches_user_adj_item[i], lib.current_tbatches_timestamp[i], lib.current_tbatches_endtimestamp[i], user_adj, CI_max_length)
                user_CIs = LiveCIU(lib.current_tbatches_user[i], lib.current_tbatches_user_adj_item[i], lib.current_tbatches_user_adj_timestamp[i], lib.current_tbatches_item[i], lib.current_tbatches_item_adj_user[i], lib.current_tbatches_timestamp[i], item_adj, CI_max_length)
                total_interaction_count += len(lib.current_tbatches_interactionids[i])
                tbatch_streamer_interaction_label = list(lib.current_tbatches_streamer_interaction_label[i])
                zero_indices = [index for index, label in enumerate(tbatch_streamer_interaction_label) if label == 0]
                non_zero_indices = [index for index, label in enumerate(tbatch_streamer_interaction_label) if label != 0]


                if len(non_zero_indices) != 0: # need to update streamer embedding

                    ''' *************** US *************** '''
                    item_CIs_us = [item_CIs[index] for index in non_zero_indices]
                    user_CIs_us = [user_CIs[index] for index in non_zero_indices]
                    tbatch_userids_with_streamer = torch.LongTensor([lib.current_tbatches_user[i][index] for index in non_zero_indices]).cuda() # userid
                    tbatch_interactionids_with_streamer = torch.LongTensor([lib.current_tbatches_interactionids[i][index] for index in non_zero_indices]).cuda()
                    tbatch_streamerids_with_streamer = torch.LongTensor([lib.current_tbatches_streamer[i][index] for index in non_zero_indices]).cuda() # streamerid
                    streamer_feature_tensor = Variable(torch.Tensor([lib.current_tbatches_streamer_feature[i][index] for index in non_zero_indices]).cuda()) # streamer feature
                    tbatch_streamerids_previous = torch.LongTensor([lib.current_tbatches_previous_streamer[i][index] for index in non_zero_indices]).cuda()
                    streamer_embedding_previous = streamer_embeddings[tbatch_streamerids_previous,:]  
                    streamer_timediffs_tensor = Variable(torch.Tensor([lib.current_tbatches_streamer_timediffs[i][index] for index in non_zero_indices]).cuda()).unsqueeze(1) # streamer timediff
                    user_streamer_timediffs_tensor_with_streamer = Variable(torch.Tensor([lib.current_tbatches_user_timediffs_streamer[i][index] for index in non_zero_indices]).cuda()).unsqueeze(1) # user timediff
                    tbatch_itemids_with_streamer = torch.LongTensor([lib.current_tbatches_item[i][index] for index in non_zero_indices]).cuda() 
                    item_feature_tensor_with_streamer = Variable(torch.Tensor([lib.current_tbatches_item_feature[i][index] for index in non_zero_indices]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_item_timediffs_tensor_with_streamer = Variable(torch.Tensor([lib.current_tbatches_user_timediffs[i][index] for index in non_zero_indices]).cuda()).unsqueeze(1)
                    item_timediffs_tensor_with_streamer = Variable(torch.Tensor([lib.current_tbatches_item_timediffs[i][index] for index in non_zero_indices]).cuda()).unsqueeze(1)
                    tbatch_itemids_previous_with_streamer = torch.LongTensor([lib.current_tbatches_previous_item[i][index] for index in non_zero_indices]).cuda()
                    item_embedding_previous_with_streamer = item_embeddings[tbatch_itemids_previous_with_streamer,:]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input_with_streamer = user_embeddings[tbatch_userids_with_streamer,:] # size (batch_size, dim)
                    streamer_embedding_input_with_streamer = streamer_embeddings[tbatch_streamerids_with_streamer, :]

                    user_projected_embedding_item = model.forward(user_embedding_input_with_streamer, item_embedding_previous_with_streamer, \
                        streamer_embedding_input_with_streamer, timediffs=user_item_timediffs_tensor_with_streamer, features=item_feature_tensor_with_streamer, select='project') # project user_emb for item

                    user_item_embedding_with_streamer = torch.cat([user_projected_embedding_item, item_embedding_previous_with_streamer, item_embedding_static[tbatch_itemids_previous_with_streamer,:], user_embedding_static[tbatch_userids_with_streamer,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING
                    predicted_item_embedding_with_streamer = model.predict_item_embedding(user_item_embedding_with_streamer)

                    # PROJECT USER EMBEDDING TO CURRENT TIME FOR STREAMER
                    user_projected_embedding_streamer = model.forward(user_embedding_input_with_streamer, streamer_embedding_previous, \
                        streamer_embedding_input_with_streamer, timediffs=user_streamer_timediffs_tensor_with_streamer, features=streamer_feature_tensor, select='project')
                    
                    user_streamer_embedding = torch.cat([user_projected_embedding_streamer, streamer_embedding_previous, streamer_embedding_static[tbatch_streamerids_previous,:], user_embedding_static[tbatch_userids_with_streamer,:]], dim=1)
                    # PREDICT NEXT STREAMER EMBEDDING
                    predicted_streamer_embedding = model.predict_streamer_embedding(user_streamer_embedding)
        
                    US_loss += MSELoss(predicted_streamer_embedding, torch.cat([streamer_embedding_input_with_streamer, streamer_embedding_static[tbatch_streamerids_with_streamer,:]], dim=1).detach())

                    # ROOM EMBEDDING INITIALIZATION
                    tbatch_itemids_list_with_streamer = tbatch_itemids_with_streamer.tolist()
                    tbatch_streamerids_list_with_streamer = tbatch_streamerids_with_streamer.tolist() 
                    len_tbatch = len(tbatch_streamerids_list_with_streamer)

                    for b in range(len_tbatch):
                        item_id = tbatch_itemids_list_with_streamer[b]
                        streamer_id = tbatch_streamerids_list_with_streamer[b]
                        if streamer_id in seen_streamer:
                            if item_id not in seen_item: # new room
                                exists_rooms = streamer_rooms[streamer_id] 
                                exits_rooms_emb = item_embeddings[exists_rooms, :]

                                mean_tensor = exits_rooms_emb.mean(dim=0, keepdim=True) 
                                streamer_emb_exists = streamer_embeddings[streamer_id, :].unsqueeze(0)
                                # # print("----streamer_emb_exists: ", streamer_emb_exists, streamer_emb_exists.shape, mean_tensor.shape)
                                combined_mean = (streamer_emb_exists + mean_tensor) / 2
                                item_embeddings[[item_id], :] = combined_mean
                                # item_embeddings[[item_id], :] = mean_tensor # ablation w/o streamer
                                # item_embeddings[[item_id], :] = streamer_emb_exists # ablation w/o mean

                                streamer_rooms[streamer_id].append(item_id) 
                        else: # new streamer
                            streamer_rooms[streamer_id].append(item_id) # 
                        seen_item.add(item_id)
                        seen_streamer.add(streamer_id)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input_with_streamer = item_embeddings[tbatch_itemids_with_streamer,:]
                    loss += MSELoss(predicted_item_embedding_with_streamer, torch.cat([item_embedding_input_with_streamer, item_embedding_static[tbatch_itemids_with_streamer,:]], dim=1).detach())
                    # print("**** predict: ", type(loss))
                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_adj_, user_length_mask, user_max_length = adj_pad([lib.current_tbatches_user_adj_item[i][index] for index in non_zero_indices])
                    item_adj_, item_length_mask, item_max_length = adj_pad([lib.current_tbatches_item_adj_user[i][index] for index in non_zero_indices])

                    user_adj_streamer_, user_length_mask_streamer, user_max_length_streamer = adj_pad([lib.current_tbatches_user_adj_streamer[i][index] for index in non_zero_indices])
                    streamer_adj_, streamer_length_mask, streamer_max_length = adj_pad([lib.current_tbatches_streamer_adj_user[i][index] for index in non_zero_indices])

                    user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]
                    item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]
                    user_adj_streamer_em = streamer_embeddings[torch.LongTensor(user_adj_streamer_).cuda(), :]
                    streamer_adj_em = user_embeddings[torch.LongTensor(streamer_adj_).cuda(), :]

                    if model.method == 'mean':
                        user_adj_embedding_with_streamer = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                user_max_length, user_embedding_input_with_streamer,
                                                                select='user_update')
                        item_adj_embedding_with_streamer = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input_with_streamer,
                                                                select='item_update')
                        
                        user_adj_streamer_embedding = model.aggregate_mean(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer),
                                                                user_max_length_streamer, user_embedding_input_with_streamer,
                                                                select='user_update_with_streamer')
                        streamer_adj_embedding = model.aggregate_mean(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input_with_streamer,
                                                                select='streamer_update')

                    elif model.method == 'attention':
                        user_adj_embedding_with_streamer = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input_with_streamer,
                                                                    select='user_update')
                        item_adj_embedding_with_streamer = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input_with_streamer,
                                                                    select='item_update')
                        user_adj_streamer_embedding = model.aggregate_attention(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer),
                                                                user_max_length_streamer, user_embedding_input_with_streamer,
                                                                select='user_update_with_streamer')
                        streamer_adj_embedding = model.aggregate_attention(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input_with_streamer,
                                                                select='streamer_update')

                    elif model.method == 'gat':
                        user_adj_embedding_with_streamer = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input_with_streamer,
                                                                    select='user_update')
                        item_adj_embedding_with_streamer = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input_with_streamer,
                                                                    select='item_update')
                    elif model.method == 'lstm':
                        user_adj_embedding_with_streamer = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                user_max_length, user_embedding_input_with_streamer,
                                                                select='user_update')
                        item_adj_embedding_with_streamer = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input_with_streamer,
                                                                select='item_update')


                    batch_size = len(item_CIs_us) 
                    # user update with CIs
                    user_CIs_tensor = torch.tensor(user_CIs_us, dtype=torch.float32).view(batch_size, 1).to('cuda')

                    user_embedding_output_with_streamer = model.forward(user_embedding_input_with_streamer, item_embedding_input_with_streamer, streamer_embedding_input_with_streamer,
                                                        timediffs=user_item_timediffs_tensor_with_streamer, features=item_feature_tensor_with_streamer,
                                                        adj_embeddings=user_adj_embedding_with_streamer, select='user_update', CIs = user_CIs_tensor, alpha=args.alpha_LiveCI)

                    # item update with CIs
                    item_CIs_tensor = torch.tensor(item_CIs_us, dtype=torch.float32).view(batch_size, 1).to('cuda')

                    item_embedding_output_with_streamer = model.forward(user_embedding_input_with_streamer, item_embedding_input_with_streamer, streamer_embedding_input_with_streamer,
                                                        timediffs=item_timediffs_tensor_with_streamer, features=item_feature_tensor_with_streamer,
                                                        adj_embeddings=item_adj_embedding_with_streamer, select='item_update', CIs = item_CIs_tensor, alpha=args.alpha_LiveCI)
                    

                    streamer_embedding_output = model.forward(user_embedding_input_with_streamer, streamer_embedding_input_with_streamer, streamer_embedding_input_with_streamer,
                                                        timediffs=streamer_timediffs_tensor, features=streamer_feature_tensor,
                                                        adj_embeddings=streamer_adj_embedding, select='streamer_update')
                    user_embedding_output_after_US = model.forward(user_embedding_output_with_streamer, item_embedding_input_with_streamer, streamer_embedding_input_with_streamer,
                                                        timediffs=user_streamer_timediffs_tensor_with_streamer, features=streamer_feature_tensor,
                                                        adj_embeddings=user_adj_streamer_embedding, select='user_update_with_streamer')

                    #user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    #item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS

                    loss += args.l2i*MSELoss(item_embedding_output_with_streamer, item_embedding_input_with_streamer.detach())
                    loss += args.l2u*MSELoss(user_embedding_output_with_streamer, user_embedding_input_with_streamer.detach())
                    US_loss += args.l2s*MSELoss(streamer_embedding_output, streamer_embedding_input_with_streamer.detach())
                    US_loss += args.l2uu*MSELoss(user_embedding_output_with_streamer, user_embedding_output_after_US)


                    item_embeddings[tbatch_itemids_with_streamer, :] = item_embedding_output_with_streamer
                    user_embeddings[tbatch_userids_with_streamer, :] = user_embedding_output_after_US
                    streamer_embeddings[tbatch_streamerids_with_streamer, :] = streamer_embedding_output
                    # sys.exit(0)
                    ''' *************** UR *************** '''
                    # don't need to update US
                    if len(zero_indices) != 0:
                        item_CIs_ur = [item_CIs[index] for index in zero_indices]
                        user_CIs_ur = [user_CIs[index] for index in zero_indices]
                        tbatch_userids = torch.LongTensor([lib.current_tbatches_user[i][index] for index in zero_indices]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                        tbatch_itemids = torch.LongTensor([lib.current_tbatches_item[i][index] for index in zero_indices]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                        tbatch_interactionids = torch.LongTensor([lib.current_tbatches_interactionids[i][index] for index in zero_indices]).cuda()
                        tbatch_streamerids = torch.LongTensor([lib.current_tbatches_streamer[i][index] for index in zero_indices]).cuda()
                        item_feature_tensor = Variable(torch.Tensor([lib.current_tbatches_item_feature[i][index] for index in zero_indices]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                        user_timediffs_tensor = Variable(torch.Tensor([lib.current_tbatches_user_timediffs[i][index] for index in zero_indices]).cuda()).unsqueeze(1)
                        item_timediffs_tensor = Variable(torch.Tensor([lib.current_tbatches_item_timediffs[i][index] for index in zero_indices]).cuda()).unsqueeze(1)
                        tbatch_itemids_previous = torch.LongTensor([lib.current_tbatches_previous_item[i][index] for index in zero_indices]).cuda()
                        item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                        # PROJECT USER EMBEDDING TO CURRENT TIME
                        user_embedding_input = user_embeddings[tbatch_userids,:] # size (batch_size, dim)
                        streamer_embedding_input = streamer_embeddings[tbatch_streamerids, :]
                        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, streamer_embedding_input, timediffs=user_timediffs_tensor, features=item_feature_tensor, select='project') # 这里只用user_embedding 传的item_embedding没用
                        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                        # PREDICT NEXT ITEM EMBEDDING
                        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                        # ROOM EMBEDDING INITIALIZATION
                        tbatch_itemids_list_with_streamer = tbatch_itemids_with_streamer.tolist()
                        tbatch_streamerids_list_with_streamer = tbatch_streamerids_with_streamer.tolist() 
                        len_tbatch = len(tbatch_streamerids_list_with_streamer)

                        for b in range(len_tbatch):
                            item_id = tbatch_itemids_list_with_streamer[b]
                            streamer_id = tbatch_streamerids_list_with_streamer[b]
                            if streamer_id in seen_streamer:
                                if item_id not in seen_item: # new room
                                    exists_rooms = streamer_rooms[streamer_id] 
                                    exits_rooms_emb = item_embeddings[exists_rooms, :]

                                    mean_tensor = exits_rooms_emb.mean(dim=0, keepdim=True) 
                                    streamer_emb_exists = streamer_embeddings[streamer_id, :].unsqueeze(0)
                                    combined_mean = (streamer_emb_exists + mean_tensor) / 2
                                    item_embeddings[[item_id], :] = combined_mean
                                    # item_embeddings[[item_id], :] = mean_tensor # ablation w/o streamer
                                    # item_embeddings[[item_id], :] = streamer_emb_exists # ablation w/o mean

                                    streamer_rooms[streamer_id].append(item_id) 
                            else: # new streamer
                                streamer_rooms[streamer_id].append(item_id) 
                            seen_item.add(item_id)
                            seen_streamer.add(streamer_id)

                        item_embedding_input = item_embeddings[tbatch_itemids,:]
                        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                        user_adj_, user_length_mask, user_max_length = adj_pad([lib.current_tbatches_user_adj_item[i][index] for index in zero_indices])
                        item_adj_, item_length_mask, item_max_length = adj_pad([lib.current_tbatches_item_adj_user[i][index] for index in zero_indices])

                        user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]
                        item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]
                        if model.method == 'mean':
                            # user_adj_embedding = model.aggregate(item_embeddings, lib.current_tbatches_user_adj_item[i], select='user_update')
                            # item_adj_embedding = model.aggregate(user_embeddings, lib.current_tbatches_item_adj_user[i], select='item_update')
                            user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input,
                                                                    select='user_update')
                            item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')
                        elif model.method == 'attention':
                            user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                        user_max_length, user_embedding_input,
                                                                        select='user_update')
                            item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                        item_max_length, item_embedding_input,
                                                                        select='item_update')
                        elif model.method == 'gat':
                            user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                        user_max_length, user_embedding_input,
                                                                        select='user_update')
                            item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                        item_max_length, item_embedding_input,
                                                                        select='item_update')
                        elif model.method == 'lstm':
                            user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input,
                                                                    select='user_update')
                            item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')



                        batch_size = len(item_CIs_ur)
                        # user update with CIs
                        user_CIs_tensor = torch.tensor(user_CIs_ur, dtype=torch.float32).view(batch_size, 1).to('cuda')

                        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                                            timediffs=user_timediffs_tensor, features=item_feature_tensor,
                                                            adj_embeddings=user_adj_embedding, select='user_update', CIs = user_CIs_tensor, alpha=args.alpha_LiveCI)
                        # item update with CIs
                        item_CIs_tensor = torch.tensor(item_CIs_ur, dtype=torch.float32).view(batch_size, 1).to('cuda')

                        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                                            timediffs=item_timediffs_tensor, features=item_feature_tensor,
                                                            adj_embeddings=item_adj_embedding, select='item_update', CIs = item_CIs_tensor, alpha=args.alpha_LiveCI)    
                                                    

                        item_embeddings[tbatch_itemids, :] = item_embedding_output
                        user_embeddings[tbatch_userids, :] = user_embedding_output


                        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                        loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
                        loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())



                else: # don't need to update US
                    # LOAD THE CURRENT TBATCH
                    tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                    tbatch_streamerids = torch.LongTensor(lib.current_tbatches_streamer[i]).cuda()
                    item_feature_tensor = Variable(torch.Tensor(lib.current_tbatches_item_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                    item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                    tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
                    

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:] # size (batch_size, dim)
                    streamer_embedding_input = streamer_embeddings[tbatch_streamerids, :]
                    user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, streamer_embedding_input, timediffs=user_timediffs_tensor, features=item_feature_tensor, select='project') # 这里只用user_embedding 传的item_embedding没用
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)



                    # ROOM EMBEDDING INITIALIZATION
                    tbatch_itemids_list = tbatch_itemids.tolist()
                    tbatch_streamerids_list = tbatch_streamerids.tolist() 
                    len_tbatch = len(tbatch_streamerids_list)

                    for b in range(len_tbatch):
                        item_id = tbatch_itemids_list[b]
                        streamer_id = tbatch_streamerids_list[b]
                        if streamer_id in seen_streamer:
                            if item_id not in seen_item: # new room
                                exists_rooms = streamer_rooms[streamer_id] 
                                if len(exists_rooms) == 0:
                                    print("Oooops!")
                                    sys.exit(0)
                                exits_rooms_emb = item_embeddings[exists_rooms, :]
                                mean_tensor = exits_rooms_emb.mean(dim=0, keepdim=True) 
                                streamer_emb_exists = streamer_embeddings[streamer_id, :].unsqueeze(0)
                                combined_mean = (streamer_emb_exists + mean_tensor) / 2
                                item_embeddings[[item_id], :] = combined_mean
                                streamer_rooms[streamer_id].append(item_id) 
                        else: # new streamer
                            streamer_rooms[streamer_id].append(item_id) # 
                        seen_item.add(item_id)
                        seen_streamer.add(streamer_id)


                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_adj_, user_length_mask, user_max_length = adj_pad(lib.current_tbatches_user_adj_item[i])
                    item_adj_, item_length_mask, item_max_length = adj_pad(lib.current_tbatches_item_adj_user[i])

                    user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]
                    item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]
                    if model.method == 'mean':
                        # user_adj_embedding = model.aggregate(item_embeddings, lib.current_tbatches_user_adj_item[i], select='user_update')
                        # item_adj_embedding = model.aggregate(user_embeddings, lib.current_tbatches_item_adj_user[i], select='item_update')
                        user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                                user_max_length, user_embedding_input,
                                                                select='user_update')
                        item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input,
                                                                select='item_update')
                    elif model.method == 'attention':
                        user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input,
                                                                    select='user_update')
                        item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')
                    elif model.method == 'gat':
                        user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                                    user_max_length, user_embedding_input,
                                                                    select='user_update')
                        item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')
                    elif model.method == 'lstm':
                        user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                                user_max_length, user_embedding_input,
                                                                select='user_update')
                        item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input,
                                                                select='item_update')



                    batch_size = len(item_CIs)
                    # user update with CIs
                    user_CIs_tensor = torch.tensor(user_CIs, dtype=torch.float32).view(batch_size, 1).to('cuda')


                    user_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                                        timediffs=user_timediffs_tensor, features=item_feature_tensor,
                                                        adj_embeddings=user_adj_embedding, select='user_update', CIs = user_CIs_tensor, alpha=args.alpha_LiveCI)

                    # item update with CIs
                    item_CIs_tensor = torch.tensor(item_CIs, dtype=torch.float32).view(batch_size, 1).to('cuda')
    

                    item_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                                        timediffs=item_timediffs_tensor, features=item_feature_tensor,
                                                        adj_embeddings=item_adj_embedding, select='item_update', CIs = item_CIs_tensor, alpha=args.alpha_LiveCI)   

                    item_embeddings[tbatch_itemids, :] = item_embedding_output
                    user_embeddings[tbatch_userids, :] = user_embedding_output


                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())


            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            if type(US_loss) != int: 
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                total_US_loss += US_loss.item()
                US_loss.backward()
                optimizer_UR.step()
                optimizer_US.step()
                optimizer_UR.zero_grad()
                optimizer_US.zero_grad()
            else: 
                total_loss += loss.item()
                loss.backward(retain_graph=True)

                optimizer_UR.step()
                optimizer_UR.zero_grad()


            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            US_loss = 0
            item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            streamer_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()
            streamer_embeddings_timeseries.detach_()

            # REINITIALIZE
            reinitialize_tbatches()
            tbatch_to_insert = -1

    seen_streamer.clear()
    seen_item.clear()
    streamer_rooms.clear()

    # END OF ONE EPOCH
    print ("\n\nTotal loss in this epoch = %f, US_loss = %f" % (total_loss, total_US_loss))
    item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
    user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
    streamer_embeddings_dystat = torch.cat([streamer_embeddings, streamer_embedding_static], dim=1)
    # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
    save_model(model, optimizer_UR, optimizer_US, args, ep, 
    user_embeddings_dystat, item_embeddings_dystat, streamer_embeddings_dystat, train_end_idx,
    user_adj, item_adj, user_adj_streamer, streamer_adj,
    user_adj_timestamp, item_adj_timestamp,
    user_embeddings_timeseries, item_embeddings_timeseries, streamer_embeddings_timeseries)

    user_embeddings = initial_user_embedding.repeat(num_users, 1)
    item_embeddings = initial_item_embedding.repeat(num_items, 1)
    streamer_embeddings = initial_streamer_embedding.repeat(num_streamers, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print ("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer_UR, optimizer_US, args, ep, 
    user_embeddings_dystat, item_embeddings_dystat, streamer_embeddings_dystat, train_end_idx,
    user_adj, item_adj, user_adj_streamer, streamer_adj,
    user_adj_timestamp, item_adj_timestamp,
    user_embeddings_timeseries, item_embeddings_timeseries, streamer_embeddings_timeseries)