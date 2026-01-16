# -*- coding: utf-8 -*
'''
This code evaluates the validation and test performance in an epoch of the model trained in DCGLive.py.
The task is: next room prediction

To calculate the performance for one epoch:
$ python evaluate_interaction_prediction.py --network kuailive --model DCGLive --method attention --epoch 10 --embedding_dim 128 --alpha_LiveCI 0.9

To calculate the performance for all epochs, use the bash file, run_eva.sh, which calls this file once for every epoch.

Reference Paper: 
Kumar, S., Zhang, X., & Leskovec, J. (2019, July). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1269-1278).
Li, X., Zhang, M., Wu, S., Liu, Z., Wang, L., & Yu, P. S. (2020, November). Dynamic graph collaborative filtering. In 2020 IEEE international conference on data mining (ICDM) (pp. 322-331). IEEE.
'''

from library_data import *
from library_models import *
import datetime
torch.set_printoptions(precision=16)

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Network/dataset name')
parser.add_argument('--model', default='DCGLive', help="Model name")
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--l2us', type=float, default=1.0, help='regular coefficient of user with streamer')
parser.add_argument('--l2s', type=float, default=1.0, help='regular coefficient of streamer')
parser.add_argument('--l2uu', type=float, default=1.0, help='regular coefficient of user after streamer')
parser.add_argument('--method', default="attention", help='The way of aggregate adj')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=50, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--lr_US', default=5e-4, type=float, help='Learning rate for optimizing US')
parser.add_argument('--lr_UR', default=1e-3, type=float, help='Learning rate for optimizing UR')
parser.add_argument('--alpha_LiveCI', default=0.9, type=float, help='Weight of LiveCI')
parser.add_argument('--CI_max_length', default=30, type=int, help='The max number of neighbors to compute LiveCI')
args = parser.parse_args()
print(args)
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print ("No interaction prediction for %s" % args.network)
    sys.exit(0)
    
# SET GPU
args.gpu, _ = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
dic = 'results/'


if args.l2u == 1.0 and args.l2i == 1.0:
    output_fname = dic + "interaction_prediction_%s.%s.%s.alpha_%.1f.txt" % (args.network, args.method, 'adj', args.alpha_LiveCI)
else:
    output_fname = dic + "interaction_prediction_%s.%s.user%.1f.item%.1f.%s.alpha_%.1f.txt" % (args.network, args.method, args.l2u, args.l2i,'adj',  args.alpha_LiveCI)



if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print ("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()


# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, user_timediffs_sequence_streamer, user_previous_streamerid_sequence,
    item2id, item_sequence_id, item_timediffs_sequence, 
    streamer2id, streamer_sequence_id, streamer_timediffs_sequence, 
    timestamp_sequence, endtime_sequence,
    item_feature_sequence, streamer_feature_sequence, y_true, streamer_interaction_label_sequence] = load_network(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra room for "none-of-these"
num_streamers = len(streamer2id) + 1 # one extra streamer for "none-of-these"
normalized_timestamp_sequence = normalize_sequence(timestamp_sequence)
normalized_end_timestamp_sequence = normalize_sequence(endtime_sequence)

num_features = len(item_feature_sequence[0])
num_streamer_features = len(streamer_feature_sequence[0])

id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}
id2streamer = {v: k for k, v in streamer2id.items()}

true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error.

print(f"*** Network statistics:\n  {num_users} users\n  {num_items} items\n  {num_interactions} interactions\n  {sum(y_true)}/{len(y_true)} true labels ***\n")

# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num

# INITIALIZE MODEL PARAMETERS
model = DCGLive(args, num_features, num_users, num_items, num_streamers, num_streamer_features).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

UR_para = []
US_para = []

for name, param in model.named_parameters():
    if 'streamer' in name:
        US_para.append(param)
    else:
        UR_para.append(param)



# INITIALIZE MODEL
learning_rate_1 = 5e-4
learning_rate_2 = 1e-3
max_length = 30
alpha = 0.8

optimizer_UR = optim.Adam(UR_para, lr=learning_rate_2, weight_decay=1e-5)
optimizer_US = optim.Adam(US_para, lr=learning_rate_1, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer_UR, optimizer_US, \
user_embeddings_dystat, item_embeddings_dystat, streamer_embeddings_dystat, \
user_adj, item_adj, user_adj_streamer, streamer_adj, \
user_adj_timestamp, item_adj_timestamp, \
user_embeddings_timeseries, item_embeddings_timeseries, streamer_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer_UR, optimizer_US, args, args.epoch)


print("Finished Loading!")

if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
# set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

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


# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []


seen_streamers = set()
seen_rooms = set()
streamer_rooms = defaultdict(list)

for idx in range(train_end_idx):
    streamer_ = streamer_sequence_id[idx]
    room_ = item_sequence_id[idx]
    if streamer_ in seen_streamers:
        if room_ not in seen_rooms:
            streamer_rooms[streamer_].append(room_)
    else:
        streamer_rooms[streamer_].append(room_)
    seen_rooms.add(room_)
    seen_streamers.add(streamer_)
    

''' 
Here we use the trained model to make predictions for the validation and testing interactions.
The model does a forward pass from the start of validation till the end of testing.
For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
This is used to calculate the rank of the true item the user actually interacts with.

After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
'''
tbatch_start_time = None
loss = 0
US_loss = 0
# FORWARD PASS
print ("*** Making interaction predictions by forward pass (no t-batching) ***")
print ('start time:', datetime.datetime.now())


for j in range(train_end_idx, test_end_idx):
    flag = 0
    # LOAD INTERACTION J
    userid = user_sequence_id[j]
    itemid = item_sequence_id[j]
    item_feature = item_feature_sequence[j]
    user_timediff = user_timediffs_sequence[j]
    item_timediff = item_timediffs_sequence[j]
    timestamp = timestamp_sequence[j]
    # streamer
    streamerid = streamer_sequence_id[j] 
    streamer_feature = streamer_feature_sequence[j]
    streamer_timediff = streamer_timediffs_sequence[j]
    streamer_interaction_label = streamer_interaction_label_sequence[j]
    user_timediff_streamer = user_timediffs_sequence_streamer[j] 
    streamerid_previous = user_previous_streamerid_sequence[j]
    current_timestamp = normalized_timestamp_sequence[j]
    end_timestamp = normalized_end_timestamp_sequence[j]

    if not tbatch_start_time:
        tbatch_start_time = timestamp
    itemid_previous = user_previous_itemid_sequence[j]

    # LOAD USER AND ITEM EMBEDDING
    user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
    user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]

    # ROOM EMBEDDING INITIALIZATION
    if streamerid in seen_streamers:
        if itemid not in seen_rooms:
            flag = 1
            exists_rooms = streamer_rooms[streamerid]
            if len(exists_rooms) == 0:
                print("Ooops!")
                sys.exit(0)

            exits_rooms_emb = item_embeddings[exists_rooms, :]
            mean_tensor = exits_rooms_emb.mean(dim=0, keepdim=True) 
            streamer_emb_exists = streamer_embeddings[streamerid, :].unsqueeze(0)
            combined_mean = (streamer_emb_exists + mean_tensor) / 2
            item_embeddings[[itemid], :] = combined_mean
            # item_embeddings[[itemid], :] = mean_tensor # ablation w/o streamer
            # item_embeddings[[itemid], :] = streamer_emb_exists # ablation w/o mean


            streamer_rooms[streamerid].append(itemid) 
    else: 
        streamer_rooms[streamerid].append(itemid) 
    
    seen_rooms.add(itemid)
    seen_streamers.add(streamerid)

    item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])] 
    item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
    streamer_embedding_input = streamer_embeddings[torch.cuda.LongTensor([streamerid])]
    streamer_embedding_static_input = streamer_embeddings_static[torch.cuda.LongTensor([streamerid])]

    item_feature_tensor = Variable(torch.Tensor(item_feature).cuda()).unsqueeze(0)
    user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
    item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
    item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

    item_CIs = LiveCIR([itemid], [item_adj[itemid]], [item_adj_timestamp[itemid]], [userid], [user_adj[userid]], [current_timestamp], [end_timestamp], user_adj, max_length)
    user_CIs = LiveCIU([userid], [user_adj[userid]], [user_adj_timestamp[userid]], [itemid], [item_adj[itemid]], [current_timestamp], item_adj, max_length)
    
    # directly update UR
    if streamer_interaction_label == 0:
        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, streamer_embedding_input,
            timediffs=user_timediffs_tensor, features=item_feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1)

        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)


        if len(user_adj[userid]) == 0:
            user_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())

        else:
            user_adj_, user_length_mask, user_max_length = adj_pad([user_adj[userid]])
            user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]

            if model.method == 'attention':
                user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                            user_max_length, user_embedding_input,
                                                            select='user_update')
            elif model.method == 'mean':
                user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                            user_max_length, user_embedding_input, select='user_update')
            elif model.method == 'lstm':
                user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                        user_max_length, user_embedding_input,
                                                        select='user_update')
            elif model.method == 'gat':
                user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                        user_max_length, user_embedding_input,
                                                        select='user_update')
        if len(item_adj[itemid]) == 0:
            item_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())
        else:
            item_adj_, item_length_mask, item_max_length = adj_pad([item_adj[itemid]])
            item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]

            if model.method == 'attention':
                item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input,
                                                                select='item_update')
            elif model.method == 'mean':
                item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')
            elif model.method == 'lstm':
                item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                    item_max_length, item_embedding_input,
                                    select='item_update')
            elif model.method == 'gat':
                item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                        item_max_length, item_embedding_input,
                                                        select='item_update')


        # user update with CIs
        user_CIs_tensor = torch.tensor(user_CIs, dtype=torch.float32).view(1, 1).to('cuda')
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                            timediffs=user_timediffs_tensor, features=item_feature_tensor,
                                            adj_embeddings=user_adj_embedding, select='user_update', CIs=user_CIs_tensor, alpha = args.alpha_LiveCI)

        # item update with CIs
        item_CIs_tensor = torch.tensor(item_CIs, dtype=torch.float32).view(1, 1).to('cuda')
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                            timediffs=item_timediffs_tensor, features=item_feature_tensor,
                                            adj_embeddings=item_adj_embedding, select='item_update', CIs=item_CIs_tensor, alpha = args.alpha_LiveCI)

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0)
        user_embeddings[userid,:] = user_embedding_output.squeeze(0)



        user_adj[userid].add(itemid)  
        item_adj[itemid].add(userid)
        user_adj_timestamp[userid].add((itemid, current_timestamp))
        item_adj_timestamp[itemid].add((userid, current_timestamp, end_timestamp))


        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())

    else: # need to update US
        streamer_feature_tensor = Variable(torch.Tensor(streamer_feature).cuda()).unsqueeze(0)
        user_timediffs_tensor_streamer = Variable(torch.Tensor([user_timediff_streamer]).cuda()).unsqueeze(0)
        streamer_embedding_previous = streamer_embeddings[torch.cuda.LongTensor([streamerid_previous])]  
        streamer_timediffs_tensor = Variable(torch.Tensor([streamer_timediff]).cuda()).unsqueeze(0)

        ''' US '''
        # PROJECT USER EMBEDDING
        user_projected_embedding_item = model.forward(user_embedding_input, item_embedding_previous, streamer_embedding_input,
            timediffs=user_timediffs_tensor, features=item_feature_tensor, select='project')

        user_item_embedding = torch.cat([user_projected_embedding_item, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)

        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1)

        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)


        # PROJECT USER EMBEDDING TO CURRENT TIME FOR STREAMER
        user_projected_embedding_streamer = model.forward(user_embedding_input, streamer_embedding_previous, streamer_embedding_previous,\
            timediffs=user_timediffs_tensor_streamer, features=streamer_feature_tensor, select='project')   

        user_streamer_embedding = torch.cat([user_projected_embedding_streamer, streamer_embedding_previous, streamer_embeddings_static[torch.cuda.LongTensor([streamerid_previous])], user_embedding_static_input], dim=1)
        # PREDICR NEXT STREAMER EMBEDDING
        predicted_streamer_embedding = model.predict_streamer_embedding(user_streamer_embedding)
        US_loss += MSELoss(predicted_streamer_embedding, torch.cat([streamer_embedding_input, streamer_embedding_static_input], dim=1).detach())
        # print(US_loss)

        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
        if len(user_adj[userid]) == 0: 
            user_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())

        else:
            user_adj_, user_length_mask, user_max_length = adj_pad([user_adj[userid]])
            user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :]


            if model.method == 'attention':
                user_adj_embedding = model.aggregate_attention(user_adj_em, torch.LongTensor(user_length_mask),
                                                            user_max_length, user_embedding_input,
                                                            select='user_update')
            elif model.method == 'mean':
                user_adj_embedding = model.aggregate_mean(user_adj_em, torch.LongTensor(user_length_mask),
                                                            user_max_length, user_embedding_input, select='user_update')
            elif model.method == 'lstm':
                user_adj_embedding = model.aggregate_lstm(user_adj_em, torch.LongTensor(user_length_mask),
                                                        user_max_length, user_embedding_input,
                                                        select='user_update')
            elif model.method == 'gat':
                user_adj_embedding = model.aggregate_gat(user_adj_em, torch.LongTensor(user_length_mask),
                                                        user_max_length, user_embedding_input,
                                                        select='user_update')
        # item_adj_user
        if len(item_adj[itemid]) == 0: 
            item_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())
        else:
            item_adj_, item_length_mask, item_max_length = adj_pad([item_adj[itemid]])
            item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :]

            if model.method == 'attention':
                item_adj_embedding = model.aggregate_attention(item_adj_em, torch.LongTensor(item_length_mask),
                                                                item_max_length, item_embedding_input,
                                                                select='item_update')
            elif model.method == 'mean':
                item_adj_embedding = model.aggregate_mean(item_adj_em, torch.LongTensor(item_length_mask),
                                                                    item_max_length, item_embedding_input,
                                                                    select='item_update')
            elif model.method == 'lstm':
                item_adj_embedding = model.aggregate_lstm(item_adj_em, torch.LongTensor(item_length_mask),
                                    item_max_length, item_embedding_input,
                                    select='item_update')
            elif model.method == 'gat':
                item_adj_embedding = model.aggregate_gat(item_adj_em, torch.LongTensor(item_length_mask),
                                                        item_max_length, item_embedding_input,
                                                        select='item_update')
        # streamer_adj_user
        if len(streamer_adj[streamerid]) == 0: 
            streamer_adj_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())

        else:
            streamer_adj_, streamer_length_mask, streamer_max_length = adj_pad([streamer_adj[streamerid]])
            streamer_adj_em = user_embeddings[torch.LongTensor(streamer_adj_).cuda(), : ]


            if model.method == 'attention':
                streamer_adj_embedding = model.aggregate_attention(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input,
                                                                select='streamer_update')

            elif model.method == 'mean':
                streamer_adj_embedding = model.aggregate_mean(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input,
                                                                select='streamer_update')
            elif model.method == 'lstm':
                streamer_adj_embedding = model.aggregate_lstm(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input,
                                                                select='streamer_update')
            elif model.method == 'gat':
                streamer_adj_embedding = model.aggregate_gat(streamer_adj_em, torch.LongTensor(streamer_length_mask),
                                                                streamer_max_length, streamer_embedding_input,
                                                                select='streamer_update')
        # user_adj_streamer
        if len(user_adj_streamer[userid]) == 0: 
            user_adj_streamer_embedding = Variable(torch.zeros(1, model.embedding_dim).cuda())

        else:
            user_adj_streamer_, user_length_mask_streamer_, user_max_length_streamer_ = adj_pad([user_adj_streamer[userid]])
            user_adj_streamer_em = streamer_embeddings[torch.LongTensor(user_adj_streamer_).cuda(), :]


            if model.method == 'attention':
                user_adj_streamer_embedding = model.aggregate_attention(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer_),
                                                            user_max_length_streamer_, user_embedding_input,
                                                            select='user_update_with_streamer')

            elif model.method == 'mean':
                user_adj_streamer_embedding = model.aggregate_mean(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer_),
                                                            user_max_length_streamer_, user_embedding_input,
                                                            select='user_update_with_streamer')
            elif model.method == 'lstm':
                user_adj_streamer_embedding = model.aggregate_lstm(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer_),
                                                            user_max_length_streamer_, user_embedding_input,
                                                            select='user_update_with_streamer')
            elif model.method == 'gat':
                user_adj_streamer_embedding = model.aggregate_gat(user_adj_streamer_em, torch.LongTensor(user_length_mask_streamer_),
                                                            user_max_length_streamer_, user_embedding_input,
                                                            select='user_update_with_streamer')



        # user update with CIs
        user_CIs_tensor = torch.tensor(user_CIs, dtype=torch.float32).view(1, 1).to('cuda')
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                            timediffs=user_timediffs_tensor, features=item_feature_tensor,
                                            adj_embeddings=user_adj_embedding, select='user_update', CIs= user_CIs_tensor, alpha = args.alpha_LiveCI)

        # item update with CIs
        item_CIs_tensor = torch.tensor(item_CIs, dtype=torch.float32).view(1, 1).to('cuda')
        item_adj_embedding_CIs = item_adj_embedding * item_CIs_tensor

        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, streamer_embedding_input,
                                            timediffs=item_timediffs_tensor, features=item_feature_tensor,
                                            adj_embeddings=item_adj_embedding, select='item_update', CIs=item_CIs_tensor, alpha = args.alpha_LiveCI)

        streamer_embedding_output = model.forward(user_embedding_input, streamer_embedding_input, streamer_embedding_input,
                                            timediffs=streamer_timediffs_tensor, features=streamer_feature_tensor,
                                            adj_embeddings=streamer_adj_embedding, select='streamer_update')
        user_embedding_output_after_US = model.forward(user_embedding_output, item_embedding_input, streamer_embedding_input,
                                            timediffs=user_timediffs_tensor_streamer, features=streamer_feature_tensor,
                                            adj_embeddings=user_adj_streamer_embedding, select='user_update_with_streamer')

        loss += args.l2i*MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += args.l2u*MSELoss(user_embedding_output, user_embedding_input.detach())
        US_loss += args.l2s*MSELoss(streamer_embedding_output, streamer_embedding_input.detach())
        US_loss += args.l2uu*MSELoss(user_embedding_output, user_embedding_output_after_US)

        # SAVE EMBEDDINGS
        item_embeddings[itemid, :] = item_embedding_output.squeeze(0)
        user_embeddings[userid, :] = user_embedding_output_after_US.squeeze(0)
        streamer_embeddings[streamerid, :] = streamer_embedding_output.squeeze(0)
        
        user_adj[userid].add(itemid) 
        item_adj[itemid].add(userid)
        user_adj_timestamp[userid].add((itemid, current_timestamp))
        item_adj_timestamp[itemid].add((userid, current_timestamp, end_timestamp))



    # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
    if timestamp - tbatch_start_time > tbatch_timespan:
        tbatch_start_time = timestamp

        if type(US_loss) != int:
            loss.backward(retain_graph = True)
            US_loss.backward()
            optimizer_UR.step()
            optimizer_US.step()
            optimizer_UR.zero_grad()
            optimizer_US.zero_grad()
        else:
            loss.backward()
            optimizer_UR.step()
            optimizer_UR.zero_grad()

        # RESET LOSS FOR NEXT T-BATCH
        loss = 0
        US_loss = 0

        item_embeddings.detach_()
        user_embeddings.detach_()
        streamer_embeddings.detach_()



            
# CALCULATE THE PERFORMANCE METRICS
def calculate_ndcg_at_k(ranks, k=10):
    dcg = 0.0
    for i, rank in enumerate(ranks):
        if rank <= k:
            dcg += 1.0 / np.log2(rank + 1)
    return dcg / len(ranks)


performance_dict = dict()
# Validation metrics
ranks = validation_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10) * 1.0 / len(ranks)
rec5 = sum(np.array(ranks) <= 5) * 1.0 / len(ranks)
ndcg10 = calculate_ndcg_at_k(ranks, 10)
ndcg5 = calculate_ndcg_at_k(ranks, 5)
performance_dict['validation'] = [ndcg10, mrr, rec10, ndcg5, rec5]

# Test metrics
ranks = test_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10) * 1.0 / len(ranks)
rec5 = sum(np.array(ranks) <= 5) * 1.0 / len(ranks)
ndcg10 = calculate_ndcg_at_k(ranks, 10)
ndcg5 = calculate_ndcg_at_k(ranks, 5)
performance_dict['test'] = [ndcg10, mrr, rec10, ndcg5, rec5]

# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['NDCG@10', 'Mean Reciprocal Rank', 'Recall@10', 'NDCG@5', 'Recall@5']
print ('end time:', datetime.datetime.now())
print ('\n\n*** Validation performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
    fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")
    
print ('\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

fw.flush()
fw.close()
