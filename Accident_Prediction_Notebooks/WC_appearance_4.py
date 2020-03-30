import cv2
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
from itertools import permutations
from accident_physical_fgsm_lighter import evaluation #_2 as evaluation
from itertools import permutations
import logging
# from qrnn import QRNN
from torchqrnn import QRNN
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# path
train_path = './dataset/features/training/' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
test_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
demo_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line


dir_str = ''

# hard_drive_path = '/media/sasha/Seagate Backup Plus Drive/datasets'
hard_drive_path = '.'

default_model_path = './model/demo_model'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

#@
# save_path = './model/SASHA/WC_appearance_2/'  #+ dir_str +'/'                   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

video_path = './dataset/videos/testing/positive/'
# batch_number
train_num = 126
test_num = 46


############## Train Parameters #################
# Parameters
learning_rate = 0.0001
n_epochs = 90
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)          #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
n_detection = 20 # number of object of each image (include image features)


## Encoder-Decoder ##########################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EmbedLayer(torch.nn.Module):
    def __init__(self, n_input, n_img_hidden, n_att_hidden, n_hidden, n_hidden_phys, p_drop): # 4096, 200, 200, 500
        super(EmbedLayer, self).__init__()
        self.linear_glob = torch.nn.Linear(n_input, n_img_hidden)
        self.linear_local = torch.nn.Linear(n_input, n_att_hidden)
        self.linear_phys = torch.nn.Linear(26, n_hidden_phys)
        self.layer_norm_phys1 = torch.nn.InstanceNorm1d(n_hidden_phys) ## number of frames, number of combinations
        self.dropout_phys1 = torch.nn.Dropout(0.5)
        
        # self.linear_phys2 = torch.nn.Linear(n_hidden_phys, n_hidden_phys)
        # # self.layer_norm_phys2 = torch.nn.InstanceNorm1d(n_hidden_phys)
        # self.dropout_phys2 = torch.nn.Dropout(0.5)
        
        self.linear_phys3 = torch.nn.Linear(n_hidden_phys, n_hidden_phys)
        self.dropout_phys3 = torch.nn.Dropout(0.5)
        
        self.agent_pool = torch.nn.MaxPool1d(19)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_hidden_phys = n_hidden_phys
        
        ########################################################################
        ## Permutation functions to efficiently evaluate agent pairs ###########
        ########################################################################
        perm_list = list(permutations(range(20), 2)) # 190
        perm_sorted, perm_ind_ls, mask_ind_ls, last_set = list(), list(), list(), set()
        for k in range(len(perm_list)):
            perm_sorted.append(np.sort(perm_list[k]))
        perm_array = np.unique(np.array(perm_sorted), axis = 0)
    
        for k in range(20):
            cur_indices = np.hstack((np.where(perm_array[:,0] == k)[0],np.where(perm_array[:,1] == k)[0]))
            perm_ind_ls.append( cur_indices )
        
        ## works only if the active agents/indices are consecutive
        for k2 in range(20):
            cur_set = set()
            last_set = last_set.union(set(perm_ind_ls[k2]))
            for k3 in range(k2+1,20):
                cur_set = cur_set.union(set(perm_ind_ls[k3]))
            cur_agent_set = last_set - cur_set
            mask_ind_ls.append(list(cur_agent_set))
        self.perm_ind_ls, self.mask_ind_ls =  perm_ind_ls, mask_ind_ls 
        self.perm_array = perm_array 
        ########################################################################
      
    def create_mask(self, glob_feat, mask_ind_ls):
        mask_array = np.zeros((10, 100, 190,1))
        # we must iterate through all agents and frames to create the agent mask
        for batch in range(10):
            for frames in range(100):
                num_agents = np.sum((np.sum(glob_feat[batch, frames, :,:],axis = -1)!= 0), axis = -1)
                if (num_agents == 0) or (num_agents == 1):
                    pass
                else:
                    mask_array[batch, frames, mask_ind_ls[num_agents-1]] = 1
        return mask_array
        

    def forward(self, x, world_coord_feat):
        # world_coord_feat is a numpy_array loaded in. The conversion from 
        # pandas dataframe=>numpy=>torch tensor happens outside the function
        x_glob = x[:,:, 0,:]
        x_loc = x[:,:,1::,:]
        glob_embed = self.linear_glob(x_glob)
        loc_embed = self.linear_local(x_loc)
        
        perm_tensor = torch.zeros(10, 100, 190, 26).to(self.device) # tensor containing agent_physical_pairs
        agent_tensor = torch.zeros(10, 100, 20, self.n_hidden_phys).to(self.device) # agent that will recieve the 
        phys_tensor = torch.from_numpy(world_coord_feat).to(self.device)
        mask_array = self.create_mask(world_coord_feat, self.mask_ind_ls) # import the original numpy array to make a mask
        mask_tensor = torch.from_numpy(mask_array).int().to(self.device)
        perm_array_tensor = torch.from_numpy(self.perm_array).to(self.device)  #!! add to device
        
        for perm_iter in range(2):
        ### {10, 100, 190, 8} <== {10, 100, 20, 4}
          perm_tensor[:,:,:,0:13] = phys_tensor.index_select(2, perm_array_tensor[:,0])
          perm_tensor[:,:,:,13::] = phys_tensor.index_select(2, perm_array_tensor[:,1])
          
        perm_tensor = perm_tensor.reshape(10,100*190,26)
        phys_embed1 = F.relu(self.linear_phys(self.dropout_phys1(perm_tensor))) # perm_tensor =>{10, 100, 190, 8}
        # phys_embed2 = F.relu(self.layer_norm_phys2(self.linear_phys2(self.dropout_phys2(phys_embed1))))
        phys_embed = self.linear_phys3(self.dropout_phys3(phys_embed1))
        phys_embed = phys_embed.reshape(10, 100, 190,-1)
        phys_embed_zeroed = phys_embed#*mask_tensor # mask out non-agents #@@

        #(2): Aggregating outputs 
        for cur_agent in range(20):
          ###################################################################################
          cur_agent_tensors = phys_embed_zeroed[:,:,self.perm_ind_ls[cur_agent] ,:]#.view(10, 100, -1)  #### CONVERT perm_ind_ls to longtensor???
          cur_agent_tensors = cur_agent_tensors.permute(0,1,3,2).reshape(10, 100*self.n_hidden_phys, 19)
          cur_agent_tensors = self.agent_pool(cur_agent_tensors)
          cur_agent_tensors = cur_agent_tensors.squeeze(dim=-1).reshape(10,100,self.n_hidden_phys)
          
          agent_tensor[:, :, cur_agent, :] = cur_agent_tensors

        return glob_embed, loc_embed, agent_tensor 
      

class Attention_LSTM_2(nn.Module):
    # Your code goes here
    def __init__(self, batch_size, hidden_size, attention_hidden_size, p_drop): ## FIGURE OUT WHERE OUTPUT SIZE GOES, INCLUDE DROPOUT
        super(Attention_LSTM_2, self).__init__()
        self.nb_layers = 1
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.p_drop = p_drop
        self.attention_hidden_size = attention_hidden_size
        self.hidden_embed = nn.Linear(hidden_size, attention_hidden_size)
        self.linear_wt = torch.nn.Linear(attention_hidden_size,1, bias = False)
        self.lstm = QRNN(input_size=hidden_size,num_layers=1, hidden_size=hidden_size, dropout = p_drop)
        # self.lstm = nn.LSTM(input_size=hidden_size,num_layers=1, hidden_size=hidden_size, dropout = p_drop)
        self.n_att_hidden = 250#attention_hidden_size+50*2 #@$ Change this!!!
        self.n_hidden = hidden_size # (512+95)
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.att_w = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.n_att_hidden, 1).type(torch.FloatTensor)), requires_grad=True)
        self.att_ua = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.n_att_hidden, self.n_att_hidden).type(torch.FloatTensor)), requires_grad=True)
        self.att_ba = nn.Parameter(nn.init.zeros_(torch.Tensor(self.n_att_hidden).type(torch.FloatTensor)), requires_grad=True)
        self.att_wa = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.n_hidden, self.n_att_hidden).type(torch.FloatTensor)), requires_grad=True)


    def forward(self, local_input, global_input, hidden):
        #(1): Attention for local features
        # ASSUME that local_inputs are corrected/sorted beforehand
        ## combine local inputs with the physical coordinates
        ## combind phys_input with local_input and global_input, based off indices given
        # zero_mask = (local_input != 0)[:,:,0]
        # zero_mask = (torch.sum(local_input != 0)[:,:,0]
        zero_mask = (torch.sum((local_input!=0), dim = -1) != 0)
        h_prev = hidden[0]
        brcst_w = self.tile(self.att_w.unsqueeze(0),0,19)
        image_part = local_input.bmm(self.tile(self.att_ua.unsqueeze(0),0,10)+self.att_ba) # won't fc do this automatically?
        e = torch.tanh(image_part.permute(1,0,2)+h_prev.squeeze().mm(self.att_wa))
        alphas = torch.softmax(e.bmm(brcst_w).squeeze(), dim = 0)
        alphas = alphas.permute(1,0).mul(zero_mask.float())
        attention_list = torch.mul(alphas.unsqueeze(2), local_input) ## Zero out agents that are not active.
        attention = torch.sum(attention_list, dim = 1)
        ## concatenate global and local input
        fusion = torch.cat((global_input, attention), 1).unsqueeze(0)
        output, hidden = self.lstm(fusion, hidden)
        return output, hidden, alphas

    def init_hidden(self):
        document_rnn_init_h = nn.Parameter(nn.init.zeros_(torch.Tensor(self.nb_layers, self.batch_size, self.hidden_size).type(torch.FloatTensor)), requires_grad=False)
        document_rnn_init_c = nn.Parameter(nn.init.zeros_(torch.Tensor(self.nb_layers, self.batch_size, self.hidden_size).type(torch.FloatTensor)), requires_grad=False)
        return (document_rnn_init_h, document_rnn_init_c)
      
    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        if torch.cuda.is_available():
          order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        else:
          order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

class Linear_Pred(nn.Module):
    # Your code goes here
    def __init__(self, batch_size, hidden_size, output_size): 
        super(Linear_Pred, self).__init__()
        self.linear_pred = torch.nn.Linear(hidden_size,output_size)

    def forward(self, my_input):
        output = self.linear_pred(my_input)
        return output
        
        
### Classification Functions: ##################################################
## loss functions
def soft_max_entropy_loss_logits(labels, pred):
  return torch.sum(- labels * F.log_softmax(pred, -1), -1)

def temp_loss(soft_max_ent_loss, i, n_frames, targets, ATTC = 0, F = 20, rho = 5):  # set rho to 4 or 5 if AdaLEA is used
  alpha = torch.exp(-torch.max(torch.tensor(0).float(),torch.tensor((90-i)-F*ATTC-rho).float()) )
#   alpha = torch.exp(torch.tensor(-(n_frames-i-1)/20.0)) ## EL loss
  pos_loss = -torch.mul(alpha,-soft_max_ent_loss) ## Add attention component
  neg_loss = soft_max_ent_loss
  temp_loss = torch.mean(torch.mul(pos_loss, targets[:,1])+torch.mul(neg_loss,targets[:,0]))
  return temp_loss
  
### Training Function: #########################################################


def run_batch(x_in, x_phys, feat_ind, target, model_1, model_lstm, model_final_layer, ATTC, train_bool):
  if train_bool:
    model_1.train()
    model_lstm.train()
    model_final_layer.train()
  else:
    model_1.eval()
    model_lstm.eval()
    model_final_layer.eval()

  glob_x, local_x, reformed_tensor = model_1(x_in, x_phys)
  glob_phys = reformed_tensor[:,:,0,:] ## check that odometry is in the first column. Dataframe order MATTERS
  local_phys = reformed_tensor[:,:,1::,:]
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  phys_mask1 = ((local_phys != 0).sum(dim = -1) != 0).sum(dim = -1)

  glob_x = torch.cat((glob_x, glob_phys), -1)
  #reformed tensor => {10, 100, 20, 95}
  hidden = model_lstm.init_hidden()
  if torch.cuda.is_available(): 
    hidden = (hidden[0].cuda(), hidden[1].cuda())
    
  loss = 0
  
  ## RETRIEVE PHYSICAL FEATURES
  
  for i in range(100):
    new_local_phys = torch.zeros(10, 19, 50).to(device) # initialize the tensor for the current local physical features
    for n_bat in range(10):
        new_local_phys[n_bat,feat_ind[n_bat, i, 0:phys_mask1[n_bat, i]], :] = local_phys[n_bat, i, 0:phys_mask1[n_bat, i], :]
    
    cur_glob, cur_loc = glob_x[:,i,:], local_x[:,i,:,:]
    cur_loc = torch.cat((cur_loc, new_local_phys), -1)
    
    _, hidden, alphas = model_lstm(cur_loc, cur_glob, hidden)
    pred = model_final_layer(hidden[0])
        # accumulate the alphas and prediction values
    alphas_1 = alphas.detach()
    pred_1 = pred.detach()
    
    if i == 0:
      all_alphas = alphas_1.squeeze().unsqueeze(1).permute(1,2,0)
      soft_pred = torch.softmax(pred_1, dim = 1).squeeze()[:,1].unsqueeze(1)

    else:
      temp_alphas = alphas_1.squeeze().unsqueeze(1).permute(1,2,0)
      temp_pred =  torch.softmax(pred_1, dim = 1).squeeze()[:,1].unsqueeze(1)
      all_alphas = torch.cat((all_alphas, temp_alphas),0)
      soft_pred = torch.cat((soft_pred, temp_pred),1)

    ent_loss = soft_max_entropy_loss_logits(target, pred)
    loss += temp_loss(ent_loss, i, 100, target, ATTC = ATTC)
    
  return loss, all_alphas, soft_pred
  
  
def approx_evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = mmber of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident
    ## Recall: Number of True Positives/(True_Positives + False_Positives)

    risk_linspace = np.linspace(0, 1, num = 100)
    # risk_linspace = np.linspace(0, len(all_pred.flatten())-1, num = 1000)
    # risk_vals = all_pred.flatten()[np.round(risk_linspace).astype(int)]
    temp_shape = len(risk_linspace)*total_time
    length = [total_time]*all_pred.shape[0]

    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in risk_linspace:
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th) # time points that the prediction exceeds the Threshold. If the label is zero, np.where should return 
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0) ## Find the length for all positive examples
        if Tp_Fp == 0:
            Precision[cnt] = np.nan # Precision doesn't exisit if there are no False Positives
        else:
            Precision[cnt] = Tp/Tp_Fp # Precision is True Positives/ (True Positive + False Positives)
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels) # Recall is True Positives/ (True Positives + False Negatives)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    ## from here we find the precision and recal

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1) # This eliminates repeat indices 
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    # Not sure what's going on in the for loop
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print( "Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " +"{:.4}".format(np.mean(new_Time) * 5) )
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print( "Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 5) )
    return np.mean(new_Time) * 5

    ### visualize

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()
        
        
def test_all(num,path, path_world, phys_path,model1,model2,model3, ATTC, name, mode, debug = False, log_file = ''):
    total_loss = 0.0
    ## display which features are being used for training
    print('Using features from path: ' + str(path))
    with torch.no_grad():
      for num_batch in range(1,num+1):
           # load test_data
           file_name = '%03d' %num_batch
           test_all_data = np.load(path+'batch_'+file_name+'.npz')
           test_data = test_all_data['data']
           test_labels = test_all_data['labels']
           
           phys_dict = np.load(phys_path+'batch_'+file_name+'.npz')
           
           phys_feat, feat_ind = phys_dict['phys_feat'], phys_dict['feat_ind']
           
        #   df_paths = ret_wrld_coord_func[0](path_world, test_all_data, mode, num, split_type=2)
        #   phys_feat, feat_ind = ret_wrld_coord_func[1](df_paths, path, 'batch_'+file_name+'.npz', num, mode=mode)
        

           if torch.cuda.is_available():
              temp_loss, all_alphas, pred = run_batch(torch.from_numpy(test_data).float().cuda(), phys_feat, feat_ind, torch.from_numpy(test_labels).float().cuda() ,model1, model2, model3, ATTC, False)
           else:
              temp_loss, all_alphas, pred = run_batch(torch.from_numpy(test_data).float(), phys_feat, feat_ind, torch.from_numpy(test_labels).float(), model1, model2, model3, ATTC, False)
#            print(temp_loss)
           total_loss += temp_loss/batch_size
           if torch.cuda.is_available():
                pred = pred.cpu().numpy()
           if num_batch <= 1:
               all_pred = pred[:,0:90]
               all_labels = np.reshape(test_labels[:,1],[batch_size,1])
               all_IDs = test_all_data['ID'] #! Added code here to keep track of all the video IDs
           else:
               all_pred = np.vstack((all_pred,pred[:,0:90]))
               all_labels = np.vstack((all_labels,np.reshape(test_labels[:,1],[batch_size,1])))
               all_IDs = np.vstack((all_IDs, test_all_data['ID']))
    if debug:
      return approx_evaluation(all_pred, all_labels)
    else:
      return evaluation(all_pred, all_labels, name, log_file)


  

