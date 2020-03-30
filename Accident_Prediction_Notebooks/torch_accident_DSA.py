import cv2
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle ## Kevin: import pickle to interact with some of the saved data
# from accident_physical_fgsm_lighter import evaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# path
train_path = './dataset/features/training/' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
test_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
demo_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

dir_str = ''

hard_drive_path = '.'

default_model_path = './model/demo_model'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
save_path = './model/torch_DSA/'  #+ dir_str +'/'                   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
video_path = './dataset/videos/testing/positive/'

## Added: Pickle path for visualization
# Kevin: this line adds the path to the risk/attention that is pickled from the demo network
pickle_path = './eval_pickle_files/demo_batch.pickle' # change to .npy file??

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

n_img_hidden = 200 # embedding image features 
n_att_hidden = 200 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video 

n_phys2 = 200
n_phys3 = 300
n_hidden =  400+ 100 # hidden layer num of LSTM
n_phys_hidden = 100 # hidden variables for the physical features
n_input_physical = 3 # input of the physical paramters

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'demo')
    parser.add_argument('--model',dest = 'model',default= default_model_path)
    parser.add_argument('--features',dest = 'features', default = '1')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    parser.add_argument('--vis_mode', default = 'test')
    parser.add_argument('--vis_negatives', default = False)
    args = parser.parse_args()

    return args

## Model pieces
class EmbedLayer(torch.nn.Module):
    def __init__(self, n_input, n_img_hidden, n_att_hidden, n_hidden, p_drop): # 4096, 200, 200, 500

        super(EmbedLayer, self).__init__()
        self.linear_glob = torch.nn.Linear(n_input, n_img_hidden)
        self.linear_local = torch.nn.Linear(n_input, n_att_hidden)
 

    def forward(self, x):
        x_glob = x[:,:, 0,:]
        x_loc = x[:,:,1::,:]
        glob_embed = self.linear_glob(x_glob)
        loc_embed = self.linear_local(x_loc)
        return glob_embed, loc_embed
      

class Attention_LSTM(nn.Module):
    # Your code goes here
    def __init__(self, batch_size, hidden_size, attention_hidden_size, p_drop): ## FIGURE OUT WHERE OUTPUT SIZE GOES, INCLUDE DROPOUT
        super(Attention_LSTM, self).__init__()
        self.nb_layers = 1
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.p_drop = p_drop
        self.attention_hidden_size = attention_hidden_size
        self.hidden_embed = nn.Linear(hidden_size, attention_hidden_size)
        self.linear_wt = torch.nn.Linear(attention_hidden_size,1, bias = False)
        self.lstm = nn.LSTM(input_size=hidden_size,num_layers=1, hidden_size=hidden_size, dropout = p_drop)

    def forward(self, local_input, global_input, hidden):
        zero_mask = (local_input != 0)[:,:,0]
#         hidden_embedding = self.hidden_embed(local_input)
        h = hidden[0]
        h_out = self.hidden_embed(h).permute(1,0,2)
        e1 = torch.tanh(self.tile(h_out,1,19) + local_input)
        alphas = torch.softmax(self.linear_wt(e1), dim = 1) ## Make sure that softmax is oriented along the correct direction
        attention_list = torch.mul(alphas, local_input) ## Zero out agents that are not active.
        attention_list = torch.mul(zero_mask.unsqueeze(2).float(),attention_list)
        attention = torch.sum(attention_list, dim = 1)
        ## concatenate global and local input
        fusion = torch.cat((global_input, attention), 1).unsqueeze(0)
        output, hidden = self.lstm(fusion, hidden)
        return output, hidden, alphas

    def init_hidden(self):
        document_rnn_init_h = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.nb_layers, self.batch_size, self.hidden_size).type(torch.FloatTensor)), requires_grad=True)
        document_rnn_init_c = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.nb_layers, self.batch_size, self.hidden_size).type(torch.FloatTensor)), requires_grad=True)
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


## Functions
def soft_max_entropy_loss_logits(labels, pred):
  return torch.sum(- labels * F.log_softmax(pred, -1), -1)

def temp_loss(soft_max_ent_loss, i, n_frames, targets):
  pos_loss = -torch.mul(torch.exp(torch.tensor(-(n_frames-i-1)/20.0)),-soft_max_ent_loss)
  neg_loss = soft_max_ent_loss
  temp_loss = torch.mean(torch.mul(pos_loss, targets[:,1])+torch.mul(neg_loss,targets[:,0]))
  return temp_loss


def run_batch(x_in, target, model_1, model_lstm, model_final_layer, train_bool):
  if train_bool:
    model_1.train()
    model_lstm.train()
    model_final_layer.train()
  else:
    model_1.eval()
    model_lstm.eval()
    model_final_layer.eval()
  
  glob_x, local_x = model_1(x_in)
  hidden = model_lstm.init_hidden()
  
  if torch.cuda.is_available(): 
    hidden = (hidden[0].cuda(), hidden[1].cuda())
    
  loss = 0
  for i in range(100):
    cur_glob, cur_loc = glob_x[:,i,:], local_x[:,i,:,:]
    _, hidden, alphas = model_lstm(cur_loc, cur_glob, hidden)
    pred = model_final_layer(hidden[0])

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
    loss += temp_loss(ent_loss, i, 100, target)
  return loss, all_alphas, soft_pred

def test_all(num,path,model1,model2,model3, name = ''):
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
           if torch.cuda.is_available():
              temp_loss, all_alphas, pred = run_batch(torch.from_numpy(test_data).float().cuda(), torch.from_numpy(test_labels).float().cuda() ,model1, lstm, linear_pred, False)
           else:
              temp_loss, all_alphas, pred = run_batch(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float(), model1, lstm, linear_pred, False)
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

    return evaluation(all_pred, all_labels, name)
# Create random Tensors to hold inputs and outputs
# x = torch.randn(10, 100, 20, 4096)
# y = torch.randn(10, 2)


def evaluation(all_pred,all_labels, name, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    ## Recall: Number of True Positives/(True_Positives + False_Positives)

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
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
    
    _,rep_index = np.unique(Recall,return_index=1) # This eliminates repeat indices I believe
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    # Not sure what's going on in the for loop
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    ## figure out this part
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

    print( "Average Precision= " + "{:.4f}".format(AP) + ", Mean time to accident= " +"{:.4}".format(np.nanmean(new_Time) * 5) )
    sort_precision = new_Precision[np.argsort(new_Recall)]
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print( "Recall@80%, Time to accident = " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 5) )
    print( "Recall@80%, Precision = " +"{:.4}".format(sort_precision[np.argmin(np.abs(sort_recall-0.8))]) )

    ### visualize

    if vis:
    	fig = plt.figure()
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        fig.savefig(os.getcwd() + '/Other_Users/torch_figures/' + name + '_recall_precision.png')
        
        plt.clf()
        fig = plt.figure()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()

        fig.savefig(os.getcwd() + '/Other_Users/torch_figures/' + name + '_recall_time.png')
        np.save(os.getcwd() + '/Other_Users/torch_figures/' + name + '_precision', new_Precision)                  
        np.save(os.getcwd() + '/Other_Users/torch_figures/' + name + '_recall', new_Recall)
        np.save(os.getcwd() + '/Other_Users/torch_figures' + name + '_time', new_Time)
        
def train():
  for epoch in range(n_epochs):
  #      epoch = epoch + 39
       # random chose batch.npz
       epoch_loss = np.zeros((train_num,1),dtype = float)
       n_batchs = np.arange(1,train_num+1)
       np.random.shuffle(n_batchs)
       tStart_epoch = time.time()
       for batch in n_batchs:
           file_name = '%03d' %batch
           batch_data = np.load(train_path+'batch_'+file_name+'.npz', allow_pickle = True)
           batch_xs = batch_data['data']
           batch_ys = batch_data['labels']
           optimizer.zero_grad()
           if torch.cuda.is_available():
              batch_loss, all_alphas, soft_pred = run_batch(torch.from_numpy(batch_xs).float().cuda(), torch.from_numpy(batch_ys).float().cuda(), model1, lstm, linear_pred, True)
           else:
              batch_loss, all_alphas, soft_pred = run_batch(torch.from_numpy(batch_xs).float(), torch.from_numpy(batch_ys).float(),model1, lstm, linear_pred, True)
           batch_loss.backward()
           optimizer.step()
           ## update steps
           epoch_loss[batch-1] = batch_loss.detach().cpu().numpy()/batch_size

          
       # print one epoch
       print( "Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss) )
       tStop_epoch = time.time()
       print( "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s" )
       sys.stdout.flush()
       if (epoch+1) % 15 == 0:
          model_name1, model_name2, model_name3 = 'model1_'+str(epoch), 'model2_'+str(epoch), 'model3_'+str(epoch)
          torch.save(model1.state_dict(),save_path + model_name1 + '.pth')
          torch.save(lstm.state_dict(),save_path + model_name2 + '.pth')
          torch.save(linear_pred.state_dict(),save_path + model_name3 + '.pth')
          print ("Training")
          test_all(train_num,train_path,model1, lstm, linear_pred)
          print ("Testing")
          test_all(test_num,test_path,model1, lstm, linear_pred)


## visualization: input and plot pickle file for reference 
## vary inputs for the visualization function to control which frames get shown (pick specific batch, include training/testing, include positives/negatives, etc.)

def vis(models_tuple, vis_mode, vis_negatives): ## add optional argument to view a specific batch(s)
    # build model
    # x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas, dy_dx_phys, xadv_phys, eps = build_model()
    [model1, lstm, linear_pred] = models_tuple

    # Kevin: Pickle file is loaded here
    infile = open(pickle_path, 'rb')
    baseline_accv = pickle.load(infile)
    infile.close()

    if vis_mode == 'test':
      bat_num = test_num
      feat_path = demo_path
      mode = 'test'
    else:
      bat_num = train_num
      feat_path = train_path
      mode = 'train'
      video_path =  './dataset/videos/training/positive/'

    # load data
    for num_batch in range(1,bat_num):
        file_name = '%03d' %num_batch
        all_data = np.load(feat_path+'batch_'+file_name+'.npz')
        data = all_data['data']
        labels = all_data['labels']
        det = all_data['det']
        ID = all_data['ID']


        (baseline_loss, baseline_pred, baseline_weight) = baseline_accv[mode+'_batch_'+str(num_batch)]
        # run result
        # [all_loss,pred,weight] = sess.run([loss,soft_pred,all_alphas], feed_dict={x: data, y: labels, keep: [0.0], eps: eps})
        if torch.cuda.is_available():
            loss, weight, pred = run_batch(torch.from_numpy(data).float().cuda(), torch.from_numpy(labels).float().cuda(), model1, lstm, linear_pred, True)
        else:
            loss, weight, pred = run_batch(torch.from_numpy(data).float(), torch.from_numpy(labels).float(),model1, lstm, linear_pred, True)

        ## Convert outputs into numpy array
        loss, weight, pred = loss.detach().cpu().numpy(), weight.detach().cpu().numpy(), pred.detach().cpu().numpy()

        

        file_list = sorted(os.listdir(video_path))
        for i in range(len(ID)):
            if labels[i][1] == 1 or vis_negatives: # if vis_negatives, all negatives are automatically analyzed\
                if vis_negatives:
                  if labels[i][1]:
                    a_mode = 'positive'
                    file_name = ID[i].decode('utf-8')
                  else:
                    a_mode = 'negative'
                    
                    str2 = 'negative'
                    if mode == train:
                      neg_vid = range(1,830)
                      cf = int(ID[i].decode('utf-8'))-1
                    else:
                      neg_vid = range(830,1+1130)
                      cf = int(ID[i].decode('utf-8'))-830
                    #  print(cf)
                    cf = np.roll(neg_vid,-2)[cf]
                    file_name = '%06d' %cf # negative videos shifted by 2
                    
                    
                  if mode == train:
                    video_path = './dataset/videos/training/'+a_mode+'/'
                  else:
                    video_path = './dataset/videos/testing/'+a_mode+'/'
                  pass ## add modifiers that change the video path to alternate between positive,negative
                counter = 0 
                fig = plt.figure()
                line0, = plt.plot(baseline_pred[i,0:90], linewidth=1.0, color='g')
                line1, = plt.plot(pred[i,0:90],linewidth=3.0)
                plt.ylim(0, 1)
                line2, = plt.plot(counter,pred[i,counter],'ro')
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                ############
                # file_name = ID[i].decode('utf-8')
                bboxes = det[i] 
                new_weight = weight[:,:,i]*255
                
                print(video_path)
                cap = cv2.VideoCapture(video_path+file_name+'.mp4') 
                ret, frame = cap.read() 
                while(ret):
                    line2.set_data(counter,pred[i,counter])
                    
                    fig.canvas.draw()
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    				# display image with offpencv or any operation you like
                    #cv2.imshow("plot",img)
                	###
                    attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                    now_weight = new_weight[counter,:]
                    new_bboxes = bboxes[counter,:,:]
                    index = np.argsort(now_weight)
                    for num_box in index:
                        if now_weight[num_box]/255.0>0.4:
                            cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(0,255,0),3)
                        else:
                            cv2.rectangle(frame,(new_bboxes[num_box,0],new_bboxes[num_box,1]),(new_bboxes[num_box,2],new_bboxes[num_box,3]),(255,0,0),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(new_bboxes[num_box,0],new_bboxes[num_box,1]), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                        attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]

                    attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                    dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                    top_left_label = str(counter+1) + ': ' + file_name +', ' + 'Positive  ::   Batch  '  + str(num_batch) ## change this for showing negative clips
                    cv2.putText(dst,top_left_label ,(10,30), font, 1,(255,255,255),3)
                    ###
                    a1,a2,a3 = img.shape
                    ratio = dst.shape[0]/a1
                    w1 = int(a2*0.75/ratio)
                    h1 = int(a1*0.75/ratio)
                    re_img = cv2.resize(img,(w1,h1), interpolation = cv2.INTER_AREA)
                    # print(dst[0:h1,w1-1:-1].shape, re_img.shape)
                    overlay = cv2.addWeighted(dst[0:h1,-w1-1:-1],0.3,re_img,0.7,0)
                    dst[0:h1,-w1-1:-1] = overlay
                    ####

                    cv2.imshow('result',dst)
                    c = cv2.waitKey(50)
                    ret, frame = cap.read()
                    if c == ord('q') and c == 27 and ret:
                        break;
                    counter += 1
              
            cv2.destroyAllWindows()



def test(model_path, debug = False):
    # load model
    x, x_phys,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas, dy_dx_phys, xadv_phys, eps, dropStream_bool = build_model()
    # inistal Session
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print( "model restore!!!" )
    if debug == False:
        print( "Training" ) 
        epsv = [0.05]
        test_all(sess,train_num,train_path, phys_path_tst,x, x_phys,keep,y,loss,lstm_variables,soft_pred, eps, dropStream_bool, debug)
        print( "Testing" )
        test_all(sess,test_num,test_path,x, x_phys ,keep,y,loss,lstm_variables,soft_pred, eps, dropStream_bool, debug)
    else:
        var_list = list()
        a1,a2,a3 = test_all(sess,train_num,train_path, phys_path_tst,x, x_phys,keep,y,loss,lstm_variables,soft_pred,eps,dropStream_bool, debug = True)
        b1,b2,b3 = test_all(sess,train_num,train_path, phys_path_tst,x, x_phys,keep,y,loss,lstm_variables,soft_pred,eps, dropStream_bool, debug = True)
        var_list.extend([a1,a2,a3,b1,b2,b3])
        return var_list


if __name__ == '__main__':
    args = parse_args()

    ## Instantiate/Build models ##############################################################################################################
    if torch.cuda.is_available():
      model1 = EmbedLayer(4096, 200, 200, 200, 0.5).cuda()
      lstm = Attention_LSTM(10,400,200, 0.5).cuda()
      linear_pred = Linear_Pred(10,400,2).cuda()
    else:
      model1 = EmbedLayer(4096, 200, 200, 200, 0.5)
      lstm = Attention_LSTM(10,400,200, 0.5)
      linear_pred = Linear_Pred(10,400,2)

    # optimizer only necessary for training
    optimizer = torch.optim.Adam(list(model1.parameters())+list(lstm.parameters())+list(linear_pred.parameters()), lr = 0.0001)

    ## Load existing models ############################################################################################################
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    map_location=torch.device('cpu')
    sorted_dir = sorted(os.listdir(save_path))
    print(sorted_dir)
    model1.load_state_dict(torch.load(save_path + sorted_dir[-3], map_location = map_location))
    lstm.load_state_dict(torch.load(save_path + sorted_dir[-2], map_location = map_location))
    linear_pred.load_state_dict(torch.load(save_path + sorted_dir[-1], map_location = map_location  ))
    model_tuple = (model1, lstm, linear_pred)

    print(str(args.model))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    if args.mode == 'train':
           train()
    elif args.mode == 'test':
           test(args.model)
    elif args.mode == 'demo':
           vis(model_tuple, args.vis_mode, args.vis_negatives)