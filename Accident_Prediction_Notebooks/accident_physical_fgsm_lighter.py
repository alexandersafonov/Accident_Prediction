import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
from itertools import permutations
import pickle

#################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
############### Global Parameters ###############
# path
train_path = './dataset/features/training/' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
test_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
demo_path = './dataset/features/testing/'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

# train_path = './dataset/preprocess/training/features/feat_set1/'
# test_path = './dataset/preprocess/testing/features/feat_set1/'
# demo_path = './dataset/preprocess/testing/features/feat_set1/'

#dir_str = '1_VGG_processed'
# dir_str = '2_reg_boundbox_append'
#dir_str = '3_reg_dist_boundbox'
dir_str = ''

# hard_drive_path = '/media/sasha/Seagate Backup Plus Drive/datasets'
hard_drive_path = '.'
phys_path_trn = hard_drive_path + '/preprocessed_features/4_backprop_xy/training/'
phys_path_tst = hard_drive_path + '/preprocessed_features/4_backprop_xy/testing/'


default_model_path = './model/demo_model'   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

save_path = './model/PHYSICAL_FGMS/'  #+ dir_str +'/'                   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line

video_path = './dataset/videos/testing/positive/'
# batch_number
train_num = 126
test_num = 46

############## Train Parameters #################
# Parameters
learning_rate = 0.0001
n_epochs = 120
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)          #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this line
n_detection = 20 # number of object of each image (include image features)

n_img_hidden = 200 # embedding image features 
n_att_hidden = 200 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video 

n_phys2 = 150
# n_phys3 = 300
n_hidden =  400+ 150 # hidden layer num of LSTM
n_phys_hidden = 150 # hidden variables for the physical features
n_input_physical = 3 # input of the physical paramters

##################################################

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'demo')
    parser.add_argument('--model',dest = 'model',default= default_model_path)
    parser.add_argument('--features',dest = 'features', default = '1')
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args


def build_model():

    # tf Graph input
    x = tf.placeholder("float", [10, n_frames ,n_detection, n_input])
    x_phys = tf.placeholder("float", [None, n_frames ,n_detection-1, n_input_physical]) ######! :: Added code
    y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float",[None])
    eps = tf.placeholder("float",[None])
    dropStream_bool = tf.placeholder(tf.int32, shape=[], name="condition")  ####!!!
    ATTC = tf.placeholder(tf.float32, (1))
    # epoch_num.reshape(1)
    
    # Define weights
    weights = {
        'em_obj': tf.Variable(tf.random_normal([n_input,n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_input,n_img_hidden], mean=0.0, stddev=0.01)),
        'att_w': tf.Variable(tf.random_normal([n_att_hidden, 1], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([n_att_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01)),

        # 'em_phys': tf.Variable(tf.random_normal([n_input_physical,n_phys_hidden], mean=0.0, stddev=0.01)),
        # 'phys2': tf.Variable(tf.random_normal([342,n_phys2], mean=0.0, stddev=0.01)),
        # 'phys3': tf.Variable(tf.random_normal([n_phys3,n_frames], mean=0.0, stddev=0.01))
        'embed_phys': tf.Variable(tf.random_normal([228,n_phys_hidden], mean=0.0, stddev=0.01))
    }
    biases = {
        'em_obj': tf.Variable(tf.random_normal([n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_img_hidden], mean=0.0, stddev=0.01)),
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01)),
        
        'embed_phys': tf.Variable(tf.random_normal([n_phys_hidden], mean=0.0, stddev=0.01))
        # 'em_phys': tf.Variable(tf.random_normal([n_phys_hidden], mean=0.0, stddev=0.01)),
        # 'phys2': tf.Variable(tf.random_normal([n_phys2], mean=0.0, stddev=0.01)),
        # 'phys3': tf.Variable(tf.random_normal([n_frames], mean=0.0, stddev=0.01))
    }

    perm_list = list(permutations(range(19), 2))[0:171]
    perm_array=np.array([np.array(xi) for xi in perm_list])
    # Define a lstm cell with tensorflow
    #####! https://www.tensorflow.org/beta/guide/keras/rnn
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden,initializer= tf.random_normal_initializer(mean=0.0,stddev=0.01),use_peepholes = True,state_is_tuple = False)
    #####! Update this for Tensorflow 2.0

    # using dropout in output of LSTM
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1 - keep[0])
    # init LSTM parameters
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    # init loss 
    loss = 0.0  
    # Mask 
    zeros_object = tf.to_float(tf.not_equal(tf.reduce_sum(tf.transpose(x[:,:,1:n_detection,:],[1,2,0,3]),3),0)) # [100, 19, batch_size]
    # Start creating graph
    for i in range(n_frames):
      with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        # input features (Faster-RCNN fc7)
        X = tf.transpose(x[:,i,:,:], [1, 0, 2])  # permute n_steps and batch_size (n x b x h) :: [20, ?, 4096]
        
        X = tf.cond(dropStream_bool > 0, lambda: tf.math.scalar_mul(1,X), lambda: tf.math.scalar_mul(0,X))
        inputs = x_phys[:,i,:,0:2]
        tfx1 = tf.gather(inputs[:,:,0],perm_array[:,0], axis = 1)
        tfx2 = tf.gather(inputs[:,:,0],perm_array[:,1], axis = 1)
        tfy1 = tf.gather(inputs[:,:,1],perm_array[:,0], axis = 1)
        tfy2 = tf.gather(inputs[:,:,1],perm_array[:,1], axis = 1)
        dis_feat = tf.math.sqrt(tf.math.square(tfx1-tfx2)+tf.math.square(tfy1-tfy2)) 
        # h_phys2 = tf.matmul(dis_feat, weights['phys2']) + biases['phys2']
        
        X_phys = tf.reshape(x_phys[:,i,:,:],[-1, 19*3])
        phys_fusion = tf.concat([dis_feat,X_phys],1)
        out_phys = tf.matmul(phys_fusion,weights['embed_phys']) + biases['embed_phys']
        # frame embedded
        image = tf.matmul(X[0,:,:],weights['em_img']) + biases['em_img'] # 1 x b x h :: Embedding Global Features
        # object embedded
        n_object = tf.reshape(X[1:n_detection,:,:], [-1, n_input]) # Change to two dimensions, [19*batch_size, 4096]
        n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj'] # (n x b) x h :: mat_multiplication ==> [19*batch_size, 256]
        n_object = tf.reshape(n_object,[n_detection-1,batch_size,n_att_hidden]) # n-1 x b x h :: reshape back to tensor, [19, batch_size, 256]
        n_object = tf.multiply(n_object,tf.expand_dims(zeros_object[i],2)) # [19, batch_size, 256] .* [19, batch_size, 1]

        ######! Added the lines below
        # before being recombined, image c [?,256], n_object c [19, ?, 256]
        # n_phys = tf.reshape(X_phys, [-1, n_input_physical]) # Change to two dimensions, [19*batch_size, 4096]
        # n_phys = tf.matmul(n_phys, weights['em_phys']) + biases['em_phys'] # (n x b) x h :: mat_multiplication ==> [19*batch_size, 256]
        # n_phys = tf.reshape(n_phys,[n_detection-1,batch_size,n_phys_hidden ]) # n-1 x b x h :: reshape back to tensor, [19, batch_size, 256]
        # n_phys = tf.multiply(n_phys,tf.expand_dims(zeros_object[i],2)) # [19, batch_size, 256] .* [19, batch_size, 1]

        # object attention
        brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [n_detection-1,1,1]) # n x h x 1 :: R c [19, 256, 1]
        image_part = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['att_ua'], 0), [n_detection-1,1,1])) + biases['att_ba'] # n x b x h :: R c [19, 10, 256]
        e = tf.tanh(tf.matmul(h_prev,weights['att_wa'])+image_part) # n x b x h :: R c [19, 10, 256]
        # the probability of each object
        alphas = tf.multiply(tf.nn.softmax(tf.reduce_sum(tf.matmul(e,brcst_w),2),0),zeros_object[i]) # multiply by mask to get rid of rows that aren't objects :: [19, 10]
        # weighting sum
        attention_list = tf.multiply(tf.expand_dims(alphas,2),n_object) #  :: [19, 10, 256]
        attention = tf.reduce_sum(attention_list,0) # b x h :: [10, 256]

        ##! 
        # attn_phys_list = tf.multiply(tf.expand_dims(alphas,2),n_phys)
        # att_phys = tf.reduce_sum(n_phys,0) 
        # embedd_2 = tf.concat([att_phys, h_phys2],axis = 1)
        # out_phys = tf.matmul(embedd_2, weights['phys3']) + biases['phys3']
        # concat frame & object
        fusion = tf.concat([image,attention, out_phys],1) # [10,512] ==> 

        with tf.variable_scope("LSTM") as vs:
            outputs,istate = lstm_cell_dropout(fusion,istate)
            lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

        # save prev hidden state of LSTM
        h_prev = outputs
        # FC to output
        pred = tf.matmul(outputs,weights['out']) + biases['out'] # b x n_classes
        # save the predict of each time step
        if i == 0:
            soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            all_alphas = tf.expand_dims(alphas,0)
        else:
            temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred),(1,0)),1),(batch_size,1))
            soft_pred = tf.concat([soft_pred,temp_soft_pred],1)
            temp_alphas = tf.expand_dims(alphas,0)
            all_alphas = tf.concat([all_alphas, temp_alphas],0)

        # positive example (exp_loss)
        d = 90 - i
        F = 20
        rho = 5
        alpha = tf.exp(-tf.maximum(float(0), d-F*ATTC-rho))
        pos_loss = -tf.multiply(alpha,-tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
        # negative example
        neg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = pred) # Softmax loss

        temp_loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss,y[:,1]),tf.multiply(neg_loss,y[:,0])))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        loss = tf.add(loss, temp_loss)
        
    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss/n_frames) # Adam Optimizer

    #fgsm code
    clip_min = 0
    clip_max = 1
    
    
    # xadv = x
    # dy_dx, = tf.gradients(loss,xadv)
    # xadv = tf.stop_gradient(xadv + eps*tf.sign(dy_dx))
    # xadv = tf.clip_by_value(xadv, clip_min, clip_max)

    #xadv_phys = x
    dy_dx_phys, = tf.gradients(loss,x)
    xadv_phys = tf.stop_gradient(x + eps*tf.sign(dy_dx_phys))
    xadv_phys = tf.clip_by_value(x, clip_min, clip_max)

    return x, x_phys, keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas, dy_dx_phys, xadv_phys, eps, dropStream_bool, ATTC

def train():
    # build model
    x, x_phys,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas, dy_dx_phys, xadv_phys, eps, dropStream_bool, ATTC = build_model()  ## is this the right way to 
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    # mkdir folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=100)
     
    model_path = './model/PHYSICAL_FGMS/model-20'
    saver.restore(sess, model_path)
    print( "model restore!!!" )
    approx_ATTC = float(2.2)
    last_ATTC = float(2.2)
    # Keep training until reach max iterations
    # start training
    for epoch in range(n_epochs):
         # random chose batch.npz
         epoch = epoch + 20
         epoch_loss = np.zeros((train_num,1),dtype = float)
         n_batchs = np.arange(1,train_num+1)
         np.random.shuffle(n_batchs)
         tStart_epoch = time.time()

         if epoch < 10:
            approx_ATTC = float(0)
            last_ATTC = float(0)

         name = 'physical_ADALea_'+str(epoch)

         for batch in n_batchs:
             file_name = '%03d' %batch
             batch_data = np.load(train_path+'batch_'+file_name+'.npz', allow_pickle = True)
             batch_data_phys = np.load(phys_path_trn+'batch_'+file_name+'.npz', allow_pickle = True)
             batch_xs = batch_data['data']
             batch_ys = batch_data['labels']
             batch_phys = batch_data_phys['data']*80

             ## call batch_phys and feed it in
             if epoch % 2 == 0:
                 _,batch_loss = sess.run([optimizer,loss], feed_dict={x: batch_xs, x_phys: batch_phys, y: batch_ys, keep: [0.5], eps: [0.01], dropStream_bool: 1, ATTC: [approx_ATTC]})
             else:
                 _,batch_loss = sess.run([optimizer,loss], feed_dict={x: batch_xs, x_phys: batch_phys, y: batch_ys, keep: [0.5], eps: [0.01], dropStream_bool: 0, ATTC: [approx_ATTC]})  
             # else:
             #     _,batch_loss = sess.run([optimizer,loss], feed_dict={x: batch_xs, x_phys: batch_phys, y: batch_ys, keep: [0.5], eps: [0.01], dropStream_bool: 1}) 
             epoch_loss[batch-1] = batch_loss/batch_size
         last_ATTC = approx_ATTC
         approx_ATTC = test_all(sess,train_num,train_path, phys_path_trn,x, x_phys,keep,y,loss,lstm_variables,soft_pred, eps,dropStream_bool, name, ATTC, approx_ATTC, debug = True)
         if epoch % 2 != 0:
            approx_ATTC = last_ATTC ## overwrite when appearance stream gets dropped for sake of consistency

         # print one epoch
         print( "Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss) )
         tStop_epoch = time.time()
         print( "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s" )
         sys.stdout.flush()
         if (epoch+1) % 5 == 0:
            saver.save(sess,save_path+"model", global_step = epoch+1)
            eps_val = [0.01]
            print ("Training")
            test_all(sess,train_num,train_path, phys_path_trn,x, x_phys,keep,y,loss,lstm_variables,soft_pred, eps,dropStream_bool, name, ATTC, approx_ATTC)
            print ("Testing")
            test_all(sess,test_num,test_path, phys_path_tst,x, x_phys,keep,y,loss,lstm_variables,soft_pred, eps,dropStream_bool, name, ATTC, approx_ATTC)
    print( "Optimization Finished!" )
    saver.save(sess, save_path+"final_model")

def test_all(sess,num,path, phys_path_tst,x, x_phys,keep,y,loss,lstm_variables,soft_pred, eps, dropStream_bool, name, ATTC, approx_ATTC, debug = False):
    total_loss = 0.0
    ## display which features are being used for training
    print('Using features from path: ' + str(path))
    for num_batch in range(1,num+1):
         # load test_data
         file_name = '%03d' %num_batch
         test_all_data = np.load(path+'batch_'+file_name+'.npz')
         test_data_phys = np.load(phys_path_tst+'batch_'+file_name+'.npz', allow_pickle = True)['data']*80
         test_data = test_all_data['data']
         test_labels = test_all_data['labels']
     
         [temp_loss,pred] = sess.run([loss,soft_pred], feed_dict={x: test_data, x_phys: test_data_phys, y: test_labels, keep: [0.0], eps: [0.01], dropStream_bool: 1, ATTC: [approx_ATTC]})
         
         total_loss += temp_loss/batch_size

         if num_batch <= 1:
             all_pred = pred[:,0:90]
             all_labels = np.reshape(test_labels[:,1],[batch_size,1])
             all_IDs = test_all_data['ID'] #! Added code here to keep track of all the video IDs
         else:
             all_pred = np.vstack((all_pred,pred[:,0:90]))
             all_labels = np.vstack((all_labels,np.reshape(test_labels[:,1],[batch_size,1])))
             all_IDs = np.vstack((all_IDs, test_all_data['ID']))
    if debug:
        approx_evaluation(all_pred,all_labels)
    else:
        evaluation(all_pred,all_labels,name)



    
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
        for i in range(len(all_pred)):  ## iterates through all prediction values
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

    return np.nanmean(new_Time) * 5

    ### visualize

    # if vis:

    #     savepath = os.getcwd() + '/model_figures/'
        
    #     if not os.path.exists(savepath):
    #         os.makedirs(savepath)

    #     recall_precision = plt.figure()
    #     fig = plt.figure()
    #     plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.ylim([0.0, 1.05])
    #     plt.xlim([0.0, 1.0])
    #     plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
    #     # plt.show()
    #     fig.savefig(savepath + name + '_recall_precision.png')
 
    #     plt.clf()
    #     fig = plt.figure()
    #     plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
    #     plt.xlabel('Recall')
    #     plt.ylabel('time')
    #     plt.ylim([0.0, 5])
    #     plt.xlim([0.0, 1.0])
    #     plt.title('Recall-mean_time' )
    #     # plt.show()
    #     fig.savefig(savepath + name + '_recall_time.png')

    #     plt.close('all')

    #     data = {}
    #     data['precision'] = new_Precision
    #     data['recall'] = new_Recall
    #     data['time'] = new_Time
    #     data['prediction'] = all_pred
        
    #     filename = savepath + name + '_dict'
    #     outfile = open(filename,'wb')
    #     pickle.dump(data,outfile)
    #     outfile.close()

def approx_evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
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

def vis(model_path):
    # build model
    x,keep,y,optimizer,loss,lstm_variables,soft_pred,all_alphas, dy_dx_phys, xadv_phys, eps = build_model()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # restore model
    saver.restore(sess, model_path)
    # load data
    for num_batch in range(1,test_num):
        file_name = '%03d' %num_batch
        all_data = np.load(demo_path+'batch_'+file_name+'.npz')
        data = all_data['data']
        labels = all_data['labels']
        det = all_data['det']
        ID = all_data['ID']
        # run result
        [all_loss,pred,weight] = sess.run([loss,soft_pred,all_alphas], feed_dict={x: data, y: labels, keep: [0.0], eps: eps})
        file_list = sorted(os.listdir(video_path))
        for i in range(len(ID)):
            if labels[i][1] == 1 :
                counter = 0 
                fig = plt.figure()
                line1, = plt.plot(pred[i,0:90],linewidth=3.0)
                plt.ylim(0, 1)
                line2, = plt.plot(counter,pred[i,counter],'ro')
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                ############
                file_name = ID[i].decode('utf-8')
                bboxes = det[i] 
                new_weight = weight[:,:,i]*255
                
                print(video_path)
                print("Batch Number: " + str(num_batch))
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
                    top_left_label = str(counter+1) + ': ' + file_name +', ' + 'Positive' ## change this for showing negative clips
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
    # if args.features == '1':
    #     train_path = './dataset/features/training/'
    #     test_path = './dataset/features/testing/'
    #     demo_path = './dataset/features/testing/'
    #     print('Using original path to features')

    # if args.features == '2':
    #     train_path = './dataset/preprocess/features/feat1_VGG_processed/training/'
    #     test_path = './dataset/preprocess/features/feat1_VGG_processed/testing/'
    #     demo_path = './dataset/preprocess/features/feat1_VGG_processed/testing/'
    #     print('Using VGG Preprocessed Features')

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
           vis(args.model)
