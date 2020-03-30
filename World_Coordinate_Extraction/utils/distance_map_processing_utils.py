
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import os
import image_utils
import cv2

import re
import time
import glob
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

import tensorflow as tf
from depth_from_video_in_the_wild import model
from scipy import stats

## change size to (720, 1280), if it is not 
# function that returns preferential crop


def conv_bb_based_off_existing_crop(det, crop_det, frame_paths):
  '''
  :::inputs:::
  input shape: input shape of the original image before resizing
  # input_shape is (720,1280)
  det shape [10,100, 19, 6]: We want to transform to
  :::output:::
  reshape_det, which has shape[10,100,19, 7], where the first four dimensions are the reshaped coordinates, the SECOND-TO-last dimension is the crop that is used
  (with 0 --> middle crop, 1--> top crop, 2--> bottom crop. This pointer is found by seeing where the center coordinate of the bounding box is located)
  '''

  top_crop = 180.  ## hard coded borders
  bottom_crop = 540.
  mid_crop = bottom_crop-top_crop
  det_reshape = np.zeros((10,100,19,7))

  for batch in range(10):
    # input_shape = input_shape_list[batch] 
    # input_shape = np.zeros((2,1))
    # input_shape[0] = 720
    # input_shape[1] = 1280
    
    global_pointer = np.max(np.max(crop_det[batch,:,:,-2]))
    
    sample_img = glob.glob(frame_paths[batch]+'/*.jpg')[0]
    # sample_img_arr = np.clip(np.asarray(Image.open( sample_img ), dtype=float) / 255, 0, 1)
    sample_img_arr = np.array(Image.open( sample_img))
    input_shape = sample_img_arr.shape
    if input_shape[0] != 720 or input_shape[1] != 1280:
        print(input_shape, frame_paths[batch])
        
    # print(input_shape)
    horizontal_shift, vertical_shift = 0, 0
    if input_shape[0] != 720:
        top_border = np.round((720-img.shape[0])/2)
        vertical_shift = int(top_border)
        input_shape[0] = 720
        
    if input_shape[1] < 1000:
        left_border = np.round((1280-img.shape[1])/2)
        horizontal_shift = int(left_border)
        input_shape[1] = 1280
    
    
    for frame in range(100):
        
      indices = np.nonzero(det[batch,frame,:,0])[0]
#       det_altered = det[batch,frame,indices,:]
      if len(indices)>0:

          crop_pointer = global_pointer
          # if input_shape[1] < 1000:
          #   crop_pointer = 1

          for agent in range(19):

            det_slice = det[batch, frame, agent,:].copy()
            if np.sum(det_slice != 0):

              # even if the input image size is different, for the horizontal axis we pad with zeros if it is not 1280, keeping the relative horizontal scaling factor the same
              final_height = 128
              final_width = 416

              det_reshape[batch,frame, agent, 4] = det[batch,frame, agent, 4].copy() # copy over class
              det_reshape[batch,frame, agent, -1] = det[batch,frame, agent, -1].copy() # copy over agent_id
              
              ## move det by scale
              det[batch,frame, agent, 0] += vertical_shift 
              det[batch,frame, agent, 2] += vertical_shift 
              det[batch,frame, agent, 1] += horizontal_shift
              det[batch,frame, agent, 3] += horizontal_shift 
              

              middle_perc = 0.50
              left = 1-middle_perc
              half = left/2


              vertical_scaling = final_height/(input_shape[0]/2)
              wdt = int((128*input_shape[1]/(input_shape[0]/2)))
              remain = wdt - 416
            #   det_h1, det_h2 = np.round(((det_slice[0]+1)*horizontal_scaling)-1), np.round(((det_slice[2]+1)*horizontal_scaling)-1)
            #   det_v1, det_v2 = np.round(((det_slice[1]+1)*vertical_scaling)-1), np.round(((det_slice[3]+1)*vertical_scaling)-1)
              h1 = np.round(((det_slice[0].astype(float))*vertical_scaling)) - (remain/2)#!!
              h2 = np.round(((det_slice[2].astype(float))*vertical_scaling)) - (remain/2)#!!
              if h1 < 0:
                  h1 = 0
              if h2 < 0:
                  h2 = 0
              if h1 > 416:
                  h1 = 416
              if h2 > 416:
                  h2 = 416
              det_reshape[batch,frame, agent, 0] = h1
              det_reshape[batch,frame, agent, 2] = h2

              ## Case 1: Fits into top region
              if crop_pointer == 1:
                det_reshape[batch,frame,agent,-2] = 1

                if det_slice[3]>mid_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                else:
                  det_reshape[batch,frame, agent, 3] = np.round(((det_slice[3].astype(float))*vertical_scaling)) #!!
                if det_slice[1]>mid_crop:
                  det_reshape[batch,frame, agent, 1] = 127
                else:
                  det_reshape[batch,frame, agent, 1] = np.round(((det_slice[1].astype(float))*vertical_scaling)) #!!


              ## Case 2: Fits into bottom region
              elif crop_pointer == 3:
                det_reshape[batch,frame,agent,-2] = 3

                if det_slice[1] < mid_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[1] = det_slice[1] - mid_crop
                  det_reshape[batch,frame, agent, 1] = np.round(((det_slice[1].astype(float))*vertical_scaling)) #!!
                    
                if det_slice[3] < mid_crop:
                  det_reshape[batch,frame, agent, 3] = 0
                else:
                  det_slice[3] = det_slice[3] - mid_crop
                  det_reshape[batch,frame, agent, 3] = np.round(((det_slice[3].astype(float))*vertical_scaling)) #!!

              ## Case 3: Fits into middle region
              else:
                det_reshape[batch,frame,agent,-2] = 2

                if det_slice[1]<top_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                  if det_slice[3]<top_crop:
                      det_reshape[batch,frame, agent, 3] = 0
                else:
                  det_slice[1] = det_slice[1] - top_crop
                  det_reshape[batch,frame, agent, 1] = np.round(((det_slice[1].astype(float))*vertical_scaling)) #!!
                  
                if det_slice[3]> bottom_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                  if det_slice[1]> bottom_crop:
                      det_reshape[batch,frame, agent, 1] =127
                  
                else:
                  det_slice[3] = det_slice[3] - top_crop
                  det_reshape[batch,frame, agent, 3] = np.round(((det_slice[3].astype(float))*vertical_scaling)) #!!
                    
  return det_reshape
# function: modified version of conversion function


def bounding_box_conversion(det, input_shape_list):
  '''
  :::inputs:::
  input shape: input shape of the original image before resizing
  det shape [10,100, 19, 6]: We want to transform to
  :::output:::
  reshape_det, which has shape[10,100,19, 7], where the first four dimensions are the reshaped coordinates, the SECOND-TO-last dimension is the crop that is used
  (with 0 --> middle crop, 1--> top crop, 2--> bottom crop. This pointer is found by seeing where the center coordinate of the bounding box is located)
  '''

  top_crop = 180.  ## hard coded borders
  bottom_crop = 540.
  mid_crop = bottom_crop-top_crop
  det_reshape = np.zeros((10,100,19,7))

  for batch in range(10):
    input_shape = input_shape_list[batch] 
    for frame in range(100):
      indices = np.nonzero(det[batch,frame,:,0])[0]
#       det_altered = det[batch,frame,indices,:]
      if len(indices)>0:
          midpoints = (det[batch,frame,indices,1] + det[batch,frame,indices,3])/2
          difference_arr = np.hstack((np.absolute((bottom_crop -top_crop)-midpoints), \
                            np.absolute(top_crop-midpoints),np.absolute(bottom_crop-midpoints)))
          crop_pointers = np.argmin(difference_arr, axis = 0)

          crop_pointer = stats.mode(crop_pointers)
         
          if input_shape[1] < 1000:
            crop_pointer = 1

          for agent in range(19):

            det_slice = det[batch, frame, agent,:].copy()
            if np.sum(det_slice != 0):

              # even if the input image size is different, for the horizontal axis we pad with zeros if it is not 1280, keeping the relative horizontal scaling factor the same
              final_height = 128
              final_width = 416

              det_reshape[batch,frame, agent, 4] = det[batch,frame, agent, 4].copy() # copy over class
              det_reshape[batch,frame, agent, -1] = det[batch,frame, agent, -1].copy() # copy over agent_id

              if input_shape[1] < 1000:
                input_shape[0] = 720
                input_shape[1] = 1280

              middle_perc = 0.50
              left = 1-middle_perc
              half = left/2

              vertical_scaling = final_height/(input_shape[0]/2)
              horizontal_scaling = final_width/input_shape[1]

              ## Case 1: Fits into top region
              if crop_pointer == 1:
                det_reshape[batch,frame,agent,-2] = 1

                # no need to shift for top crop
                det_reshape[batch,frame, agent, 0] = np.round((det_slice[0])*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)

                if det_slice[3]>top_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                else:
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)
                if det_slice[1]>top_crop:
                  det_reshape[batch,frame, agent, 1] = 127
                else:
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)


              ## Case 2: Fits into bottom region
              elif crop_pointer == 2:
                det_reshape[batch,frame,agent,-2] = 3
                det_reshape[batch,frame, agent, 0] = np.round(det_slice[0]*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)

                if det_slice[1] < mid_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[1] = det_slice[1] - mid_crop
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)
                    
                if det_slice[3] < mid_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[3] = det_slice[3] - mid_crop
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)

              ## Case 3: Fits into middle region
              else:
                det_reshape[batch,frame,agent,-2] = 2
                ## vertical crop off-sets coordinates
                #  det_slice[0] = det_slice[0] - top_crop
                #  det_slice[2] = det_slice[2] - top_crop
                det_reshape[batch,frame, agent, 0] = np.round(det_slice[0]*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)

                if det_slice[1]<top_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[1] = det_slice[1] - top_crop
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)
                if det_slice[3]> bottom_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                else:
                  det_slice[3] = det_slice[3] - top_crop
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)
                
                if (det_slice[1]> bottom_crop and det_slice[3]> bottom_crop) or (det_slice[1]< top_crop and det_slice[3]< top_crop):
                    det_reshape[batch,frame, agent, 1] = 0
                    det_reshape[batch,frame, agent, 3] = 0
                    
  return det_reshape

def retrieve_distance_dir(labels,ID, str_mode, dist_dir, n_batches):
    dist_imgs = list()

    #ignore_boxes_flags = np.zeros([len(labels),])

    # list of strings, gives names of videos
    for ak in range(len(labels)):
        if labels[ak,1] == 1:
            str2 = 'positive'
            file_name = ID[ak].decode('utf-8')
        else:
            str2 = 'negative'
            if str_mode == 'training':
              neg_vid = range(1,1+len(os.listdir(dist_dir +str_mode+'/'+str2+'/')))
              cf = int(ID[ak].decode('utf-8'))-1
            else:
              neg_vid = range(830,1+1130)
              cf = int(ID[ak].decode('utf-8'))-830
            cf = np.roll(neg_vid,-2)[cf]
            file_name = '%06d' %cf # negative videos shifted by 2      

        #hardcode so that the correct video file name is outputted
        #it will correspond with the previously called bounding boxes
        corr = file_name
        if labels[ak,1] == 1: #positive
            if (str_mode == 'test'):
                if (file_name == '000556'):
                    corr = '000506'
                elif (file_name == '000606'):
                    corr = '000556'     
                #elif (file_name == '000506'):
                #    ignore_boxes_flags[ak] = 1           
            else: #training
                if (file_name == '000101'):
                    corr = '000051'
                elif (file_name == '000251'):
                    corr = '000201'
                elif (file_name == '000301'):
                    corr = '000251'
                elif (file_name == '000351'):
                    corr = '000301'
                elif (file_name == '000401'):
                    corr = '000351'
                #elif ((file_name == '000051') or file_name == '000201' ):
                #    ignore_boxes_flags[ak] = 1
                                
        else: #negative
            if (str_mode == 'test'): 
                if (file_name == '001030'):
                    corr = '000930'
                elif (file_name == '001130'):
                    corr = '001030'
                #elif (file_name == '000930'):
                #    ignore_boxes_flags[ak] = 1
            else: #training
                if (file_name == '000201'):
                    corr = '000101'
                elif (file_name == '000301'):
                    corr = '000201'
                elif (file_name == '000401'):
                    corr = '000301'
                elif (file_name == '000501'):
                    corr = '000401'
                elif (file_name == '000601'):
                    corr = '000501'
                elif (file_name == '000701'):
                    corr = '000601'
                elif (file_name == '000801'):
                    corr = '000701'
                elif (file_name == '000001'):
                    corr = '000801'
                elif (file_name == '000101'):
                    corr = '000001'            
        file_name = corr
        dist_imgs.append(dist_dir +str_mode+'/'+ str2 +'/' + file_name + '.npz')
#         npy_file = np.load(dist_dir +str_mode+'/'+ str2 +'/' + file_name + '.npz' )
#         dist_imgs.append(npy_file['distance'])   

    return dist_imgs

def binary_rectangle(image, x1,y1, x2,y2):
  output_img = image
  output_img[x1,y1:y2] = 10
  output_img[x2,y1:y2] = 10
  output_img[x1:x2,y1] = 10
  output_img[x1:x2,y2] = 10
  return output_img


def retrieve_distance_dir(labels,ID, str_mode, dist_dir, n_batches):
    dist_imgs = list()
    # list of strings, gives names of videos
    for ak in range(len(labels)):
        if labels[ak,1] == 1:
            str2 = 'positive'
            file_name = ID[ak].decode('utf-8')
        else:
            str2 = 'negative'
            if str_mode == 'training':
              neg_vid = range(1,1+len(os.listdir(dist_dir +str_mode+'/'+str2+'/')))
              cf = int(ID[ak].decode('utf-8'))-1
            else:
              neg_vid = range(830,1+1130)
              cf = int(ID[ak].decode('utf-8'))-830
            cf = np.roll(neg_vid,-2)[cf]
            file_name = '%06d' %cf # negative videos shifted by 2       
        dist_imgs.append(dist_dir +str_mode+'/'+ str2 +'/' + file_name + '.npz')
        
        #hardcode so that the correct video file name is outputted
        #it will correspond with the previously called bounding boxes       
        corr = file_name
        if labels[ak,1] == 1: #positive
            if (str_mode == "test"):
                if (file_name == "000556"):
                    corr = "000506"
                elif (file_name == "000606"):
                    corr = "000556"     
                #elif (file_name == "000506"):
                #    ignore_boxes_flags[ak] = 1           
            else: #training
                if (file_name == "000101"):
                    corr = "000051"
                elif (file_name == "000251"):
                    corr = "000201"
                elif (file_name == "000301"):
                    corr = "000251"
                elif (file_name == "000351"):
                    corr = "000301"
                elif (file_name == "000401"):
                    corr = "000351"
                #file 301 is repeated in batch 122 but already corrected it earlier?
                # elif ((file_name == "000051") or file_name == "000201")
                #    ignore_boxes_flags[ak] = 1
                                
        else: #negative
            if (str_mode == "test"): 
                if (file_name == "001030"):
                    corr = "000930"
                elif (file_name == "001130"):
                    corr = "001030"
                #elif (file_name == "000930"):
                #    ignore_boxes_flags[ak] = 1
            else: #training
                if (file_name == "000201"):
                    corr = "000101"
                elif (file_name == "000301"):
                    corr = "000201"
                elif (file_name == "000401"):
                    corr = "000301"
                elif (file_name == "000501"):
                    corr = "000401"
                elif (file_name == "000601"):
                    corr = "000501"
                elif (file_name == "000701"):
                    corr = "000601"
                elif (file_name == "000801"):
                    corr = "000701"
                elif (file_name == "000001"):
                    corr = "000801"
                elif (file_name == "000101"):
                    corr = "000001"            
        file_name = corr
#         npy_file = np.load(dist_dir +str_mode+'/'+ str2 +'/' + file_name + '.npz' )
#         dist_imgs.append(npy_file['distance'])   

    return dist_imgs


def retrieve_frame_dir1(labels,ID, str_mode, frame_dir, n_batches):
    frame_paths = list()
    frame_path1 = frame_dir
    # list of strings, gives names of videos
    for ak in range(len(labels)):
        if labels[ak,1] == 1: # conditional for positive example
            str2 = 'positive'
            file_name = ID[ak].decode('utf-8')
        else:
            str2 = 'negative'
            if str_mode == 'training':
              neg_vid = range(1,1+len(os.listdir(frame_path1 +str_mode+'/'+str2+'/')))
              cf = int(ID[ak].decode('utf-8'))-1
            else:
              neg_vid = range(830,1+1130)
              cf = int(ID[ak].decode('utf-8'))-830
            print(cf)
            cf = np.roll(neg_vid,-2)[cf]
            file_name = '%06d' %cf # negative videos shifted by 2
            
            
        frames_path = frame_path1 +str_mode+'/'+str2+'/'+file_name+'/'
        frame_paths.append(frames_path)
    return frame_paths

# random functions
def obtain_keys(lis):
    key_ls = list()
    for k in lis:
        integer = re.split('([0-9]+)', k)[1]
        key_ls.append(int(integer))
    return(key_ls)
  
def obtain_keys1(lis):
    key_ls = list()
    for k in lis:
        integer = re.split('([0-9]+)', k)[3]
        key_ls.append(int(integer))
    return(key_ls)
  
def h(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
  
def sort_dir(lis):
    key_ls = h(obtain_keys(lis))
    return [lis[i] for i in key_ls]  
    
    
    
def find_bb_conv(det):
  '''
  :::inputs:::
  input shape: input shape of the original image before resizing
  # input_shape is (720,1280)
  det shape [10,100, 19, 6]: We want to transform to
  :::output:::
  reshape_det, which has shape[10,100,19, 7], where the first four dimensions are the reshaped coordinates, the SECOND-TO-last dimension is the crop that is used
  (with 0 --> middle crop, 1--> top crop, 2--> bottom crop. This pointer is found by seeing where the center coordinate of the bounding box is located)
  '''

  top_crop = 180.  ## hard coded borders
  bottom_crop = 540.
  mid_crop = bottom_crop-top_crop
  det_reshape = np.zeros((10,100,19,7))

  for batch in range(10):
    # input_shape = input_shape_list[batch] 
    input_shape = np.zeros((2,1))
    input_shape[0] = 720
    input_shape[1] = 1280
    
    ## find the correct crops across each frame
    pointers_list = list()
    for frame in range(100):
      indices = np.nonzero(det[batch,frame,:,0])[0]
#       det_altered = det[batch,frame,indices,:]
      if len(indices)>0:
          midpoints = (det[batch,frame,indices,1] + det[batch,frame,indices,3])/2
          difference_arr = np.vstack(( np.absolute(top_crop-midpoints), \
                        np.absolute((bottom_crop -top_crop)-midpoints), np.absolute(bottom_crop-midpoints)))
          crop_pointers = np.argmin(difference_arr, axis = 0)
          crop_pointer = stats.mode(crop_pointers)[0]
          
          pointers_list.append(crop_pointer)
    if len(pointers_list) > 0:
        # print('Pointers List::')
        # print(pointers_list)
        # print(stats.mode(pointers_list)[0][0])
        global_pointer = stats.mode(pointers_list)[0] + 1
    else:
        global_pointer = [[2]]
        
    global_pointer = global_pointer[0][0]
    # print(global_pointer)
    
    for frame in range(100):
        
      indices = np.nonzero(det[batch,frame,:,0])[0]
#       det_altered = det[batch,frame,indices,:]
      if len(indices)>0:

          crop_pointer = global_pointer
          # if input_shape[1] < 1000:
          #   crop_pointer = 1

          for agent in range(19):

            det_slice = det[batch, frame, agent,:].copy()
            if np.sum(det_slice != 0):

              # even if the input image size is different, for the horizontal axis we pad with zeros if it is not 1280, keeping the relative horizontal scaling factor the same
              final_height = 128
              final_width = 416

              det_reshape[batch,frame, agent, 4] = det[batch,frame, agent, 4].copy() # copy over class
              det_reshape[batch,frame, agent, -1] = det[batch,frame, agent, -1].copy() # copy over agent_id

              middle_perc = 0.50
              left = 1-middle_perc
              half = left/2

              vertical_scaling = final_height/(input_shape[0]/2)
              horizontal_scaling = final_width/input_shape[1]

              ## Case 1: Fits into top region
              if crop_pointer == 1:
                det_reshape[batch,frame,agent,-2] = 1

                # no need to shift for top crop
                ################################################################################
                det_reshape[batch,frame, agent, 0] = np.round((det_slice[0])*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)
                ################################################################################

                if det_slice[3]>mid_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                else:
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)
                if det_slice[1]>mid_crop:
                  det_reshape[batch,frame, agent, 1] = 127
                else:
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)


              ## Case 2: Fits into bottom region
              elif crop_pointer == 3:
                det_reshape[batch,frame,agent,-2] = 3
                ################################################################################
                det_reshape[batch,frame, agent, 0] = np.round(det_slice[0]*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)
                ################################################################################
                
                if det_slice[1] < mid_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[1] = det_slice[1] - mid_crop
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)
                    
                if det_slice[3] < mid_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[3] = det_slice[3] - mid_crop
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)

              ## Case 3: Fits into middle region
              else:
                det_reshape[batch,frame,agent,-2] = 2
                ################################################################################
                det_reshape[batch,frame, agent, 0] = np.round(det_slice[0]*horizontal_scaling)
                det_reshape[batch,frame, agent, 2] = np.round(det_slice[2]*horizontal_scaling)
                ################################################################################

                if det_slice[1]<top_crop:
                  det_reshape[batch,frame, agent, 1] = 0
                else:
                  det_slice[1] = det_slice[1] - top_crop
                  det_reshape[batch,frame, agent, 1] = np.round(det_slice[1]*vertical_scaling)
                if det_slice[3]> bottom_crop:
                  det_reshape[batch,frame, agent, 3] = 127
                else:
                  det_slice[3] = det_slice[3] - top_crop
                  det_reshape[batch,frame, agent, 3] = np.round(det_slice[3]*vertical_scaling)
                
                if (det_slice[1]> bottom_crop and det_slice[3]> bottom_crop) or (det_slice[1]< top_crop and det_slice[3]< top_crop):
                    det_reshape[batch,frame, agent, 1] = 0
                    det_reshape[batch,frame, agent, 3] = 0
                    
  return det_reshape
# function: modified version of conversion function