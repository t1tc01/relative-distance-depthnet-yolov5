from __future__ import absolute_import, division, print_function

import torch
import cv2 as cv

import torch
import cv2 as cv

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from util import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
import numpy as np
from PIL import Image
import copy

import time


#Return pandas DataFrame contains (x_min, y_min) and (x_max, y_max) and classes of object in frame
def model_yolov5(frame, model):
  # Model
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 

  # Inference
  results = model(frame)
  results.print()
  return results.pandas().xyxy[0]



#Return an numpy array represents for depth map
def depth_predict(image_path, encoder ,depth_decoder):
  
  original_width, original_height = image_path.shape[1], image_path.shape[0]

  # PREDICTING ON EACH IMAGE IN TURN
  with torch.no_grad():
      
      # Load image and preprocess
      image_path = cv.resize(image_path,(feed_width, feed_height), cv.INTER_LANCZOS4)
      image_path = transforms.ToTensor()(image_path).unsqueeze(0)

      # PREDICTION, encoder and decoder
      image_path = image_path.to(device)
      features = encoder(image_path)
      outputs = depth_decoder(features)

      disp = outputs[("disp", 0)]
      disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

      scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
      metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

      metric_depth = resize_depth_map(metric_depth, original_width, original_height)

      return metric_depth


def resize_depth_map(metric_depth, original_width, original_height):
  metric_depth = torch.from_numpy(metric_depth)
  metric_depth_resized = torch.nn.functional.interpolate(metric_depth,
    (original_height, original_width), mode="bilinear", align_corners=False)
  
  # Saving colormapped depth image
  metric_depth_resized_np = metric_depth_resized.squeeze().cpu().numpy()
  return metric_depth_resized_np


#Calculate relative distance of objects in the image from depth map and bounder box 
#depth_map : nparray
#data: Dataframe obtains from yolov5
#return dataframe that contain collumn "rev_distance"
def calculate_rev(depth_map, data):
  rev_dis = []
  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])
    y_max = int(row[1]['ymax'])

    rev = 0
    num = (y_max - y_min) * (x_max - x_min)
    for i in range(y_min, y_max):
      for j in range(x_min, x_max):
        rev += depth_map[i, j]
    rev /= num
    rev_dis.append(rev)

  data['rev_distance'] = rev_dis
  return data
  


#Drawing label and distance on frame
#frame: image 
#data: dataFrame contain relative distance
def drawing_output(frame, model, encoder, depth_decoder):
  frame_temp = copy.copy(frame)
  y = model_yolov5(frame_temp, model)
  map = depth_predict(frame, encoder, depth_decoder)
  data = calculate_rev(map, y)

  for row in data.iterrows():
    x_min = int(row[1]['xmin'])
    y_min = int(row[1]['ymin'])
    x_max = int(row[1]['xmax'])
    y_max = int(row[1]['ymax'])

    name_label = row[1]['name']
    rev = row[1]['rev_distance']

    str_output = name_label + ": " + str(int(rev))

    cv.rectangle(frame,(x_min,y_min),
                      (x_max, y_max),
                      (0, 0, 255), 2, 8)
    
    
    cv.putText(frame,str_output,(x_min, y_min) ,cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv.LINE_AA)
  return frame



if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) 
    model_name = "mono+stereo_640x192"
    
    #use GPU
    if torch.cuda.is_available():
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")
      
    #
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    
    
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    #
    cap = cv.VideoCapture(0)
    pre_timeframe = 0
    new_timeframe = 0
    
    #runtest on video
    # vid_path = os.path.join('assets','driving.mp4')
    # cap = cv.VideoCapture(vid_path)
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out_path = os.path.join('assets','outpy.avi')
    # out = cv.VideoWriter(out_path,cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    while True:
        ret, frame = cap.read()
        if ret == False: 
            break

        frame = drawing_output(frame, model, encoder, depth_decoder)
        
        new_timeframe = time.time()
        fps = 1/(new_timeframe- pre_timeframe)
        pre_timeframe = new_timeframe
        fps = int(fps)
        cv.putText(frame, str(fps), (8, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 4)
        
        #out.write(frame)
        cv.imshow("video",frame)
        
        if cv.waitKey(1) == ord('q'):
            break
    #out.release()
    cap.release()
    cv.destroyAllWindows()
  


