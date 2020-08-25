from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()

def resizeImage(img, dims):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = dims
    nw = int(img_w*min(w/img_w, h/img_h)) 
    nh = int(img_h*min(w/img_w, h/img_h))
 
    resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    
    out = np.full((h, w, 3), 128)
    out[(h-nh)//2:(h-nh)//2+nh,(w-nw)//2:(w-nw)//2+nw,:] = resized_img

    return out

def prepImage(img, dims):
    img = cv2.resize(img, (dims, dims))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img


def writeImage(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def load_classes(nf):
    return(open(nf, 'r').read().split('\n')[:-1])

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()


classes = load_classes("data/coco.names")
num_classes = len(classes)

model = darknet(args.cfgfile)
model.load_weights(args.weightsfile)

model.net_info['height'] = args.reso

assert int(args.reso) % 32 == 0
assert int(args.reso) > 32

if CUDA:
    model.cuda()

model.eval()

try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

imgs = [cv2.imread(x) for x in imlist]

im_batches = list(map(prepImage, imgs, [int(args.reso) for x in range(len(imlist))]))

im_dim_list = [(x.shape[1], x.shape[0]) for x in imgs]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

leftover = 0

if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist)//batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)]  



write = 0

for i, batch in enumerate(im_batches):
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)
    prediction = writeResults(prediction, confidence, num_classes, nms_conf=nms_thresh)

    if type(prediction) == int:
        for j, image in enumerate(imlist[i*batch_size:min((i+1)*(batch_size, len(imlist)))]):
            im_id = i*batch_size+j
        continue

    prediction[:,0] += i*batch_size
    
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for j, image in enumerate(imlist[i*batch_size:min((i+1)*batch_size, len(imlist))]):
        im_id = i*batch_size+j
        objs = [classes[int(x[-1])] for x in output if int(x[0] == im_id)]

    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError as e:
    print("No detections!")
    print(e)
    exit()


im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (int(args.reso) - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (int(args.reso) - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

list(map(lambda x: writeImage(x, imgs), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, imgs))

torch.cuda.empty_cache()
    


