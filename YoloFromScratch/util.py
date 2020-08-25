from __future__ import division


import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predictTransform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    # takes detection feature map and creates list of bounding boxes

    batch_size = prediction.size(0)
    stride = inp_dim//prediction.size(2)
    grid_size = inp_dim//stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

     #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors


    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))


    prediction[:,:,:4] *= stride


    return prediction

def writeResults(prediction, confidence, num_classes, nms_conf = 0.4):
    # create & apply mask over prediction: set bounding boxes w/ objectness < conf threshold to 0
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    # calculate IoU

    # transform bounding box from center(x,y), w, h to top_left, top_right, bottom_left, bottom_right
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write=False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)

        image_pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero(image_pred[:,4], as_tuple=False).squeeze()

        try:
            image_pred = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred.shape[0] == 0:
            continue
        img_classes = unique(image_pred)

        img_classes = unique(image_pred[:, -1])

        for cls in img_classes:
            # get detections for each class (cls) 
            cls_mask = image_pred*(image_pred[:,-1]==cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2], as_tuple=False).squeeze()
            image_pred_class = image_pred[class_mask_ind].view(-1,7)

            n_detections = image_pred_class.size(0)
            for i in range(n_detections):
                try:
                    ious = IoU(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except (ValueError, IndexError):
                    break

                iou_mask = (ious<nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask


                non_zero_ind = torch.nonzero(image_pred_class[:,4].squeeze(), as_tuple=False)
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True

            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output

    except:
        return 0




def IoU(b1, b2):
    ir_x1 = torch.max(b1[:,0], b2[:,0])
    ir_y1 = torch.max(b1[:,1], b2[:,1])
    ir_x2 = torch.max(b1[:,2], b2[:,2])
    ir_y2 = torch.max(b1[:,3], b2[:,3])

    intersection = torch.clamp(ir_x2 - ir_x1 + 1, min=0) * torch.clamp(ir_y2-ir_y1+1, min=0)

    u1 = (b1[:,2]-b1[:,0]+1)*(b1[:,3]-b1[:,1]+1)    
    u2 = (b2[:,2]-b2[:,0]+1)*(b2[:,3]-b2[:,1]+1)    

    return (intersection/(u1+u2-intersection))


def unique(tensor):
    t = tensor.cpu().numpy()
    t= np.unique(t)
    t = torch.from_numpy(t)

    res = tensor.new(t.shape)
    res.copy_(t)
    return res

