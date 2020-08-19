from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *
import numpy as np
import cv2


def getImg(img_path, dims = (416,416)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dims)
    img = img[:, :, ::-1].transpose((2,0,1)) # BGR -> RGB -> HWXC --> C x H x W
    img = img[np.newaxis, :, :, :]/255.0 #add axis, convert to 0-1 range
    img = torch.from_numpy(img).float()
    return Variable(img)


def parseCfg(cfg):
    f = open(cfg, 'r')
    lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for l in lines:
        if l[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = l[1:-1].rstrip()
        else:
            key, val=l.split("=")
            block[key.rstrip()] = val.lstrip()
    
    blocks.append(block)

    return blocks




def createModules(blocks):
    net_info = blocks[0] # first entry in 
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    # create pytorch module for each block
    for i, x in enumerate(blocks[1:]): #ignore first net block 
        module = nn.Sequential() #Each block has >1 layer; create sequential set for each block
        if x['type'] == "convolutional":
            activation = x['activation']
            try:
                batch_normalize=int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size-1)//2
            else:
                pad = 0

            c = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(i), c)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batchNorm_{0}".format(i), bn)
            if activation == 'leaky':
                a = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('activationLeaky_{0}'.format(i), a)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{0}".format(i), upsample)

        elif x['type'] == 'route':
            #https://github.com/AlexeyAB/darknet/issues/487#issuecomment-374902735
            # route layers combine current layer output with prior in order to bring in earlier features

            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start-i
            if end > 0:
                end = end -i
            route = emptyLayer()
            module.add_module("routeLayer_{0}".format(i), route)
            
            if end < 0:
                filters = output_filters[i+start] + output_filters[i+end]
            else:
                filters = output_filters[i + start]

        elif x['type'] == "shortcut":
            shortcut = emptyLayer()
            module.add_module("shortcut_{0}".format(i), shortcut)

        elif x['type'] == 'yolo':
            mask = [int(y) for y in x['mask'].split(",")]
            anchors = [int(y) for y in x['anchors'].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            d = detectionLayer(anchors)
            module.add_module("detection_{0}".format(i), d)

        module_list.append(module)
        prev_filters=filters
        output_filters.append(filters)
    
    return (net_info, module_list)


class detectionLayer(nn.Module):
    def __init__(self, anchors):
        super(detectionLayer, self).__init__()
        self.anchors = anchors


class emptyLayer(nn.Module):
    def __init__(self):
        super(emptyLayer, self).__init__() 

class darknet(nn.Module):
    def __init__(self, cfg):
        super(darknet, self).__init__()
        self.blocks = parseCfg(cfg)
        self.net_info, self.module_list = createModules(self.blocks)
    def forward(self, x, CUDA):
        # x: input
        # CUDA: accelerate w/ gpu (t/f)
        modules=self.blocks[1:]
        outputs = {} #cache output in this dict for route & shortcut layers

        write = 0
        for i, m in enumerate(modules):
            module_type=(m["type"])

            if ((module_type == "convolutional") or (module_type == "upsample")):
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = m['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1]-i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2,), 1)
            elif module_type == 'shortcut':
                x = outputs[i-1] + outputs[i+int(m['from'])]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(m['classes'])

                x = x.data
                x = predictTransform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x

        return detections


# ~main~

def createModules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = emptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = emptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = detectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)


model = darknet("cfg/yolov3.cfg")
inp = getImg('/home/ihasdapie/Projects/YOLOv3_Adventures/YoloFromScratch/dog-cycle-car.png')
pred = model(inp, torch.cuda.is_available())
print (pred)


