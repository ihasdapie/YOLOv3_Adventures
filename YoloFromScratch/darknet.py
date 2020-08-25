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


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

# def parseCfg(cfg):
#     f = open(cfg, 'r')
#     lines = f.read().split('\n')
#     lines = [x for x in lines if len(x) > 0]
#     lines = [x for x in lines if x[0] != '#']
#     lines = [x.rstrip().lstrip() for x in lines]

#     block = {}
#     blocks = []

#     for l in lines:
#         if l[0] == "[":
#             if len(block) != 0:
#                 blocks.append(block)
#                 block = {}
#             block['type'] = l[1:-1].rstrip()
#         else:
#             key, val=l.split("=")
#             block[key.rstrip()] = val.lstrip()
    
#     blocks.append(block)

#     return blocks


def parseCfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
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
            upsample = nn.Upsample(align_corners=True, scale_factor=2, mode="bilinear")
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
                from_ = int(m['from'])
                x = outputs[i-1] + outputs[i+from_]

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

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

