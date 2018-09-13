from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import random
import math
import os


#################################
### training.py made to be executed in the jupyter notbook
### modify the following import  : from utils import * => from .utils import *
### if you want to use it from external .py file


import dataset
from utils import *
from cfg import parse_cfg
from darknet import Darknet




########################## 
# see README for more info

class BatchIt:
    def __init__(self, trainLoader):
        self.n = trainLoader.__iter__()
        self.error = 0

    def __iter__(self):
        return self

    def next(self):
        try:
            nextB = self.n.next()
        except StopIteration:
            raise StopIteration()
        except Exception as e:
            print(e)
            self.error += 1
            nextB = self.next()
            print(self.error)

        return nextB
    
    
    
    
class Training:
    
    
    def __init__(self, dataFile, cfgFile, weightFile = None):
        
        # Training settings
        self.datacfg = dataFile
        self.cfgfile = cfgFile
        self.weightfile = weightFile
        
        self.data_options = read_data_cfg(self.datacfg)
        self.net_options = parse_cfg(self.cfgfile)[0]
        
        self.trainlist = self.data_options['train']
        self.testlist = self.data_options['valid']
        self.backupdir = self.data_options['backup']
        self.nsamples = file_lines(self.trainlist)
        self.gpus = self.data_options['gpus']  # e.g. 0,1,2,3
        self.ngpus = len(self.gpus.split(','))
        self.num_workers = int(self.data_options['num_workers'])
        
        self.batch_size = int(self.net_options['batch'])
        self.max_batches = int(self.net_options['max_batches'])
        self.learning_rate = float(self.net_options['learning_rate'])
        self.momentum = float(self.net_options['momentum'])
        self.decay = float(self.net_options['decay'])
        self.steps = [float(step) for step in self.net_options['steps'].split(',')]
        self.scales = [float(scale) for scale in self.net_options['scales'].split(',')]
        
        # Train parameters
        self.max_epochs = self.max_batches * self.batch_size / self.nsamples + 1
        self.use_cuda = True
        self.seed = int(time.time())
        self.eps = 1e-5
        self.save_interval = 1  # epoches
        self.dot_interval = 70  # batches
        
        self.max_epochs = 200
        
        # Test parameters
        self.conf_thresh = 0.25
        self.nms_thresh = 0.4
        self.iou_thresh = 0.5
        
        if not os.path.exists(self.backupdir):
            os.mkdir(self.backupdir)

        ###############
        torch.manual_seed(self.seed)
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
            torch.cuda.manual_seed(self.seed)
        
        self.model = Darknet(parse_cfg(self.cfgfile))
        self.region_loss = self.model.loss
        
        if self.weightfile != None:
            self.model.load_weights(self.weightfile)
        
        self.model.print_network()
        
        self.region_loss.seen = self.model.seen
        self.processed_batches = self.model.seen / self.batch_size
        
        self.init_width = self.model.width
        self.init_height = self.model.height
        self.init_epoch = self.model.seen / self.nsamples
        
        self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True} if self.use_cuda else {}
        self.test_loader = torch.utils.data.DataLoader(
            dataset.listDataset(self.testlist, shape=(self.init_width, self.init_height),
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]), train=False),
            batch_size=self.batch_size, shuffle=False, **self.kwargs)
        
        if self.use_cuda:
            if self.ngpus > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()
        
        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                params += [{'params': [value], 'weight_decay': self.decay * self.batch_size}]

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate / self.batch_size, momentum=self.momentum, dampening=0,
                              weight_decay=self.decay * self.batch_size)


    def startTraining(self):
        print("start at " + str(self.init_epoch) + " to " + str(self.max_epochs))
        for epoch in range(self.init_epoch, self.max_epochs):
            self.train(epoch)
            self.test(epoch)
            
    
    ###########################
    
    def adjust_learning_rate(self, optimizer, batch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learning_rate
        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if batch >= self.steps[i]:
                lr = lr * scale
                if batch == self.steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / self.batch_size
        return lr
    
    
    def train(self, epoch):
        t0 = time.time()
        if self.ngpus > 1:
            cur_model = self.model.module
        else:
            cur_model = self.model
        train_loader = torch.utils.data.DataLoader(
            dataset.listDataset(self.trainlist, shape=(self.init_width, self.init_height),
                                shuffle=True,
                                train=True,
                                seen=self.model.module.seen,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers),
            batch_size=self.batch_size, shuffle=False, drop_last=True, **self.kwargs)
    
        lr = self.adjust_learning_rate(self.optimizer, self.processed_batches)
        logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
        self.model.train()
        t1 = time.time()
        avg_time = torch.zeros(9)
        for batch_idx, (data, target) in enumerate(BatchIt(train_loader)):
            t2 = time.time()
            self.adjust_learning_rate(self.optimizer, self.processed_batches)
            self.processed_batches = self.processed_batches + 1
            # if (batch_idx+1) % dot_interval == 0:
            #    sys.stdout.write('.')
    
            if self.use_cuda:
                data = data.cuda()
                # target= target.cuda()
            t3 = time.time()
            data, target = Variable(data), Variable(target)
            t4 = time.time()
            self.optimizer.zero_grad()
            t5 = time.time()
            print(data.shape)
            output = self.model(data)
            print(output.shape)
            print(target.shape)
            t6 = time.time()
            self.region_loss.seen = self.region_loss.seen + data.data.size(0)
            loss = self.region_loss(output, target)
            t7 = time.time()
            loss.backward()
            t8 = time.time()
            self.optimizer.step()
            t9 = time.time()
            if False and batch_idx > 1:
                avg_time[0] = avg_time[0] + (t2 - t1)
                avg_time[1] = avg_time[1] + (t3 - t2)
                avg_time[2] = avg_time[2] + (t4 - t3)
                avg_time[3] = avg_time[3] + (t5 - t4)
                avg_time[4] = avg_time[4] + (t6 - t5)
                avg_time[5] = avg_time[5] + (t7 - t6)
                avg_time[6] = avg_time[6] + (t8 - t7)
                avg_time[7] = avg_time[7] + (t9 - t8)
                avg_time[8] = avg_time[8] + (t9 - t1)
                print('-------------------------------')
                print('       load data : %f' % (avg_time[0] / (batch_idx)))
                print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
                print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
                print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
                print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
                print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
                print('        backward : %f' % (avg_time[6] / (batch_idx)))
                print('            step : %f' % (avg_time[7] / (batch_idx)))
                print('           total : %f' % (avg_time[8] / (batch_idx)))
            t1 = time.time()
        print('')
        t1 = time.time()
        logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
        if (epoch + 1) % self.save_interval == 0:
            logging('save weights to %s/%06d.weights' % (self.backupdir, epoch + 1))
            cur_model.seen = (epoch + 1) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' % (self.backupdir, epoch + 1))
    
    
    
    def test(self, epoch):
        
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i
    
        self.model.eval()
        if self.ngpus > 1:
            cur_model = self.model.module
        else:
            cur_model = self.model
        num_classes = cur_model.num_classes
        anchors = cur_model.anchors
        num_anchors = cur_model.num_anchors
        total = 0.0
        proposals = 0.0
        correct = 0.0
    
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if self.use_cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = self.model(data).data
            all_boxes = get_region_boxes(output, self.conf_thresh, num_classes, anchors, num_anchors)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, self.nms_thresh)
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
    
                total = total + num_gts
    
                for i in range(len(boxes)):
                    if boxes[i][4] > self.conf_thresh:
                        proposals = proposals + 1
    
                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    for j in range(len(boxes)):
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou
                    if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                        correct = correct + 1
    
        precision = 100.0 * correct / (proposals + self.eps)
        recall = 100.0 * correct / (total + self.eps)
        fscore = 200.0 * precision * recall / (precision + recall + self.eps)
        logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    

