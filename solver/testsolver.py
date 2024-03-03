#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2021-01-19 21:00:18
@Description: file content
'''
from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        
        self.model = net(
            num_channels=self.cfg['data']['n_colors'], 
            base_filter=64,  
            args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0]) 
            
            self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        avg_time = []
        for batch in self.data_loader:
            ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
            if self.cuda:
                ms_image = ms_image.cuda(self.gpu_ids[0])
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])
                #print(torch.max(ms_image))
                #print(torch.min(ms_image))  

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(lms_image, bms_image, pan_image)

            t1 = time.time()

            if self.cfg['data']['normalize']:
                ms_image = (ms_image+1) /2
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            #print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK') #
            self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        
    def eval(self):
        self.model.eval()
        avg_time= []
        for batch in self.data_loader:
            ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
            if self.cuda:
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(lms_image, bms_image, pan_image)

            t1 = time.time()

            if self.cfg['data']['normalize']:
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        #print((save_img.max()))
        # save img
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name
        save_img = np.uint8(save_img*255).astype('uint8') #
        #print(save_img.max())
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)
  
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':
            self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':
            self.dataset = get_eval_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.eval()
        else:
            raise ValueError('Mode error!')