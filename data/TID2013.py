import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import cv2
import random
import torch
import torchvision

import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
import time


class TID2013Folder(data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(TID2013Folder, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                score, dis = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(os.path.join(dis_path, 'distorted_images', dis))
                    score_data.append(score)
        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = self.read_image_with_retry(d_img_name)

        # 读取扩散模型生成的模型
        basename = os.path.basename(d_img_name)
        dir_path = '/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/diffusion_imgs' # 全部数据
        diff_0_path = os.path.join(dir_path, basename + '_0.png')
        diff_1_path = os.path.join(dir_path, basename + '_1.png')
        diff_2_path = os.path.join(dir_path, basename + '_2.png')


        d0 = self.read_image_with_retry(diff_0_path)
        d1 = self.read_image_with_retry(diff_1_path)
        d2 = self.read_image_with_retry(diff_2_path)

        d_img, d0, d1, d2 = self.preprocess(d_img), self.preprocess(d0), self.preprocess(d1), self.preprocess(d2)
        score = self.data_dict['score_list'][idx]
        # print(d_img.shape,d0.shape)
        # if self.transform:
        #     t_d_img = self.transform(d_img)
        #     t_d0 = self.transform(d0)
        #     t_d1 = self.transform(d1)
        #     t_d2 = self.transform(d2)
        # # print(t_d0.shape, d0.shape)

        sample = {
            'd_img_org': d_img,
            'd0': d0,
            'd1': d1,
            'd2': d2,
            'score': score
        }
        sample = self.transform(sample)

        # print(sample)
        # asd
        
        return sample

    def preprocess(self, d_img):
        # x = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        x = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype('float32') / 255
        x = np.transpose(x, (2, 0, 1))   
        return x  
    
        # try:
        #     # 检查图像是否为空
        #     if d_img is None:
        #         print(d_img)
        #         raise ValueError("Image is None")
            
        #     # 图像预处理
        #     x = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        #     x = np.array(x).astype('float32') / 255
        #     x = np.transpose(x, (2, 0, 1))
        #     return x
        # except Exception as e:
        #     print(f"Error processing image: {e}")
        #     return None



    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def read_image_or_report(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Failed to read image from path: {path}")
        return image
    
    def read_image_with_retry(self, path, retries=3, delay=1):
        attempt = 0
        while attempt < retries:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                return image
            else:
                print(f"Error: Failed to read image from path: {path}. Attempt {attempt + 1} of {retries}")
                time.sleep(delay)
                attempt += 1
        raise ValueError(f"Failed to read image after {retries} attempts: {path}")



# image = cv2.imread('/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/diffusion_imgs/i09_20_3.bmp_0.png', cv2.IMREAD_COLOR)
# print(image)