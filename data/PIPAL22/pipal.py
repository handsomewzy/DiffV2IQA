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
from glob import glob


class PIPAL(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, transform, keep_ratio):
        super(PIPAL, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        name_type = {}
        
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                
                # obtain the spliting parts
                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in name_type.keys():
                    name_type[img_name + '_' + dis_type] = 1
                else:
                    name_type[img_name + '_' + dis_type] += 1

        count_name_type = {}
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]

                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in count_name_type.keys():
                    count_name_type[img_name + '_' + dis_type] = 1
                else:
                    count_name_type[img_name + '_' + dis_type] += 1

                if count_name_type[img_name + '_' + dis_type] <= int(name_type[img_name + '_' + dis_type] * keep_ratio):
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class PIPALFolder(data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(PIPALFolder, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)
        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(d_img_name, cv2.IMREAD_COLOR)

        # 读取扩散模型生成的模型
        basename = os.path.basename(d_img_name)
        dirname = os.path.dirname(d_img_name)
        dir_path = dirname.replace('PIPAL', 'PIPAL_diffusion')
        diff_0_path = os.path.join(dir_path, basename + '_0.png')
        diff_1_path = os.path.join(dir_path, basename + '_1.png')
        diff_2_path = os.path.join(dir_path, basename + '_2.png')

        d0 = cv2.imread(diff_0_path, cv2.IMREAD_COLOR)
        d1 = cv2.imread(diff_1_path, cv2.IMREAD_COLOR)
        d2 = cv2.imread(diff_2_path, cv2.IMREAD_COLOR)

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

        # LIVE 数据大小参差不齐，进行大小匹配
        # new_height = (d_img.shape[0] // 16) * 16
        # new_width = (d_img.shape[1] // 16) * 16
        # x = cv2.resize(d_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        x = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype('float32') / 255
        x = np.transpose(x, (2, 0, 1))   
        return x  

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

# import os
# from glob import glob

# # 获取所有图像的绝对路径
# dis_list = glob('/mnt/workspace/workgroup/zhaoyang.wzy/PIPAL/**/*.bmp', recursive=True)

# # 创建一个映射，将图像名称映射到对应的绝对路径
# name_to_path = {os.path.basename(path): path for path in dis_list}

# # 读取包含图像名称和分数的文件
# with open('/mnt/workspace/workgroup/zhaoyang.wzy/MANIQA/data/PIPAL22/pipal22_train.txt', 'r') as listFile:
#     # 准备用于存储匹配后结果的列表
#     matched_data = []
    
#     # 遍历listFile中的每一行
#     for line in listFile:
#         dis, score = line.split(',')
#         # 检查图像名称是否在name_to_path映射中
#         if dis in name_to_path:
#             # 获取对应的绝对路径
#             abs_path = name_to_path[dis]
#             # 存储绝对路径和分数的元组
#             matched_data.append((abs_path, float(score)))

# # 将匹配后的结果写入新的文本文件
# with open('/mnt/workspace/workgroup/zhaoyang.wzy/MANIQA/data/PIPAL_path_label.txt', 'w') as outputFile:
#     # 遍历所有匹配的数据
#     for path, score in matched_data:
#         # 将绝对路径和分数写入文件，中间以空格隔开
#         outputFile.write(f"{path} {score}\n")