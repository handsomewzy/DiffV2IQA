import os
import torch
import numpy as np
import cv2


class Kadid10k(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Kadid10k, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                if dis[1:3] in list_name:
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


class Kadid10k_diff(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Kadid10k_diff, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                if dis[1:3] in list_name:
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
        # d_img = cv2.imread(d_img_name, cv2.IMREAD_COLOR)
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)

        # 读取扩散模型生成的模型
        basename = os.path.basename(d_img_name)
        dir_path = '/mnt/workspace/workgroup/zhaoyang.wzy/kadid10k/diffusion_imgs_pipal' # 全部数据
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
        x = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype('float32') / 255
        x = np.transpose(x, (2, 0, 1))   
        return x