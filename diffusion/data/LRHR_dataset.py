from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from glob import glob
from torchvision import transforms

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            # print(self.hr_path)
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        # 直接读取TID2013数据集进行训练
        elif datatype == 'tid2013':
            # self.hr_path = Util.get_paths_from_images('/mnt/workspace/workgroup/zhaoyang.wzy/tid2013/reference_images')
            # self.sr_path = Util.get_paths_from_images('/mnt/workspace/workgroup/zhaoyang.wzy/tid2013/distorted_images')

            self.hr_path = Util.get_paths_from_images('/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/reference_images')
            self.sr_path = Util.get_paths_from_images('/home/vipsl416-10-wangzhaoyang/IQA_dataset/tid2013/distorted_images')

            # 只抓取一种类型的噪声
            # self.sr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/tid2013/distorted_images/*_02_*')
            # print(self.sr_path)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        # 直接读取LIVE数据集进行训练
        elif datatype == 'live':
            self.hr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/live/databaserelease2' + "/**/*.bmp", recursive=True)
            self.sr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/live/databaserelease2' + "/**/*.bmp", recursive=True)

            # 只抓取一种类型的噪声
            # self.sr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/tid2013/distorted_images/*_02_*')
            # print(self.sr_path)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        # koniq
        elif datatype == 'koniq10k':
            self.hr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/koniq/koniq10k/imgs' + "/**/*.jpg", recursive=True)
            self.sr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/koniq/koniq10k/imgs' + "/**/*.jpg", recursive=True)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        # kadid10k
        elif datatype == 'kadid10k':
            self.hr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/kadid/kadid10k/images' + "/**/*.png", recursive=True)
            self.sr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/kadid/kadid10k/images' + "/**/*.png", recursive=True)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        elif datatype == 'csiq':
            self.hr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/csiq/dst_imgs' + "/**/*.png", recursive=True)
            self.sr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/csiq/dst_imgs' + "/**/*.png", recursive=True)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        elif datatype == 'waterpool':
            self.hr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/exploration_database_and_code/distorted_imgs' + "/**/*.bmp", recursive=True)
            self.sr_path = glob('/home/vipsl416-10-wangzhaoyang/IQA_dataset/exploration_database_and_code/distorted_imgs' + "/**/*.bmp", recursive=True)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
                
        elif datatype == 'pipal':
            self.hr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/PIPAL/Train_Ref' + "/**/*.bmp", recursive=True)
            self.sr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/PIPAL' + "/**/*.bmp", recursive=True)

            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
                # self.data_len = 10

        elif datatype == 'livec':
            self.hr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/ChallengeDB_release/Images' + "/**/*.bmp", recursive=True) + glob('/mnt/workspace/workgroup/zhaoyang.wzy/ChallengeDB_release/Images' + "/**/*.JPG", recursive=True)
            self.sr_path = glob('/mnt/workspace/workgroup/zhaoyang.wzy/ChallengeDB_release/Images' + "/**/*.bmp", recursive=True) + glob('/mnt/workspace/workgroup/zhaoyang.wzy/ChallengeDB_release/Images' + "/**/*.JPG", recursive=True)
            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
                # self.data_len = 10
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        
        # 直接读取tid2013数据集进行训练，参考图像作为HR图片，有噪声的图片作为SR，不要LR
        # 此处是针对所有的噪声类型进行训练，结果不稳定，难以收敛，可能是batch size太小，不能够全部进行拟合的原因。
        elif self.datatype == 'tid2013':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            dir_name = os.path.dirname(self.sr_path[index])
            basename = os.path.basename(self.sr_path[index])
            # img_HR = Image.open(self.hr_path[0]).convert("RGB")

            id_number = basename[1:3]
            hr_path = dir_name.replace('distorted_images', 'reference_images')
            hr_path = os.path.join(hr_path, 'I' + id_number + '.BMP')
            img_HR = Image.open(hr_path).convert("RGB")
            # print(hr_path)
            # asd

        elif self.datatype == 'live':
            # 部分数据尺寸有问题，进行尺寸微调
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")

            original_width, original_height = img_SR.size
            # 定义目标尺寸
            target_width = original_width - (original_width % 16)
            target_height = original_height - (original_height % 16)

            # 定义图像变换
            resize = transforms.Compose([
                transforms.Resize((target_height, target_width)),
            ])
            img_SR = resize(img_SR)
            img_HR = resize(img_HR)

        elif self.datatype == 'koniq10k':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")

        elif self.datatype == 'kadid10k':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")

        elif self.datatype == 'csiq':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")
            
        elif self.datatype == 'waterpool':
            # 部分数据尺寸有问题，进行尺寸微调
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")

            original_width, original_height = img_SR.size
            # 定义目标尺寸
            target_width = original_width - (original_width % 16)
            target_height = original_height - (original_height % 16)

            # 定义图像变换
            resize = transforms.Compose([
                transforms.Resize((target_height, target_width)),
            ])
            img_SR = resize(img_SR)
            img_HR = resize(img_HR)

        elif self.datatype == 'pipal':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_name = os.path.basename(self.sr_path[index])[:5]
            HR_path = os.path.join('/mnt/workspace/workgroup/zhaoyang.wzy/PIPAL/Train_Ref', img_name + '.bmp')
            img_HR = Image.open(HR_path).convert("RGB")

        elif self.datatype == 'livec':
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_HR = Image.open(self.sr_path[index]).convert("RGB")

            original_width, original_height = img_SR.size
            # 定义目标尺寸
            target_width = original_width - (original_width % 16)
            target_height = original_height - (original_height % 16)

            # 定义图像变换
            resize = transforms.Compose([
                transforms.Resize((target_height, target_width)),
            ])
            img_SR = resize(img_SR)
            img_HR = resize(img_HR)
        else:
            # 每次读取第一帧
            # img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_HR = Image.open(self.hr_path[0]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'path': self.sr_path[index]}
