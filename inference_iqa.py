import argparse
import glob
import os
# from pyiqa import create_metric
from tqdm import tqdm
import csv
from time import time
from config import Config
# import time

import torch
from models.VCRNet import demoIQA
from models.maniqa import MANIQA
# from models.maniqa_diff2 import MANIQA
import cv2
import numpy as np
from torchvision import transforms
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def main():
    defend_model_name = 'DiViIQA'
    attack_model_name = 'VCRNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = Config({
        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
    })
    
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default='/home/vipsl416-10-wangzhaoyang/DIVIIQA/gMAD/{}_bottom50'.format(defend_model_name), help='input image/folder path.')
    # parser.add_argument('-t', '--target', type=str, default='/home/vipsl416-10-wangzhaoyang/IQA_dataset/exploration_database_and_code/distorted_imgs', help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='NR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, default='/home/vipsl416-10-wangzhaoyang/DIVIIQA/gMAD/{}_bottom50/{}.csv'.format(defend_model_name,attack_model_name), help='path to save results.')
    # parser.add_argument('--save_file', type=str, default='/home/vipsl416-10-wangzhaoyang/DIVIIQA/gMAD/{}.csv'.format(defend_model_name), help='path to save results.')

    args = parser.parse_args()

    metric_name = args.metric_name.lower()
    
    transform=transforms.Compose([Normalize(0.5, 0.5),ToTensor()])
    # set up IQA model
    # iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
    # metric_mode = iqa_model.metric_mode
    # print(iqa_model.score_range)
    
    # 加载自己对应的模型，VCRNet，MANIQA和DIVIIQA，再LIVE数据集上训练了，进行gMAD
    iqa_model = demoIQA()
    # iqa_model = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
    # patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
    # depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    metric_mode = 'NR'
    
    iqa_model.load_state_dict(torch.load("/home/vipsl416-10-wangzhaoyang/DIVIIQA/output/models/gMAD/LIVE_VCRNet/epoch223.pt"))
    # iqa_model.load_state_dict(torch.load("/home/vipsl416-10-wangzhaoyang/DIVIIQA/output/models/gMAD/LIVE_maniqa/epoch49.pt"))
    # iqa_model.load_state_dict(torch.load("/home/vipsl416-10-wangzhaoyang/DIVIIQA/output/models/gMAD/LIVE_Diviiqa/epoch43.pt"))
    

    if os.path.isfile(args.target):
        input_paths = [args.target]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.target, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

    if args.save_file:
        sf = open(args.save_file, 'w')
        sfwriter = csv.writer(sf)

    avg_score = 0
    test_img_num = len(input_paths)
    if metric_name != 'fid':
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == 'FR':
                # print(img_path)
                ref_img_path = ref_paths[idx]
                
                # ref_img_path = 
            else:
                ref_img_path = None

            start_time = time()
            # score = iqa_model(img_path, ref_img_path).cpu().item()
            # print(img_path)
            img = read_image_with_retry1(img_path)
            img = preprocess(img)
            img = img.unsqueeze(0)
            
            # # 读取其余的图片，扩散模型生成的
            # diff_path = img_path.replace('pristine_images', 'diffusion_imgs')
            # basename = os.path.basename(img_path)
            # diff_path = '/home/vipsl416-10-wangzhaoyang/IQA_dataset/exploration_database_and_code/distorted_diffusion_imgs'
            # d1_path = os.path.join(diff_path,basename + '_0.png')
            # d2_path = os.path.join(diff_path,basename + '_1.png')
            # d3_path = os.path.join(diff_path,basename + '_2.png')
            
            # d1,tag = read_image_with_retry(d1_path)
            # # print(d1,tag)
            # if tag == 1:
            #     continue
            # d2,tag = read_image_with_retry(d2_path)
            # if tag == 1:
            #     continue
            # d3,tag = read_image_with_retry(d3_path)
            # if tag == 1:
            #     continue
            
            # # print(d1,d2)
            # d1,d2,d3 = preprocess(d1),preprocess(d2),preprocess(d3)
            # d1,d2,d3 = d1.unsqueeze(0),d2.unsqueeze(0),d3.unsqueeze(0)
                  
            # 将模型和输入数据都移动到 CPU 上
            iqa_model = iqa_model.to('cuda')
            img = img.to('cuda')
            # d1 = d1.to('cuda')
            # d2 = d2.to('cuda')
            # d3 = d3.to('cuda')
            
            score = iqa_model(img).item()
            # score = iqa_model(img,d1,d2,d3).item()
            
            end_time = time()
            avg_score += score
            pbar.update(1)
            pbar.set_description(f'{metric_name} of {img_name}: {score}')
            pbar.write(f'{metric_name} of {img_name}: {score}\tTime: {end_time - start_time:.2f}s')
            if args.save_file:
                sfwriter.writerow([img_name, score])
            
        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(args.target), 'input path must be a folder for FID.'
        avg_score = iqa_model(args.target, args.ref)
    
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    msg = f'Average {metric_name} score of {args.target} with {test_img_num} images is: {avg_score}'
    print(msg)
    if args.save_file:
        sf.close()

    if args.save_file:
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')

def preprocess(d_img):
    x = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.array(x).astype('float32') / 255
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).type(torch.cuda.FloatTensor)
    return x  


# def preprocess(d_img):
#     # x = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)

#     # LIVE 数据大小参差不齐，进行大小匹配
#     new_height = (d_img.shape[0] // 16) * 16
#     new_width = (d_img.shape[1] // 16) * 16
#     x = cv2.resize(d_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

#     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     x = np.array(x).astype('float32') / 255
#     x = np.transpose(x, (2, 0, 1))   
#     return x  

from time import time
def read_image_with_retry1(path, retries=3, delay=1):
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


def read_image_with_retry(path, retries=2):
    attempt = 0
    while attempt < retries:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            tag = 0
            return image,tag
        else:
            print(f"Error: Failed to read image from path: {path}. Attempt {attempt + 1} of {retries}")
            # time.sleep(delay)
            # attempt += 1
            tag =1
            return None, tag
    raise ValueError(f"Failed to read image after {retries} attempts: {path}")



def process_image(img_path):
    """处理单个图像"""
    # diff_path = img_path.replace('pristine_images', 'diffusion_imgs')
    basename = os.path.basename(img_path)
    diff_path = '/home/vipsl416-10-wangzhaoyang/IQA_dataset/exploration_database_and_code/distorted_diffusion_imgs'
    d1_path = os.path.join(diff_path, basename + '_0.png')
    d2_path = os.path.join(diff_path, basename + '_1.png')
    d3_path = os.path.join(diff_path, basename + '_2.png')

    d1 = read_image_with_retry(d1_path)
    d2 = read_image_with_retry(d2_path)
    d3 = read_image_with_retry(d3_path)

    if d1 is None or d2 is None or d3 is None:
        print(f"跳过图像 {img_path} 由于读取失败")
        return  (0,0,0)
    else:
        d1, d2, d3 = preprocess(d1), preprocess(d2), preprocess(d3)
        d1, d2, d3 = d1.unsqueeze(0), d2.unsqueeze(0), d3.unsqueeze(0)
        return d1, d2, d3

if __name__ == '__main__':
    main()
