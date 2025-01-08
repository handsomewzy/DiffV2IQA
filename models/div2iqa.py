import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import numpy as np
from einops import rearrange
import os
# device = torch.device("cuda:0")  # 替换为目标设备的索引

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class RQABlock(nn.Module):
    def __init__(self, dim=1000, drop=0.1):
        super().__init__()
        self.x_linear = nn.Linear(dim, 784)
        self.c_q = nn.Linear(dim, 784)
        self.c_k = nn.Linear(dim, 784)
        self.c_v = nn.Linear(dim, 784)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.input_size = 28

    def forward(self, x, d0, d1, d2):
        _x = self.x_linear(x)
        # B, C, N = x.shape
        q = self.c_q(torch.abs(x-d0))
        k = self.c_k(torch.abs(x-d1))
        v = self.c_v(torch.abs(x-d2))
        
        _x = rearrange(_x, 'b (c h w) -> b (h w) c', h=self.input_size, w=self.input_size, c=1)
        q = rearrange(q, 'b (c h w) -> b (h w) c', h=self.input_size, w=self.input_size, c=1)
        k = rearrange(k, 'b (c h w) -> b (h w) c', h=self.input_size, w=self.input_size, c=1)
        v = rearrange(v, 'b (c h w) -> b (h w) c', h=self.input_size, w=self.input_size, c=1)
        B, C, N = _x.shape
        
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        # x = attn @ v
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class DIV2IQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)
        
        # 缩减一下噪声等级的维度 embed_dim / 4
        self.noise_embed_dim = 192
        self.noise_embedding = nn.Embedding(4, self.noise_embed_dim)  # 噪声等级的嵌入层，直接进行编码
        self.noise_levels = nn.Parameter(torch.zeros(1, 4, self.noise_embed_dim)) # 可供学习的噪声等级

        # self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        # 四个，每个数据提取2个特征，再加上噪声的embedding，先保持embedding一致，维度下降过快，加入一些缓和的
        self.conv1 = nn.Conv2d(embed_dim * 8 + self.noise_embed_dim * 4, embed_dim * 3, 1, 1, 0)
        # self.conv1 = nn.Conv2d(embed_dim * 8, embed_dim * 3, 1, 1, 0)
        # 尝试加入额外的两层
        self.bn1 = nn.BatchNorm2d(embed_dim * 3)
        self.conv3 = nn.Conv2d(embed_dim * 3, embed_dim, 1, 1, 0)
        

        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        self.res = models.resnet50(pretrained=True)
        # 将模型设置为评估模式
        # self.res.eval()
        self.qa = RQABlock()
        # self.conv4 = nn.Conv2d(1000, 256, 1, 1, 0)
        self.conv_s1 = nn.Conv2d(1, 32, 1, 1, 0)
        self.rq_score = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, num_outputs),
            nn.ReLU()
        )
        
        self.fc_score1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, num_outputs),
            nn.ReLU()
        )
        self.fc_weight1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output, noise_levels=1):
        # 6789
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]        

        # 噪声等级信息的嵌入
        noise_levels = torch.tensor(noise_levels).cuda()  # 噪声等级，假设为四个样本的噪声等级
        # noise_embedded = self.noise_embedding(noise_levels)  # [B, hidden_dim]

        # 使用可供学习的参数
        noise_embedded = self.noise_levels[0, noise_levels]

        # 将噪声等级信息与特征进行连接
        # x6 = torch.cat((x6, noise_embedded.reshape(1, 1, -1).expand(x6.shape[0], 1, 768)), dim=1) 
        # x7 = torch.cat((x7, noise_embedded.reshape(1, 1, -1).expand(x6.shape[0], 1, 768)), dim=1)
        # x8 = torch.cat((x8, noise_embedded.reshape(1, 1, -1).expand(x6.shape[0], 1, 768)), dim=1)
        # x9 = torch.cat((x9, noise_embedded.reshape(1, 1, -1).expand(x6.shape[0], 1, 768)), dim=1)
        if noise_levels == 0:
            x = torch.cat((x6, x7, x8, x9), dim=2)
        if noise_levels == 1:
            x = torch.cat((x6, x7), dim=2)
        if noise_levels == 2:
            x = x8
        if noise_levels == 3:
            x = x9
        
        # print(x.shape, noise_embedded.shape)
        x = torch.cat((x, noise_embedded.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 784, -1)), dim=2)
        return x

    def forward(self, x, d0, d1, d2):
        # 针对增强结果的差进行，属于是恢复评分 Restore QA，在这个阶段不需要加入噪声等级的embedding
        
        # # 保存原始图片
        # f_maps = []
        # w_maps = []
        # s_maps = []
        # # Extract the first image (shape [3, 224, 224])
        # image = x
        # # Create output directory if it doesn't exist
        # os.makedirs('output_img', exist_ok=True)

        # # Save each channel as a separate image
        # for channel in range(image.shape[0]):
        #     plt.figure()
        #     plt.imshow(image[channel].detach().cpu().numpy().transpose(1, 2, 0))
        #     plt.axis('off')
        #     plt.savefig(f'output_img/channel_{channel + 1}.png', bbox_inches='tight', pad_inches=0.1)
        #     plt.close()
        # # Combine channels into a single image
        # combined_image = np.vstack([plt.imread(f'output_img/channel_{i + 1}.png') for i in range(image.shape[0])])
        # plt.imsave('output_img/combined_image.png', combined_image, cmap='gray')
        # print("Images saved successfully.")

            
        f = self.res(x)
        f0 = self.res(d0)
        f1 = self.res(d1)
        f2 = self.res(d2)
        s1 = self.qa(f,f0,f1,f2) # (4,784,1)
        
        s1 = rearrange(s1, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            s1 = tab(s1)
        s1 = rearrange(s1, 'b c (h w)  -> b c h w', h=self.input_size, w=self.input_size)
        s1 = self.conv_s1(s1)
        s1 = rearrange(s1, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        
        score_1 = torch.tensor([]).cuda()
        for i in range(s1.shape[0]):
            f = self.fc_score1(s1[i])
            w = self.fc_weight1(s1[i]) 
            _s = torch.sum(f * w) / torch.sum(w)
            score_1 = torch.cat((score_1, _s.unsqueeze(0)), 0)

        # 4211
        ####################################################################################################################################################
        # 注意！！！！现在在改动这一块的消融实验!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! self.vit(d1)这里
        ####################################################################################################################################################
        _x = self.vit(x)
        x_ori = self.extract_feature(self.save_output, noise_levels=0)
        self.save_output.outputs.clear()

        _x = self.vit(d0)
        x0 = self.extract_feature(self.save_output, noise_levels=1)
        self.save_output.outputs.clear()

        _x = self.vit(d1)
        x1 = self.extract_feature(self.save_output, noise_levels=2)
        self.save_output.outputs.clear()

        _x = self.vit(d1)
        x2 = self.extract_feature(self.save_output, noise_levels=3)
        self.save_output.outputs.clear()

        x = torch.cat((x_ori, x0, x1, x2), dim=2)
        # x_re = x_ori[:, :, :-768] - torch.cat((x0[:, :, :-768], x1[:, :, :-768], x2[:, :, :-768]), dim=2)
        # x_re = x_ori - torch.cat((x0, x1, x2), dim=2)

        # print(x.shape)
        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        # 额外的层
        # x = self.bn1(x) # 不加入BN层效果似乎更好一些
        x = self.conv3(x)
        
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)


        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])

            # # 尝试画出中间的patch maps
            # f_maps.append(f.detach().cpu().numpy())
            # w_maps.append(w.detach().cpu().numpy())
            # s = f * w / torch.sum(w)
            # s_maps.append(s.detach().cpu().numpy())
               
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
                    
        # # Convert lists to numpy arrays
        # f_maps = np.array(f_maps)
        # w_maps = np.array(w_maps)
        # s_maps = np.array(s_maps)
        # # Reshape for visualization
        # f_maps = f_maps.reshape(x.shape[0], self.input_size, self.input_size)
        # w_maps = w_maps.reshape(x.shape[0], self.input_size, self.input_size)
        # s_maps = s_maps.reshape(x.shape[0], self.input_size, self.input_size)
        # # Save the plots as PNG
        # plot_and_save_maps(f_maps, 'Feature Map (f)', 'f')
        # plot_and_save_maps(w_maps, 'Weight Map (w)', 'w')
        # plot_and_save_maps(s_maps, 'Final Map (s)', 's')
        # print("PNG images saved successfully.")
        

        
        # score = score + 5 * score_1.squeeze()
        score = score_1
        return score


def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


# Function to plot the maps and save as PNG
def plot_and_save_maps(maps, title, prefix):
    os.makedirs('output_img', exist_ok=True)  # Create output directory if it doesn't exist
    for i in range(maps.shape[0]):
        plt.figure()
        # plt.title(f"{title} {i+1}", fontsize=16)
        plt.imshow(maps[i], cmap='rainbow')
        plt.axis('off')
        # plt.colorbar()
        plt.savefig(f'output_img/{prefix}_map_{i+1}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    # Combine channels into a single image
    combined_image = np.vstack([plt.imread(f'output_img/{prefix}_map_{i+1}.png') for i in range(maps.shape[0])])
    plt.imsave(f'output_img/{prefix}_combined_image.png', combined_image, cmap='gray')