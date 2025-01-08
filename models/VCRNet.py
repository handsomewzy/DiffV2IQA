import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class VCRNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(VCRNet, self).__init__()

        center_offset_from_origin_border = padding - kernel_size // 2
        ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)

        if center_offset_from_origin_border >= 0:
            self.ver_conv_crop_layer = nn.Sequential()
            ver_conv_padding = ver_pad_or_crop
            self.hor_conv_crop_layer = nn.Sequential()
            hor_conv_padding = hor_pad_or_crop

        else:
            self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
            ver_conv_padding = (0, 0)
            self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
            hor_conv_padding = (0, 0)

        self.relu = nn.PReLU()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv1_3 = nn.Conv2d(16, 16, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv2_3 = nn.Conv2d(16, 16, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.resconv2 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=2, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=2, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv4_3 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        self.resconv4 = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=2, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=2, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv6_3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn6 = nn.BatchNorm2d(64)

        self.resconv6 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=2, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv7_3 = nn.Conv2d(128, 256, kernel_size=(1, 3), stride=2, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn8 = nn.BatchNorm2d(256)

        self.resconv8 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=(3, 1), stride=2, padding=ver_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.conv9_3 = nn.Conv2d(256, 512, kernel_size=(1, 3), stride=2, padding=hor_conv_padding, dilation=1, groups=1,
                                 bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10_2 = nn.Conv2d(512, 512, kernel_size=(3, 1), stride=1, padding=ver_conv_padding, dilation=1,
                                  groups=1, bias=False)
        self.conv10_3 = nn.Conv2d(512, 512, kernel_size=(1, 3), stride=1, padding=hor_conv_padding, dilation=1,
                                  groups=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_bn4 = nn.BatchNorm2d(32)  # FRN(num_features=32)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.GP_conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        self.GP_pool1 = nn.MaxPool2d((2, 2))
        self.GP_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.GP_pool2 = nn.MaxPool2d((4, 4))
        self.GP_conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # self.fc1 = nn.Linear(11520, 1)
    def forward(self, x):

        img = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        GP0 = self.GP_conv0(img)
        # print("GP0", GP0.size())  #[16, 16, 224, 224]

        GP1 = self.GP_pool1(img)
        # print("GP1", GP1.size())  #[16, 3, 112, 112]
        GP1 = self.GP_conv1(GP1)
        # print("GP1", GP1.size())  #[16, 32, 112, 112]

        GP2 = self.GP_pool2(img)
        # print("GP2", GP2.size())  #[16, 3, 56, 56]
        GP2 = self.GP_conv2(GP2)
        # print("GP2", GP2.size())  #[16, 64, 56, 56]

        ####################################################################################
        conv = self.conv(img)
        # print('conv', conv.size())  #[16, 16, 224, 224]


        conv3x3_1 = self.conv1(conv)
        # print('conv3x3_1', conv3x3_1.size())  #[16, 16, 224, 224]
        conv3x1_1 = self.conv1_2(conv)
        # print('conv3x1_1', conv3x1_1.size())  #[16, 16, 224, 224]
        conv1x3_1 = self.conv1_3(conv)
        # print('conv1x3_1', conv1x3_1.size())  #[16, 16, 224, 224]
        conv1 = self.relu(self.bn1(conv3x3_1 + conv3x1_1 + conv1x3_1))
        # print('conv1', conv1.size())  # [16, 16, 224, 224]

        conv3x3_1_2 = self.conv2(conv1)
        # print('conv3x3_1_2', conv3x3_1_2.size())#conv3x3_1_2 torch.Size([128, 16, 64, 64])
        conv3x1_1_2 = self.conv2_2(conv1)
        # print('conv3x1_1_2', conv3x1_1_2.size())#conv3x1_1_2 torch.Size([128, 16, 64, 64])
        conv1x3_1_2 = self.conv2_3(conv1)
        # print('conv1x3_1_2', conv1x3_1_2.size())#conv1x3_1_2 torch.Size([128, 16, 64, 64])
        conv2 = self.bn2(conv3x3_1_2 + conv3x1_1_2 + conv1x3_1_2)
        # print('conv2', conv2.size())#conv2 torch.Size([128, 16, 64, 64])

        res1 = self.relu(conv + conv2)
        # print('res1', res1.size())#res1 torch.Size([128, 16, 64, 64])
        res1 = torch.cat((GP0, res1), 1)
        # print('res1', res1.size())#[1, 32, 224, 224]

        ######################################################################################
        conv3x3_2 = self.conv3(res1)
        # print('conv3x3_2',conv3x3_2.size())#conv3x3_2 torch.Size([128, 32, 32, 32])
        conv3x1_2 = self.conv3_2(res1)
        # print('conv3x1_2', conv3x1_2.size())#conv3x1_2 torch.Size([128, 32, 32, 32])
        conv1x3_2 = self.conv3_3(res1)
        # print('conv1x3_2', conv1x3_2.size())#conv1x3_2 torch.Size([128, 32, 32, 32])
        conv3 = self.relu(self.bn3(conv3x3_2 + conv3x1_2 + conv1x3_2))
        # print('conv3', conv3.size())#conv3 torch.Size([128, 32, 32, 32])

        conv3x3_2_2 = self.conv4(conv3)
        # print('conv3x3_2_2', conv3x3_2_2.size())#conv3x3_2_2 torch.Size([128, 32, 32, 32])
        conv3x1_2_2 = self.conv4_2(conv3)
        # print('conv3x1_2_2', conv3x1_2_2.size())#conv3x1_2_2 torch.Size([128, 32, 32, 32])
        conv1x3_2_2 = self.conv4_3(conv3)
        # print('conv1x3_2_2', conv1x3_2_2.size())#conv1x3_2_2 torch.Size([128, 32, 32, 32])
        conv4 = self.bn4(conv3x3_2_2 + conv3x1_2_2 + conv1x3_2_2)
        # print('conv4', conv4.size())#conv4 torch.Size([128, 32, 32, 32])
        resconv2 = self.resconv2(res1)
        # print('resconv2', resconv2.size())#[128, 32, 32, 32]
        res2 = self.relu(resconv2 + conv4)
        # print('res2', res2.size())#res2 torch.Size([128, 32, 32, 32])
        res2 = torch.cat((GP1, res2), 1)
        # print('res2', res2.size())#[1, 64, 112, 112]

        ############################################################################################
        conv3x3_3 = self.conv5(res2)
        # print("conv3x3_3", conv3x3_3.size())#conv3x3_3 torch.Size([128, 64, 16, 16])
        conv3x1_3 = self.conv5_2(res2)
        # print("conv3x1_3", conv3x1_3.size())#conv3x1_3 torch.Size([128, 64, 16, 16])
        conv1x3_3 = self.conv5_3(res2)
        # print("conv1x3_3", conv1x3_3.size())#conv1x3_3 torch.Size([128, 64, 16, 16])
        conv5 = self.relu(self.bn5(conv3x3_3 + conv3x1_3 + conv1x3_3))
        # print("conv5", conv5.size())#conv5 torch.Size([128, 64, 16, 16])

        conv3x3_3_2 = self.conv6(conv5)
        # print("conv3x3_3_2", conv3x3_3_2.size())#conv3x3_3_2 torch.Size([128, 64, 16, 16])
        conv3x1_3_2 = self.conv6_2(conv5)
        # print("conv3x1_3_2", conv3x1_3_2.size())#conv3x1_3_2 torch.Size([128, 64, 16, 16])
        conv1x3_3_2 = self.conv6_3(conv5)
        # print("conv1x3_3_2", conv1x3_3_2.size())#conv1x3_3_2 torch.Size([128, 64, 16, 16])
        conv6 = self.bn6(conv3x3_3_2 + conv3x1_3_2 + conv1x3_3_2)
        # print("conv6", conv6.size())#conv6 torch.Size([128, 64, 16, 16])

        resconv4 = self.resconv4(res2)
        # print("resconv4", resconv4.size())#[128, 64, 16, 16]
        res3 = self.relu(resconv4 + conv6)
        res3 = torch.cat((GP2, res3), 1)
        # print("res3", res3.size())#[1, 128, 56, 56]

        ############################################################################################
        conv3x3_4 = self.conv7(res3)
        # print("conv3x3_4", conv3x3_4.size())#conv3x3_4 torch.Size([128, 256, 8, 8])
        conv3x1_4 = self.conv7_2(res3)
        # print("conv3x1_4", conv3x1_4.size())#conv3x1_4 torch.Size([128, 256, 8, 8])
        conv1x3_4 = self.conv7_3(res3)
        # print("conv1x3_4", conv1x3_4.size())#conv1x3_4 torch.Size([128, 256, 8, 8])
        conv7 = self.relu(self.bn7(conv3x3_4 + conv3x1_4 + conv1x3_4))
        # print("conv7", conv7.size())#conv7 torch.Size([128, 256, 8, 8])

        conv3x3_4_2 = self.conv8(conv7)
        # print("conv3x3_4_2", conv3x3_4_2.size())#conv3x3_4_2 torch.Size([128, 256, 8, 8])
        conv3x1_4_2 = self.conv8_2(conv7)
        # print("conv3x1_4_2", conv3x1_4_2.size())#conv3x1_4_2 torch.Size([128, 256, 8, 8])
        conv1x3_4_2 = self.conv8_3(conv7)
        # print("conv1x3_4_2", conv1x3_4_2.size())#conv1x3_4_2 torch.Size([128, 256, 8, 8])
        conv8 = self.bn8(conv3x3_4_2 + conv3x1_4_2 + conv1x3_4_2)
        # print("conv8", conv8.size())#conv8 torch.Size([128, 256, 8, 8])
        resconv6 = self.resconv6(res3)
        # print("resconv6", resconv6.size())#[128, 256, 8, 8]
        res4 = self.relu(resconv6 + conv8)
        # print("res4", res4.size())#[1, 256, 28, 28]

        ################################################################################################
        conv3x3_5 = self.conv9(res4)
        # print("conv3x3_5", conv3x3_5.size())#conv3x3_5 torch.Size([128, 512, 4, 4])
        conv3x1_5 = self.conv9_2(res4)
        # print("conv3x1_5", conv3x1_5.size())#conv3x1_5 torch.Size([128, 512, 4, 4])
        conv1x3_5 = self.conv9_3(res4)
        # print("conv1x3_5", conv1x3_5.size())#conv1x3_5 torch.Size([128, 512, 4, 4])
        conv9 = self.relu(self.bn9(conv3x3_5 + conv3x1_5 + conv1x3_5))
        # print("conv9", conv9.size())#conv9 torch.Size([128, 512, 4, 4])

        conv3x3_5_2 = self.conv10(conv9)
        # print("conv3x3_5_2", conv3x3_5_2.size())#conv3x3_5_2 torch.Size([128, 512, 4, 4])
        conv3x1_5_2 = self.conv10_2(conv9)
        # print("conv3x1_5_2", conv3x1_5_2.size())#conv3x1_5_2 torch.Size([128, 512, 4, 4])
        conv1x3_5_2 = self.conv10_3(conv9)
        # print("conv1x3_5_2", conv1x3_5_2.size())#conv1x3_5_2 torch.Size([128, 512, 4, 4])
        conv10 = self.bn10(conv3x3_5_2 + conv3x1_5_2 + conv1x3_5_2)
        resconv8 = self.resconv8(res4)
        # print("resconv8", resconv8.size())#[128, 512, 4, 4]
        res5 = self.relu(resconv8 + conv10)
        # print("res5", res5.size())#[1, 512, 14, 14]

        ###########################################################################################
        deconv1 = self.relu(self.deconv_bn1(self.deconv1(res5)))
        concatenate1 = torch.cat((deconv1, res4), 1)
        # print("concatenate1", concatenate1.size())#[1, 512, 28, 28]

        deconv2 = self.relu(self.deconv_bn2(self.deconv2(concatenate1)))
        concatenate2 = torch.cat((deconv2, res3), 1)
        # print("concatenate2", concatenate2.size())#[1, 256, 56, 56]

        deconv3 = self.relu(self.deconv_bn3(self.deconv3(concatenate2)))
        concatenate3 = torch.cat((deconv3, res2), 1)
        # print("concatenate3", concatenate3.size())#[1, 128, 112, 112]

        deconv4 = self.relu(self.deconv_bn4(self.deconv4(concatenate3)))
        concatenate4 = torch.cat((deconv4, res1), 1)
        # print("concatenate4", concatenate4.size())#[1, 64, 224, 224]

        restore_output = self.deconv5(concatenate4)
        # print("restore_output", restore_output.size())  # [16, 3, 224, 224]

        # return self.GP_pool2(concatenate1), self.GP_pool2(concatenate2), self.GP_pool2(concatenate3), self.GP_pool2(
        #     concatenate4)

        return concatenate1, concatenate2, concatenate3, concatenate4

    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class inception_module1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(inception_module1, self).__init__()
        # self.cbam = CBAM(gate_channels=outchannel)
        self.conv1_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=2, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(outchannel * 3)

    def forward(self, x):
        x1 = self.conv1_2(x)
        # print('x1', x1.size())#[16, 24, 28, 28]
        x2 = self.conv3_1(x)
        # print('x2', x2.size())#[16, 24, 28, 28]
        x3 = self.conv3_2(x)
        # print('x3', x3.size())#[16, 24, 28, 28]
        x3 = self.conv3_3(x3)
        # print('x3', x3.size())#[16, 24, 28, 28]
        x = torch.cat((x1, x2, x3), 1)
        # print('x', x.size())#
        x = self.prelu(self.bn(x))
        return x

class inception_module2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(inception_module2, self).__init__()
        # self.cbam = CBAM(gate_channels=outchannel)
        self.conv1_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=8, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=8, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=8, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(outchannel * 3)

    def forward(self, x):
        x1 = self.conv1_2(x)
        # print('x1', x1.size())#[16, 24, 28, 28]
        x2 = self.conv3_1(x)
        # print('x2', x2.size())#[16, 24, 28, 28]

        x3 = self.conv3_2(x)
        # print('x3', x3.size())#[16, 24, 28, 28]
        x3 = self.conv3_3(x3)
        # print('x3', x3.size())#[16, 24, 28, 28]
        x = torch.cat((x1, x2, x3), 1)
        # print('x', x.size())#
        x = self.prelu(self.bn(x))
        return x


class demoIQA(nn.Module):
    def __init__(self):
        super(demoIQA, self).__init__()

        self.efficient = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficient.eval()

        self.VCR = VCRNet().cuda()
        self.VCR.load_state_dict(torch.load('/home/vipsl416-10-wangzhaoyang/IQA-pytorch/VCRNet.pth'))
        self.VCR.eval()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.prelu = nn.PReLU()

        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        self.inception1_1 = inception_module1(inchannel=24, outchannel=24)
        self.inception1_11 = inception_module2(inchannel=64, outchannel=24)

        self.inception2_1 = inception_module1(inchannel=40, outchannel=24)
        self.inception2_11 = inception_module2(inchannel=128, outchannel=24)

        self.inception3_1 = inception_module1(inchannel=112, outchannel=24)
        self.inception3_11 = inception_module2(inchannel=256, outchannel=24)

        self.inception4_1 = inception_module1(inchannel=1280, outchannel=24)
        self.inception4_11 = inception_module2(inchannel=512, outchannel=24)

    def forward(self, input):

        input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
        endpoints = self.efficient.extract_endpoints(input)
        layer1 = endpoints['reduction_2']
        layer2 = endpoints['reduction_3']
        layer3 = endpoints['reduction_4']
        layer4 = endpoints['reduction_6']
        # print('layer1', layer1.size())#[128, 24, 56, 56]
        # print('layer2', layer2.size())#[128, 40, 28, 28]
        # print('layer3', layer3.size())#[128, 112, 14, 14]
        # print('layer4', layer4.size())#[128, 1280, 7, 7]

        d1, d2, d3, d4 = self.VCR(input)
        # print('d1', d1.size())#[16, 512, 28, 28]
        # print('d2', d2.size())#[16, 256, 56, 56]
        # print('d3', d3.size())#[16, 128, 112, 112]
        # print('d4', d4.size())#[16, 64, 224, 224]

        x1 = self.inception1_1(layer1)
        # print('x1', x1.size())#[16, 72, 28, 28]
        l1 = self.inception1_11(d4)
        # print('l1', l1.size())#[16, 72, 28, 28]
        c1 = torch.cat((x1, l1), 1)
        # print('c1', c1.size())#[16, 144, 28, 28]
        c1 = self.gap(c1)

        x2 = self.inception2_1(layer2)
        # print('x2', x2.size())#
        l2 = self.inception2_11(d3)
        # print('l2', l2.size())#[1, 72, 14, 14]
        c2 = torch.cat((x2, l2), 1)
        # print('c2', c2.size())#[16, 144, 14, 14]
        c2 = self.gap(c2)

        x3 = self.inception3_1(layer3)
        l3 = self.inception3_11(d2)
        # print('l3', l3.size())#[1, 72, 7, 7]
        c3 = torch.cat((x3, l3), 1)
        # print('c3', c3.size())#[16, 576, 7, 7]
        c3 = self.gap(c3)

        x4 = self.inception4_1(layer4)
        # print('x4', x4.size())#[1, 72, 4, 4]
        l4 = self.inception4_11(d1)
        # print('l4', l4.size())#[1, 72, 4, 4]
        c4 = torch.cat((x4, l4), 1)
        # print('c4', c4.size())#[16, 144, 4, 4]
        c4 = self.gap(c4)

        full1 = torch.cat((c1, c2, c3, c4), 1)
        # print('full1', full1.size())#[128, 288, 1, 1]

        full1 = full1.squeeze(3).squeeze(2)
        q = self.prelu(self.fc1(full1))
        q = F.dropout(q)
        q = self.prelu(self.fc2(q))
        q = self.fc3(q)

        return q