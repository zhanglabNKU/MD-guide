import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import fusion_strategy
from args_fusion import args

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder

        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # self.conv2 = ConvLayer(input_nc2, nb_filter[0], kernel_size, stride)
        # self.DB2 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)##concate之后为128通道
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input_s1,input_s2):
        x1 = self.conv1(input_s1)
        x_DB = self.DB1(x1)
        x2 = self.conv1(input_s2)
        x_DB2 = self.DB1(x2)
        # db_fusion = torch.cat((x_DB, x_DB2),dim = 1) ##concate在一起
        db_fusion = torch.max(x_DB, x_DB2)
        return [db_fusion]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = (en1[0] + en2[0])/2
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]


# DenseFuse network
class MedFuse_net(nn.Module):
    def __init__(self, pretrained_dict,input_nc=1, output_nc=1):
        super(MedFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)  ##concate之后为128通道
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        ## encoder
        # encoder
        # self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        # self.DB1 = denseblock(nb_filter[0], kernel_size, stride)


        # Initialize conv1 with the pretrained model and freeze its parameters
        for p in pretrained_dict.parameters():
            p.requires_grad = False
        self.conv1 = pretrained_dict.conv1
        self.conv1.stride = stride
        self.conv1.padding = (0, 0)
        #
        self.DB1 = pretrained_dict.DB1
        self.DB1.stride = stride
        self.DB1.padding = (0, 0)
    # self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
    # self.DB1 = denseblock(nb_filter[0], kernel_size, stride)


    def encoder(self, input_s1, input_s2):
        x1 = self.conv1(input_s1)
        x_DB = self.DB1(x1)
        x2 = self.conv1(input_s2)
        x_DB2 = self.DB1(x2)
        # db_fusion = torch.cat((x_DB, x_DB2),dim = 1) ##concate在一起
        db_fusion = torch.max(x_DB, x_DB2)
        return [db_fusion]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = (en1[0] + en2[0]) / 2
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]

def medtrain(input_nc, output_nc):
    # pre-trained model
    if args.resume is not None:
        print('pretrained model using weight from {}.'.format(args.resume))  ##恢复与初始化权重
        pretrained_model = DenseFuse_net(input_nc, output_nc)
        pretrained_model.load_state_dict(torch.load(args.resume))
        # pretrained_dict = torch.load(args.resume)
        # print(pretrained_dict.items())

    # our model
    net = MedFuse_net(pretrained_model,input_nc, output_nc)
    model_dict = net.state_dict()
    # print(model_dict.items())
    #
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)  # 用预训练模型参数更新new_model中的部分参数
    # net.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中
    #
    # ##### 冻结部分参数
    # for param in net.parameters():
    #     param.requires_grad = False  # 设置所有参数不可导，下面选择设置可导的参数
    for param in net.conv1.parameters():
        print("param: ",param.requires_grad)
    # for param in net.DB1.parameters():
    #     param.requires_grad = True

    return net
