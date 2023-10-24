'''
Baseline unet: Test baseline without any operations
'''

import torch.nn as nn
import torch

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


n1 = 32
filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 3
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0)) # [1, 256, 32, 32]
        x4_0 = self.conv4_0(self.pool(x3_0)) # [1, 512, 16, 16]
        # print("x30 shape:", x3_0.shape)
        return [x0_0, x1_0, x2_0, x3_0, x4_0]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        out_ch = 2
        # self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[4], filters[4])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.conv1_3 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv0_4 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.Up4_0 = up(filters[4])
        self.Up3_1 = up(filters[3])
        self.Up2_2 = up(filters[2])
        self.Up1_3 = up(filters[1])
        self.conv_final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x_A):
        self.x0_0A, self.x1_0A, self.x2_0A, self.x3_0A, self.x4_0A = x_A
        # self.x0_0B, self.x1_0B, self.x2_0B, self.x3_0B, self.x4_0B = x_B
        x3_1 = self.conv3_1(torch.cat([self.x3_0A, self.Up4_0(self.x4_0A)], 1)) # [1,512,32,32]
        print("x31 shape:", x3_1.shape)
        # x2_2 = self.conv2_2(torch.cat([self.x2_0A, self.x2_0B, self.Up3_1(x3_1)], 1)) # [1, 128, 64, 64]
        # x1_3 = self.conv1_3(torch.cat([self.x1_0A, self.x1_0B, self.Up2_2(x2_2)], 1)) # [1, 64, 128, 128]
        # x0_4 = self.conv0_4(torch.cat([self.x0_0A, self.x0_0B, self.Up1_3(x1_3)], 1)) # [1, 64, 256, 128]
        # print("x04:", x0_4.shape)
        # out = self.conv_final(x0_4)
        return x3_1


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        torch.nn.Module.dump_patches = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        '''Encoder return multi feature map to nested
        '''
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, batch):
        encode = self.encoder(batch)
        output = self.decoder.forward(encode)

        return output
#         x0_B = self.encoder(xB)
#         out = self.decoder.forward(x0_A, x0_B)
#         return (out, ), [out]
#
# if __name__ == "__main__":
#     num_classes = [2,2]
#     current_task = 1
#     unet = UNet()
#     image_A = torch.randn((4, 3, 256, 256))
#     image_B = torch.randn((4, 3, 256, 256))
#     for name, m in unet.named_parameters():
#         print(name)
#     outputs_change, feature_map = unet(image_A, image_B, current_task)
