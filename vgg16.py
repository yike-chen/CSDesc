import torch
import torch.nn as nn
import torchvision
from torchvision  import transforms
from PIL import Image
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        torch.nn.init.xavier_normal_(self.conv1[0].weight)
        torch.nn.init.xavier_normal_(self.conv2[0].weight)
        torch.nn.init.xavier_normal_(self.conv3[0].weight)
        torch.nn.init.xavier_normal_(self.conv4[0].weight)
        torch.nn.init.xavier_normal_(self.conv5[0].weight)
        torch.nn.init.xavier_normal_(self.conv6[0].weight)
        torch.nn.init.xavier_normal_(self.conv7[0].weight)
        torch.nn.init.xavier_normal_(self.conv8[0].weight)
        torch.nn.init.xavier_normal_(self.conv9[0].weight)
        torch.nn.init.xavier_normal_(self.conv10[0].weight)
        torch.nn.init.xavier_normal_(self.conv11[0].weight)
        torch.nn.init.xavier_normal_(self.conv12[0].weight)
        torch.nn.init.xavier_normal_(self.conv13[0].weight)
        # print(self.conv1[0].weight)
        # print('cccc')

    
    def forward(self, x, query_num = -1):

        x_list = []

        x_list.append(torch.mean(x, 0, True))
        
        conv1_out = self.conv1(x)
        x_list.append(torch.mean(conv1_out, 0, True))

        conv2_out = self.conv2(conv1_out)
        x_list.append(torch.mean(conv2_out, 0, True))

        conv3_out = self.conv3(conv2_out)
        x_list.append(torch.mean(conv3_out, 0, True))

        conv4_out = self.conv4(conv3_out)
        x_list.append(torch.mean(conv4_out, 0, True))

        conv5_out = self.conv5(conv4_out)
        x_list.append(torch.mean(conv5_out, 0, True))

        conv6_out = self.conv6(conv5_out)
        x_list.append(torch.mean(conv6_out, 0, True))

        conv7_out = self.conv7(conv6_out)
        x_list.append(torch.mean(conv7_out, 0, True))

        conv8_out = self.conv8(conv7_out)
        x_list.append(torch.mean(conv8_out, 0, True))

        conv9_out = self.conv9(conv8_out)
        x_list.append(torch.mean(conv9_out, 0, True))

        conv10_out = self.conv10(conv9_out)
        x_list.append(torch.mean(conv10_out, 0, True))

        conv11_out = self.conv11(conv10_out)
        x_list.append(torch.mean(conv11_out, 0, True))

        conv12_out = self.conv12(conv11_out)
        x_list.append(torch.mean(conv12_out, 0, True))

        conv13_out = self.conv13(conv12_out)

        return conv13_out, x_list



