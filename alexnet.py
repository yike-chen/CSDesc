import torch
import torch.nn as nn
import torchvision
from torchvision  import transforms
from PIL import Image
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # 方法一
        # self.features = nn.Sequential()
        # self.features.add_module("conv1", nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        # self.features.add_module("relu1", nn.ReLU(inplace=True))
        # self.features.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=2))
        # self.features.add_module("conv2", nn.Conv2d(64, 192, kernel_size=5, padding=2))
        # self.features.add_module("relu2", nn.ReLU(inplace=True))
        # self.features.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2))
        # self.features.add_module("conv3", nn.Conv2d(192, 384, kernel_size=3, padding=1))
        # self.features.add_module("relu3", nn.ReLU(inplace=True))
        # self.features.add_module("conv4", nn.Conv2d(384, 256, kernel_size=3, padding=1))
        # self.features.add_module("relu4", nn.ReLU(inplace=True))
        # self.features.add_module("conv5", nn.Conv2d(256, 256, kernel_size=3, padding=1))
        # self.features.add_module("relu5", nn.ReLU(inplace=True))
        # self.features.add_module("pool5", nn.MaxPool2d(kernel_size=3, stride=2))

        # self.classifier = nn.Sequential()
        # self.classifier.add_module("drop6", nn.Dropout())
        # self.classifier.add_module("fc6", nn.Linear(256 * 6 * 6, 4096))
        # self.classifier.add_module("relu6", nn.ReLU(inplace=True))
        # self.classifier.add_module("drop7", nn.Dropout())
        # self.classifier.add_module("fc7", nn.Linear(4096, 4096))
        # self.classifier.add_module("relu7", nn.ReLU(inplace=True))
        # self.classifier.add_module("fc8", nn.Linear(4096, num_classes))

        # 方法二
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )


        # 方法三
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )

        # torch.nn.init.xavier_normal_(self.conv1[0].weight)
        # torch.nn.init.xavier_normal_(self.conv2[0].weight)
        # torch.nn.init.xavier_normal_(self.conv3[0].weight)
        # torch.nn.init.xavier_normal_(self.conv4[0].weight)
        # torch.nn.init.xavier_normal_(self.conv5[0].weight)

        # torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        # torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        # torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        # torch.nn.init.xavier_uniform_(self.conv4[0].weight)
        # torch.nn.init.xavier_uniform_(self.conv5[0].weight)

        torch.nn.init.kaiming_normal_(self.conv1[0].weight)
        torch.nn.init.kaiming_normal_(self.conv2[0].weight)
        torch.nn.init.kaiming_normal_(self.conv3[0].weight)
        torch.nn.init.kaiming_normal_(self.conv4[0].weight)
        torch.nn.init.kaiming_normal_(self.conv5[0].weight)

        # torch.nn.init.kaiming_uniform_(self.conv1[0].weight)
        # torch.nn.init.kaiming_uniform_(self.conv2[0].weight)
        # torch.nn.init.kaiming_uniform_(self.conv3[0].weight)
        # torch.nn.init.kaiming_uniform_(self.conv4[0].weight)
        # torch.nn.init.kaiming_uniform_(self.conv5[0].weight)

        # self.fc6 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True)
        # )
        # self.fc7 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        # )
        # self.fc8 = nn.Sequential(
        #     nn.Linear(4096, num_classes)
        # )

    def forward(self, x, query_num = -1):

        # x_list = []
        # if query_num != -1:

        #     x_query = x[0:query_num]
            
        #     x_list.append(torch.mean(x_query, 0, True))
        #     conv1_query_out = self.conv1(x)
        #     x_list.append(torch.mean(conv1_query_out, 0, True))

        #     conv2_query_out = self.conv2(conv1_query_out)
        #     x_list.append(torch.mean(conv2_query_out, 0, True))

        #     conv3_query_out = self.conv3(conv2_query_out)
        #     x_list.append(torch.mean(conv3_query_out, 0, True))

        #     conv4_query_out = self.conv4(conv3_query_out)
        #     x_list.append(torch.mean(conv4_query_out, 0, True))

        #     conv5_query_out = self.conv5(conv4_query_out)
        #     conv5_Relu_Pool_out = self.conv5_Relu_Pool(conv5_query_out)
        #     x_list.append(torch.mean(conv5_Relu_Pool_out, 0, True))

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

        # res = conv5_out.view(conv5_Relu_Pool_out.size(0), -1)
        # fc6_out = self.fc6(res)
        # fc7_out = self.fc7(fc6_out)
        # fc8_out = self.fc8(fc8_out)
        return conv5_out, x_list

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x



if __name__ == '__main__':

    # transform1 = transforms.Compose([
	#     # transforms.Resize((224,224)), # 只能对PIL图片进行裁剪
	#     transforms.ToTensor(),
	# ])

    # to_pil_image = transforms.ToPILImage()

    # img_path = "/home/gzz/Documents/1.jpg"
    # img_PIL = Image.open(img_path).convert("RGB")
    # img_PIL = img_PIL.resize([224,224])
    # img_PIL.show()
    # img_PIL_Tensor = transform1(img_PIL)


    # model = AlexNet()
    model = models.alexnet(pretrained=True)
    print(model)

    # input = torch.unsqueeze(img_PIL_Tensor,0)
    # # input = torch.randn(1,3,224,224)
    # x = input
    # for index, layer in enumerate(model.features):
    #     x = layer(x)

    # convxx = model.features[1]

    # xx = convxx(x)

    # image = to_pil_image(xx[0])
    # image = image.resize([224,224])
    # image.show()
    # out = model(input)
    # print(out.shape)