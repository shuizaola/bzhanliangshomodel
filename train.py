#练习pytorch教程小土堆
#蔡鹏
import time

import torch
import torchvision
from torch import nn
# 准备数据集
#from torch.nn.Modules import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from model import *
#定义一个设备
device = torch.device("cuda")
train_data = torchvision.datasets.CIFAR10(root="data",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)


#length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
#如果train_data
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用DataLoader 来加载数据


train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=61)


#创建网络模型

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x = self.model(x)
        return x
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()
#创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_setp = 0
#记录测试的次数
total_test_setp = 0
#训练次数
epoch = 30

#添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("----第{}轮训练开始----".format(i+1))


    #训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = tudui(imgs)
        Loss = loss_fn(output,targets)

        #优化器优化模型
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        total_train_setp = total_train_setp + 1
        if total_train_setp % 100 == 0:
            end_time = time.time()
            print("进行时间为：",end_time-start_time)
            print("训练次数：{},Loss:{}".format(total_train_setp,Loss.item()))
            writer.add_scalar("train_loss",Loss.item(),total_train_setp)


    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = tudui(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy +accuracy


    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))

    writer.add_scalar("test_loss",total_test_loss,total_test_setp)
    writer.add_scalar("test+accurary",total_accuracy/test_data_size)
    total_test_setp = total_test_setp + 1

    torch.save(tudui,"tudui_{}.pth".format(i))
    print("模型已经保存")

writer.close()
