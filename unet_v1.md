## Loading data

`

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

```python
DATA_DIR = 'camvid'

x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'valid_images')
y_valid_dir = os.path.join(DATA_DIR, 'valid_labels')

x_test_dir = os.path.join(DATA_DIR, 'test_images')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

```

```python
# helper function for data visualization
# 定义绘图函数
def visualize(**images):
    # 获取要绘制的图像数量
    n = len(images)
    # 设置绘图窗口大小
    plt.figure(figsize=(16, 5))
    # 遍历所有要绘制的图像
    for i, (name, image) in enumerate(images.items()):
        # 在当前行的第i+1个位置上绘制图像
        plt.subplot(1, n, i + 1)
        # 隐藏坐标轴
        plt.xticks([])
        plt.yticks([])
        # 设置子图标题
        plt.title(' '.join(name.split('_')).title())
        # 绘制图像
        plt.imshow(image)
    # 显示所有子图
    plt.show()
```

```python
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
```

```python
# 定义数据集类
class Dataset(BaseDataset):
    def __init__(
            self, 
            images_dir, # 存放图像的文件夹路径
            masks_dir, # 存放掩码的文件夹路径
            augmentation=None, # 图像增强函数
    ):
        # 获取数据集中的所有图像文件名
        self.ids = os.listdir(images_dir)
        # 构造图像文件路径列表
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # 构造掩码文件路径列表
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # 存储图像增强函数
        self.augmentation = augmentation

    def __getitem__(self, i):
        # 读取图像和掩码数据
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # 对掩码进行预处理，提取sky部分的掩码
        mask = (mask==17)
        mask = mask.astype('float')   

        # 应用图像增强函数
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 返回增强后的图像和掩码
        return image, mask.reshape(1,320,320)

    def __len__(self):
        # 返回数据集大小（即图像数量）
        return len(self.ids)
```

### Augmentations

```python
import albumentations as albu
```

```python
# 定义训练集的图像增强函数
def get_training_augmentation():
    train_transform = [
        # 随机水平翻转图像
        albu.HorizontalFlip(p=0.5),
        # 将图像大小调整为320x320像素
        albu.Resize(height=320, width=320, always_apply=True),
        # 随机缩放、旋转和平移图像
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, shift_limit=0.1, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)

# 定义测试集的图像增强函数
def get_test_augmentation():
    train_transform = [
        # 将图像大小调整为320x320像素
        albu.Resize(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)

# 创建数据集对象，并应用训练集的图像增强函数
augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
)


# same image with different random transforms
for i in range(1):
    image, mask = augmented_dataset[1]
    print(np.min(image),np.max(image))
    visualize(image=image, mask=mask[0,:,:])
```

    0 255

![png](output_11_1.png)

## Create model and train

```python
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
```

```python
#这段代码是一个基于U-Net的卷积神经网络模型，用于图像分割任务。其中包含了三个类：DoubleConv、Down和Up。
#DoubleConv类定义了一个双卷积层，包括两个卷积层、Batch Normalization和ReLU激活函数。
#Down类定义了一个下采样层，包括一个最大池化层和一个双卷积层。
#Up类定义了一个上采样层，包括一个上采样操作和一个双卷积层。
#其中，上采样操作可以选择使用双线性插值或转置卷积来实现。
#在forward函数中，输入x1和x2是两个特征图，x1经过上采样操作后与x2进行拼接，并进行双卷积操作。
```

```python
//解释代码 
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

```python
#这是一个PyTorch实现的U-Net结构，通常用于图像分割任务。
#U-Net结构是一个完全卷积神经网络，由编码器路径和解码器路径组成，
#其中编码器和解码器之间有相应级别的跳跃连接。
#编码器路径用于捕获图像的上下文信息，而解码器路径用于恢复编码过程中丢失的空间信息。
#该结构包含下采样层（Down），用于降低特征图的空间维度，以及上采样层（Up），用于增加特征图的空间维度。
#DoubleConv模块由两个3x3卷积层和一个ReLU激活函数组成。
#OutConv模块是一个最终的卷积层，用于产生输出的logits，sigmoid函数应用于logits以获得最终输出。
```

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.out  = torch.sigmoid
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.out(logits)
        return logits
```

```python
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.out  = torch.sigmoid
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.out(logits)
        return logits

# 可视化模型结构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
summary(model, input_size=(3, 320, 320))

```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 32, 320, 320]             896
           BatchNorm2d-2         [-1, 32, 320, 320]              64
                  ReLU-3         [-1, 32, 320, 320]               0
                Conv2d-4         [-1, 32, 320, 320]           9,248
           BatchNorm2d-5         [-1, 32, 320, 320]              64
                  ReLU-6         [-1, 32, 320, 320]               0
            DoubleConv-7         [-1, 32, 320, 320]               0
             MaxPool2d-8         [-1, 32, 160, 160]               0
                Conv2d-9         [-1, 64, 160, 160]          18,496
          BatchNorm2d-10         [-1, 64, 160, 160]             128
                 ReLU-11         [-1, 64, 160, 160]               0
               Conv2d-12         [-1, 64, 160, 160]          36,928
          BatchNorm2d-13         [-1, 64, 160, 160]             128
                 ReLU-14         [-1, 64, 160, 160]               0
           DoubleConv-15         [-1, 64, 160, 160]               0
                 Down-16         [-1, 64, 160, 160]               0
            MaxPool2d-17           [-1, 64, 80, 80]               0
               Conv2d-18          [-1, 128, 80, 80]          73,856
          BatchNorm2d-19          [-1, 128, 80, 80]             256
                 ReLU-20          [-1, 128, 80, 80]               0
               Conv2d-21          [-1, 128, 80, 80]         147,584
          BatchNorm2d-22          [-1, 128, 80, 80]             256
                 ReLU-23          [-1, 128, 80, 80]               0
           DoubleConv-24          [-1, 128, 80, 80]               0
                 Down-25          [-1, 128, 80, 80]               0
            MaxPool2d-26          [-1, 128, 40, 40]               0
               Conv2d-27          [-1, 256, 40, 40]         295,168
          BatchNorm2d-28          [-1, 256, 40, 40]             512
                 ReLU-29          [-1, 256, 40, 40]               0
               Conv2d-30          [-1, 256, 40, 40]         590,080
          BatchNorm2d-31          [-1, 256, 40, 40]             512
                 ReLU-32          [-1, 256, 40, 40]               0
           DoubleConv-33          [-1, 256, 40, 40]               0
                 Down-34          [-1, 256, 40, 40]               0
            MaxPool2d-35          [-1, 256, 20, 20]               0
               Conv2d-36          [-1, 256, 20, 20]         590,080
          BatchNorm2d-37          [-1, 256, 20, 20]             512
                 ReLU-38          [-1, 256, 20, 20]               0
               Conv2d-39          [-1, 256, 20, 20]         590,080
          BatchNorm2d-40          [-1, 256, 20, 20]             512
                 ReLU-41          [-1, 256, 20, 20]               0
           DoubleConv-42          [-1, 256, 20, 20]               0
                 Down-43          [-1, 256, 20, 20]               0
             Upsample-44          [-1, 256, 40, 40]               0
               Conv2d-45          [-1, 128, 40, 40]         589,952
          BatchNorm2d-46          [-1, 128, 40, 40]             256
                 ReLU-47          [-1, 128, 40, 40]               0
               Conv2d-48          [-1, 128, 40, 40]         147,584
          BatchNorm2d-49          [-1, 128, 40, 40]             256
                 ReLU-50          [-1, 128, 40, 40]               0
           DoubleConv-51          [-1, 128, 40, 40]               0
                   Up-52          [-1, 128, 40, 40]               0
             Upsample-53          [-1, 128, 80, 80]               0
               Conv2d-54           [-1, 64, 80, 80]         147,520
          BatchNorm2d-55           [-1, 64, 80, 80]             128
                 ReLU-56           [-1, 64, 80, 80]               0
               Conv2d-57           [-1, 64, 80, 80]          36,928
          BatchNorm2d-58           [-1, 64, 80, 80]             128
                 ReLU-59           [-1, 64, 80, 80]               0
           DoubleConv-60           [-1, 64, 80, 80]               0
                   Up-61           [-1, 64, 80, 80]               0
             Upsample-62         [-1, 64, 160, 160]               0
               Conv2d-63         [-1, 32, 160, 160]          36,896
          BatchNorm2d-64         [-1, 32, 160, 160]              64
                 ReLU-65         [-1, 32, 160, 160]               0
               Conv2d-66         [-1, 32, 160, 160]           9,248
          BatchNorm2d-67         [-1, 32, 160, 160]              64
                 ReLU-68         [-1, 32, 160, 160]               0
           DoubleConv-69         [-1, 32, 160, 160]               0
                   Up-70         [-1, 32, 160, 160]               0
             Upsample-71         [-1, 32, 320, 320]               0
               Conv2d-72         [-1, 32, 320, 320]          18,464
          BatchNorm2d-73         [-1, 32, 320, 320]              64
                 ReLU-74         [-1, 32, 320, 320]               0
               Conv2d-75         [-1, 32, 320, 320]           9,248
          BatchNorm2d-76         [-1, 32, 320, 320]              64
                 ReLU-77         [-1, 32, 320, 320]               0
           DoubleConv-78         [-1, 32, 320, 320]               0
                   Up-79         [-1, 32, 320, 320]               0
               Conv2d-80          [-1, 1, 320, 320]              33
              OutConv-81          [-1, 1, 320, 320]               0
    ================================================================
    Total params: 3,352,257
    Trainable params: 3,352,257
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 1.17
    Forward/backward pass size (MB): 703.91
    Params size (MB): 12.79
    Estimated Total Size (MB): 717.87
    ----------------------------------------------------------------

```python
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
)

# valid_dataset = Dataset(
#     x_valid_dir, 
#     y_valid_dir, 
#     augmentation=augmentation, 
# )

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
```

```python
net = UNet(n_channels=3, n_classes=1)
```

```python
image, mask = train_dataset[1]
with torch.no_grad():
    net.to('cuda')
    image = image/255.
    image = image.astype('float32')
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    image = image.to()
    print(image.shape)
    
    #pred = torch.argmax(pred, dim=1).squeeze().numpy()
    pred = net(image.unsqueeze(0).cuda())
    pred = pred.cpu()

plt.figure('mask')
print(mask.shape)
print(np.min(mask),np.max(mask))
plt.imshow(mask[0,:,:])
plt.show()

plt.figure('pred')
# pred = pred.view(720,960)
print(pred.shape)
print(np.min(pred.numpy()),np.max(pred.numpy()))
plt.imshow(pred[0,0,:,:])
plt.show()
```

    torch.Size([3, 320, 320])
    (1, 320, 320)
    0.0 1.0

![png](output_21_1.png)

    torch.Size([1, 1, 320, 320])
    0.078995936 0.87089837

![png](output_21_3.png)

```python
#添加loss曲线记录
```

```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
net.cuda()

optimizer = optim.RMSprop(net.parameters(), lr=0.003, weight_decay=1e-8)
criterion = nn.BCELoss()
# 创建空列表
loss_list = []

for epoch in range(1):
    
    net.train()
    epoch_loss = 0
    
    for data in train_loader:
        
        images,labels = data
        images = images.permute(0,3,1,2)
        images = images/255.
        images = Variable(images.to(device=device, dtype=torch.float32))
        labels = Variable(labels.to(device=device, dtype=torch.float32))
        

        pred = net(images)
        
        # wrong to use loss = criterion(pred.view(-1), labels.view(-1))
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 将每个epoch的loss添加到列表中
    loss_list.append(epoch_loss)
    print('epoch: {}, loss: {}'.format(epoch, epoch_loss))

# 绘制loss曲线
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("over")

```

```python
#loss曲线以及权重保存
```

```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
net.cuda()

optimizer = optim.RMSprop(net.parameters(), lr=0.003, weight_decay=1e-8)
criterion = nn.BCELoss()
# 创建空列表
loss_list = []
device = 'cuda'
for epoch in range(100):
    
    net.train()
    epoch_loss = 0
    
    for data in train_loader:
        
        images,labels = data
        images = images.permute(0,3,1,2)
        images = images/255.
        images = Variable(images.to(device=device, dtype=torch.float32))
        labels = Variable(labels.to(device=device, dtype=torch.float32))
        

        pred = net(images)
        
        # wrong to use loss = criterion(pred.view(-1), labels.view(-1))
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 将每个epoch的loss添加到列表中
    loss_list.append(epoch_loss)
    print('epoch: {}, loss: {}'.format(epoch, epoch_loss))
# 保存模型的权重
torch.save(net.state_dict(), 'test_best_model_weights.pth')

# 绘制loss曲线
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

    epoch: 0, loss: 12.591967642307281
    epoch: 1, loss: 8.95906463265419
    epoch: 2, loss: 6.887917742133141
    epoch: 3, loss: 5.665618501603603
    epoch: 4, loss: 5.01784361153841
    epoch: 5, loss: 4.544267825782299
    epoch: 6, loss: 4.280987352132797
    epoch: 7, loss: 4.323632791638374
    epoch: 8, loss: 3.7231258749961853
    epoch: 9, loss: 4.10112252086401
    epoch: 10, loss: 3.7079135552048683
    epoch: 11, loss: 3.321175195276737
    epoch: 12, loss: 3.153899487107992
    epoch: 13, loss: 4.416652657091618
    epoch: 14, loss: 2.9464755579829216
    epoch: 15, loss: 3.0585293285548687
    epoch: 16, loss: 2.823112454265356
    epoch: 17, loss: 2.470846589654684
    epoch: 18, loss: 2.963736031204462
    epoch: 19, loss: 2.497673250734806
    epoch: 20, loss: 2.59400087967515
    epoch: 21, loss: 2.538392435759306
    epoch: 22, loss: 2.404545810073614
    epoch: 23, loss: 2.423521541059017
    epoch: 24, loss: 2.430318921804428
    epoch: 25, loss: 2.341623492538929
    epoch: 26, loss: 2.1188094913959503
    epoch: 27, loss: 2.8676876723766327
    epoch: 28, loss: 2.2404664158821106
    epoch: 29, loss: 2.1453492417931557
    epoch: 30, loss: 2.1019953712821007
    epoch: 31, loss: 3.711727175861597
    epoch: 32, loss: 2.203179758042097
    epoch: 33, loss: 2.0334148220717907
    epoch: 34, loss: 1.893696766346693
    epoch: 35, loss: 1.9344691820442677
    epoch: 36, loss: 1.9536889903247356
    epoch: 37, loss: 2.0192645378410816
    epoch: 38, loss: 1.9665000140666962
    epoch: 39, loss: 2.2353107258677483
    epoch: 40, loss: 1.9161368533968925
    epoch: 41, loss: 1.6752426363527775
    epoch: 42, loss: 2.1763712242245674
    epoch: 43, loss: 1.8628390319645405
    epoch: 44, loss: 1.8358018808066845
    epoch: 45, loss: 1.857149451971054
    epoch: 46, loss: 1.7690518461167812
    epoch: 47, loss: 1.8195709250867367
    epoch: 48, loss: 1.664567720144987
    epoch: 49, loss: 2.467387370765209
    epoch: 50, loss: 1.8329112268984318
    epoch: 51, loss: 1.6650442853569984
    epoch: 52, loss: 1.6278569027781487
    epoch: 53, loss: 1.689095165580511
    epoch: 54, loss: 1.621750097721815
    epoch: 55, loss: 1.6082189120352268
    epoch: 56, loss: 2.4669259935617447
    epoch: 57, loss: 1.650984201580286
    epoch: 58, loss: 1.580072220414877
    epoch: 59, loss: 1.678004927933216
    epoch: 60, loss: 1.508884310722351
    epoch: 61, loss: 1.5919685177505016
    epoch: 62, loss: 1.480792060494423
    epoch: 63, loss: 2.647854156792164
    epoch: 64, loss: 1.5759475827217102
    epoch: 65, loss: 1.5202841460704803
    epoch: 66, loss: 1.5907666943967342
    epoch: 67, loss: 1.480655461549759
    epoch: 68, loss: 1.4498658291995525
    epoch: 69, loss: 1.4090162087231874
    epoch: 70, loss: 1.443433292210102
    epoch: 71, loss: 1.686006534844637
    epoch: 72, loss: 1.51864705234766
    epoch: 73, loss: 1.4041262231767178
    epoch: 74, loss: 1.4716542307287455
    epoch: 75, loss: 1.528109073638916
    epoch: 76, loss: 1.5686896592378616
    epoch: 77, loss: 1.3910310063511133
    epoch: 78, loss: 1.3555485662072897
    epoch: 79, loss: 1.374175637960434
    epoch: 80, loss: 1.5465534813702106
    epoch: 81, loss: 1.5382845047861338
    epoch: 82, loss: 1.3650132827460766
    epoch: 83, loss: 1.3558984640985727
    epoch: 84, loss: 1.287435369566083
    epoch: 85, loss: 1.341321922838688
    epoch: 86, loss: 1.3610958363860846
    epoch: 87, loss: 1.3387511409819126
    epoch: 88, loss: 1.260100118815899
    epoch: 89, loss: 1.595480389893055
    epoch: 90, loss: 1.2429084610193968
    epoch: 91, loss: 1.2533308863639832
    epoch: 92, loss: 1.5488744229078293
    epoch: 93, loss: 1.3442915491759777
    epoch: 94, loss: 1.5134109612554312
    epoch: 95, loss: 1.417487483471632
    epoch: 96, loss: 1.2214672174304724
    epoch: 97, loss: 1.221143428236246
    epoch: 98, loss: 1.2049008999019861
    epoch: 99, loss: 1.2115022838115692





    Text(0, 0.5, 'Loss')

![png](output_25_2.png)

```python
#添加dsc .iou指数
```

```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

net.cuda()
optimizer = optim.RMSprop(net.parameters(), lr=0.003, weight_decay=1e-8)
criterion = nn.BCELoss()
# 创建空列表
loss_list = []
device = 'cuda'
for epoch in range(100):
    
    net.train()
    epoch_loss = 0
    
    y_true = []
    y_pred = []
    
    for data in train_loader:
        
        images,labels = data
        images = images.permute(0,3,1,2)
        images = images/255.
        images = Variable(images.to(device=device, dtype=torch.float32))
        labels = Variable(labels.to(device=device, dtype=torch.float32))
        
        pred = net(images)
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_true.extend(labels.cpu().detach().numpy().flatten())
        y_pred.extend(pred.cpu().detach().numpy().flatten())
        
    # 计算准确率、精确度、召回率、F1 分数
    acc = accuracy_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    precision = precision_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    recall = recall_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    f1 = f1_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    
    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    loss_list.append(epoch_loss)
    
    print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")


# 保存模型的权重
torch.save(net.state_dict(), 'best1_model_weights.pth')
# 绘制损失函数曲线
plt.plot(loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

```

    Epoch 1: Loss: 12.5288, Acc: 0.8345, Precision: 0.6435, Recall: 0.6538, F1 Score: 0.6486, ROC AUC: 0.8804
    Epoch 2: Loss: 8.1156, Acc: 0.9008, Precision: 0.8286, Recall: 0.7245, F1 Score: 0.7730, ROC AUC: 0.9518
    Epoch 3: Loss: 6.3522, Acc: 0.9226, Precision: 0.8539, Recall: 0.8044, F1 Score: 0.8284, ROC AUC: 0.9709
    Epoch 4: Loss: 5.9908, Acc: 0.9278, Precision: 0.8641, Recall: 0.8163, F1 Score: 0.8395, ROC AUC: 0.9752
    Epoch 5: Loss: 5.8596, Acc: 0.9306, Precision: 0.8701, Recall: 0.8255, F1 Score: 0.8473, ROC AUC: 0.9752
    Epoch 6: Loss: 6.5732, Acc: 0.9206, Precision: 0.8569, Recall: 0.7922, F1 Score: 0.8233, ROC AUC: 0.9686
    Epoch 7: Loss: 4.4578, Acc: 0.9469, Precision: 0.8929, Recall: 0.8792, F1 Score: 0.8860, ROC AUC: 0.9861
    Epoch 8: Loss: 4.4520, Acc: 0.9471, Precision: 0.8934, Recall: 0.8818, F1 Score: 0.8876, ROC AUC: 0.9863
    Epoch 9: Loss: 4.1874, Acc: 0.9501, Precision: 0.8975, Recall: 0.8864, F1 Score: 0.8919, ROC AUC: 0.9875
    Epoch 10: Loss: 3.8317, Acc: 0.9545, Precision: 0.9061, Recall: 0.8988, F1 Score: 0.9024, ROC AUC: 0.9898
    Epoch 11: Loss: 3.6524, Acc: 0.9560, Precision: 0.9119, Recall: 0.8973, F1 Score: 0.9046, ROC AUC: 0.9905
    Epoch 12: Loss: 4.2344, Acc: 0.9502, Precision: 0.8991, Recall: 0.8874, F1 Score: 0.8932, ROC AUC: 0.9874
    Epoch 13: Loss: 3.6976, Acc: 0.9564, Precision: 0.9113, Recall: 0.9021, F1 Score: 0.9066, ROC AUC: 0.9904
    Epoch 14: Loss: 4.5103, Acc: 0.9460, Precision: 0.8957, Recall: 0.8703, F1 Score: 0.8828, ROC AUC: 0.9853
    Epoch 15: Loss: 3.6654, Acc: 0.9571, Precision: 0.9194, Recall: 0.8957, F1 Score: 0.9074, ROC AUC: 0.9905
    Epoch 16: Loss: 3.1709, Acc: 0.9631, Precision: 0.9215, Recall: 0.9207, F1 Score: 0.9211, ROC AUC: 0.9929
    Epoch 17: Loss: 3.0799, Acc: 0.9628, Precision: 0.9194, Recall: 0.9218, F1 Score: 0.9206, ROC AUC: 0.9934
    Epoch 18: Loss: 2.8233, Acc: 0.9665, Precision: 0.9295, Recall: 0.9284, F1 Score: 0.9290, ROC AUC: 0.9944
    Epoch 19: Loss: 4.4725, Acc: 0.9485, Precision: 0.9020, Recall: 0.8734, F1 Score: 0.8875, ROC AUC: 0.9852
    Epoch 20: Loss: 3.1417, Acc: 0.9629, Precision: 0.9211, Recall: 0.9206, F1 Score: 0.9209, ROC AUC: 0.9930
    Epoch 21: Loss: 2.7882, Acc: 0.9677, Precision: 0.9296, Recall: 0.9313, F1 Score: 0.9304, ROC AUC: 0.9945
    Epoch 22: Loss: 2.6708, Acc: 0.9692, Precision: 0.9312, Recall: 0.9378, F1 Score: 0.9345, ROC AUC: 0.9950
    Epoch 23: Loss: 3.0527, Acc: 0.9642, Precision: 0.9244, Recall: 0.9229, F1 Score: 0.9237, ROC AUC: 0.9934
    Epoch 24: Loss: 3.2383, Acc: 0.9630, Precision: 0.9229, Recall: 0.9183, F1 Score: 0.9206, ROC AUC: 0.9925
    Epoch 25: Loss: 2.4930, Acc: 0.9708, Precision: 0.9370, Recall: 0.9399, F1 Score: 0.9385, ROC AUC: 0.9957
    Epoch 26: Loss: 2.4111, Acc: 0.9720, Precision: 0.9394, Recall: 0.9417, F1 Score: 0.9405, ROC AUC: 0.9960
    Epoch 27: Loss: 2.7475, Acc: 0.9683, Precision: 0.9319, Recall: 0.9326, F1 Score: 0.9322, ROC AUC: 0.9947
    Epoch 28: Loss: 2.3947, Acc: 0.9724, Precision: 0.9400, Recall: 0.9431, F1 Score: 0.9416, ROC AUC: 0.9960
    Epoch 29: Loss: 2.2599, Acc: 0.9737, Precision: 0.9437, Recall: 0.9446, F1 Score: 0.9441, ROC AUC: 0.9964
    Epoch 30: Loss: 2.4713, Acc: 0.9713, Precision: 0.9372, Recall: 0.9406, F1 Score: 0.9389, ROC AUC: 0.9957
    Epoch 31: Loss: 2.3485, Acc: 0.9728, Precision: 0.9413, Recall: 0.9414, F1 Score: 0.9413, ROC AUC: 0.9961
    Epoch 32: Loss: 2.3415, Acc: 0.9732, Precision: 0.9416, Recall: 0.9421, F1 Score: 0.9419, ROC AUC: 0.9961
    Epoch 33: Loss: 2.2300, Acc: 0.9741, Precision: 0.9450, Recall: 0.9443, F1 Score: 0.9446, ROC AUC: 0.9965
    Epoch 34: Loss: 2.2361, Acc: 0.9738, Precision: 0.9439, Recall: 0.9433, F1 Score: 0.9436, ROC AUC: 0.9964
    Epoch 35: Loss: 2.3025, Acc: 0.9737, Precision: 0.9437, Recall: 0.9434, F1 Score: 0.9436, ROC AUC: 0.9962
    Epoch 36: Loss: 2.2307, Acc: 0.9745, Precision: 0.9454, Recall: 0.9457, F1 Score: 0.9455, ROC AUC: 0.9965
    Epoch 37: Loss: 2.2640, Acc: 0.9737, Precision: 0.9437, Recall: 0.9425, F1 Score: 0.9431, ROC AUC: 0.9964
    Epoch 38: Loss: 4.1002, Acc: 0.9543, Precision: 0.9084, Recall: 0.8947, F1 Score: 0.9015, ROC AUC: 0.9872
    Epoch 39: Loss: 2.2040, Acc: 0.9748, Precision: 0.9436, Recall: 0.9480, F1 Score: 0.9458, ROC AUC: 0.9966
    Epoch 40: Loss: 2.1098, Acc: 0.9757, Precision: 0.9467, Recall: 0.9492, F1 Score: 0.9480, ROC AUC: 0.9969
    Epoch 41: Loss: 1.9970, Acc: 0.9774, Precision: 0.9499, Recall: 0.9530, F1 Score: 0.9515, ROC AUC: 0.9971
    Epoch 42: Loss: 1.9958, Acc: 0.9774, Precision: 0.9506, Recall: 0.9517, F1 Score: 0.9512, ROC AUC: 0.9972
    Epoch 43: Loss: 2.1351, Acc: 0.9758, Precision: 0.9474, Recall: 0.9488, F1 Score: 0.9481, ROC AUC: 0.9968
    Epoch 44: Loss: 2.8511, Acc: 0.9706, Precision: 0.9350, Recall: 0.9390, F1 Score: 0.9370, ROC AUC: 0.9949
    Epoch 45: Loss: 2.6028, Acc: 0.9707, Precision: 0.9396, Recall: 0.9346, F1 Score: 0.9371, ROC AUC: 0.9950
    Epoch 46: Loss: 1.9557, Acc: 0.9779, Precision: 0.9530, Recall: 0.9528, F1 Score: 0.9529, ROC AUC: 0.9972
    Epoch 47: Loss: 1.9163, Acc: 0.9782, Precision: 0.9524, Recall: 0.9540, F1 Score: 0.9532, ROC AUC: 0.9973
    Epoch 48: Loss: 1.9782, Acc: 0.9774, Precision: 0.9514, Recall: 0.9530, F1 Score: 0.9522, ROC AUC: 0.9972
    Epoch 49: Loss: 1.8652, Acc: 0.9786, Precision: 0.9532, Recall: 0.9557, F1 Score: 0.9545, ROC AUC: 0.9975
    Epoch 50: Loss: 1.9077, Acc: 0.9781, Precision: 0.9531, Recall: 0.9542, F1 Score: 0.9537, ROC AUC: 0.9975
    Epoch 51: Loss: 1.8224, Acc: 0.9794, Precision: 0.9544, Recall: 0.9565, F1 Score: 0.9555, ROC AUC: 0.9976
    Epoch 52: Loss: 1.8685, Acc: 0.9783, Precision: 0.9537, Recall: 0.9542, F1 Score: 0.9540, ROC AUC: 0.9975
    Epoch 53: Loss: 1.7041, Acc: 0.9803, Precision: 0.9566, Recall: 0.9588, F1 Score: 0.9577, ROC AUC: 0.9979
    Epoch 54: Loss: 1.7699, Acc: 0.9797, Precision: 0.9554, Recall: 0.9575, F1 Score: 0.9565, ROC AUC: 0.9977
    Epoch 55: Loss: 1.7471, Acc: 0.9799, Precision: 0.9545, Recall: 0.9586, F1 Score: 0.9565, ROC AUC: 0.9978
    Epoch 56: Loss: 2.5103, Acc: 0.9716, Precision: 0.9397, Recall: 0.9389, F1 Score: 0.9393, ROC AUC: 0.9953
    Epoch 57: Loss: 1.8432, Acc: 0.9789, Precision: 0.9529, Recall: 0.9572, F1 Score: 0.9550, ROC AUC: 0.9976
    Epoch 58: Loss: 1.7010, Acc: 0.9805, Precision: 0.9576, Recall: 0.9590, F1 Score: 0.9583, ROC AUC: 0.9980
    Epoch 59: Loss: 1.6874, Acc: 0.9806, Precision: 0.9577, Recall: 0.9600, F1 Score: 0.9589, ROC AUC: 0.9980
    Epoch 60: Loss: 1.7026, Acc: 0.9804, Precision: 0.9566, Recall: 0.9596, F1 Score: 0.9581, ROC AUC: 0.9979
    Epoch 61: Loss: 1.7358, Acc: 0.9803, Precision: 0.9565, Recall: 0.9585, F1 Score: 0.9575, ROC AUC: 0.9979
    Epoch 62: Loss: 1.8529, Acc: 0.9786, Precision: 0.9535, Recall: 0.9544, F1 Score: 0.9539, ROC AUC: 0.9976
    Epoch 63: Loss: 1.7402, Acc: 0.9799, Precision: 0.9552, Recall: 0.9592, F1 Score: 0.9572, ROC AUC: 0.9979
    Epoch 64: Loss: 1.5686, Acc: 0.9818, Precision: 0.9602, Recall: 0.9620, F1 Score: 0.9611, ROC AUC: 0.9983
    Epoch 65: Loss: 1.6151, Acc: 0.9815, Precision: 0.9594, Recall: 0.9625, F1 Score: 0.9610, ROC AUC: 0.9981
    Epoch 66: Loss: 2.3115, Acc: 0.9739, Precision: 0.9464, Recall: 0.9420, F1 Score: 0.9442, ROC AUC: 0.9961
    Epoch 67: Loss: 1.7556, Acc: 0.9802, Precision: 0.9561, Recall: 0.9589, F1 Score: 0.9575, ROC AUC: 0.9978
    Epoch 68: Loss: 1.5672, Acc: 0.9819, Precision: 0.9601, Recall: 0.9628, F1 Score: 0.9614, ROC AUC: 0.9983
    Epoch 69: Loss: 1.5731, Acc: 0.9818, Precision: 0.9602, Recall: 0.9619, F1 Score: 0.9610, ROC AUC: 0.9982
    Epoch 70: Loss: 1.6820, Acc: 0.9806, Precision: 0.9565, Recall: 0.9606, F1 Score: 0.9586, ROC AUC: 0.9980
    Epoch 71: Loss: 1.5407, Acc: 0.9824, Precision: 0.9616, Recall: 0.9638, F1 Score: 0.9627, ROC AUC: 0.9983
    Epoch 72: Loss: 1.4788, Acc: 0.9829, Precision: 0.9624, Recall: 0.9645, F1 Score: 0.9634, ROC AUC: 0.9984
    Epoch 73: Loss: 1.6847, Acc: 0.9807, Precision: 0.9580, Recall: 0.9602, F1 Score: 0.9591, ROC AUC: 0.9980
    Epoch 74: Loss: 1.5720, Acc: 0.9817, Precision: 0.9590, Recall: 0.9633, F1 Score: 0.9612, ROC AUC: 0.9983
    Epoch 75: Loss: 1.6606, Acc: 0.9813, Precision: 0.9588, Recall: 0.9615, F1 Score: 0.9602, ROC AUC: 0.9980
    Epoch 76: Loss: 1.6001, Acc: 0.9818, Precision: 0.9606, Recall: 0.9619, F1 Score: 0.9612, ROC AUC: 0.9982
    Epoch 77: Loss: 1.5060, Acc: 0.9827, Precision: 0.9624, Recall: 0.9645, F1 Score: 0.9634, ROC AUC: 0.9984
    Epoch 78: Loss: 1.4248, Acc: 0.9835, Precision: 0.9640, Recall: 0.9664, F1 Score: 0.9652, ROC AUC: 0.9986
    Epoch 79: Loss: 1.5909, Acc: 0.9820, Precision: 0.9597, Recall: 0.9634, F1 Score: 0.9616, ROC AUC: 0.9981
    Epoch 80: Loss: 2.4412, Acc: 0.9741, Precision: 0.9440, Recall: 0.9456, F1 Score: 0.9448, ROC AUC: 0.9956
    Epoch 81: Loss: 1.7403, Acc: 0.9804, Precision: 0.9555, Recall: 0.9612, F1 Score: 0.9583, ROC AUC: 0.9978
    Epoch 82: Loss: 1.4764, Acc: 0.9831, Precision: 0.9609, Recall: 0.9666, F1 Score: 0.9637, ROC AUC: 0.9984
    Epoch 83: Loss: 1.4954, Acc: 0.9830, Precision: 0.9613, Recall: 0.9654, F1 Score: 0.9633, ROC AUC: 0.9984
    Epoch 84: Loss: 1.3937, Acc: 0.9838, Precision: 0.9644, Recall: 0.9666, F1 Score: 0.9655, ROC AUC: 0.9986
    Epoch 85: Loss: 1.3727, Acc: 0.9842, Precision: 0.9645, Recall: 0.9674, F1 Score: 0.9659, ROC AUC: 0.9986
    Epoch 86: Loss: 1.5015, Acc: 0.9828, Precision: 0.9625, Recall: 0.9636, F1 Score: 0.9630, ROC AUC: 0.9984
    Epoch 87: Loss: 1.3593, Acc: 0.9843, Precision: 0.9650, Recall: 0.9677, F1 Score: 0.9663, ROC AUC: 0.9987
    Epoch 88: Loss: 1.4211, Acc: 0.9837, Precision: 0.9631, Recall: 0.9669, F1 Score: 0.9650, ROC AUC: 0.9986
    Epoch 89: Loss: 1.4501, Acc: 0.9834, Precision: 0.9628, Recall: 0.9663, F1 Score: 0.9645, ROC AUC: 0.9985
    Epoch 90: Loss: 1.3683, Acc: 0.9842, Precision: 0.9652, Recall: 0.9674, F1 Score: 0.9663, ROC AUC: 0.9987
    Epoch 91: Loss: 1.3655, Acc: 0.9841, Precision: 0.9649, Recall: 0.9681, F1 Score: 0.9665, ROC AUC: 0.9987
    Epoch 92: Loss: 1.3224, Acc: 0.9847, Precision: 0.9659, Recall: 0.9688, F1 Score: 0.9674, ROC AUC: 0.9988
    Epoch 93: Loss: 1.3447, Acc: 0.9845, Precision: 0.9654, Recall: 0.9683, F1 Score: 0.9669, ROC AUC: 0.9987
    Epoch 94: Loss: 1.3468, Acc: 0.9846, Precision: 0.9661, Recall: 0.9684, F1 Score: 0.9673, ROC AUC: 0.9987
    Epoch 95: Loss: 1.3824, Acc: 0.9842, Precision: 0.9641, Recall: 0.9682, F1 Score: 0.9661, ROC AUC: 0.9986
    Epoch 96: Loss: 1.3023, Acc: 0.9851, Precision: 0.9669, Recall: 0.9689, F1 Score: 0.9679, ROC AUC: 0.9988
    Epoch 97: Loss: 1.5625, Acc: 0.9826, Precision: 0.9616, Recall: 0.9639, F1 Score: 0.9628, ROC AUC: 0.9982
    Epoch 98: Loss: 1.3565, Acc: 0.9845, Precision: 0.9651, Recall: 0.9691, F1 Score: 0.9671, ROC AUC: 0.9987
    Epoch 99: Loss: 1.2687, Acc: 0.9852, Precision: 0.9665, Recall: 0.9703, F1 Score: 0.9684, ROC AUC: 0.9989
    Epoch 100: Loss: 1.2905, Acc: 0.9852, Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9686, ROC AUC: 0.9988

![png](output_27_1.png)

```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

net.cuda()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-8)
criterion = nn.BCELoss()
# 创建空列表
loss_list = []
acc_list = []
prec_list = []
rec_list = []
f1_list = []
roc_auc_list = []

device = 'cuda'
for epoch in range(100):
    
    net.train()
    epoch_loss = 0
    
    y_true = []
    y_pred = []
    
    for data in train_loader:
        
        images,labels = data
        images = images.permute(0,3,1,2)
        images = images/255.
        images = Variable(images.to(device=device, dtype=torch.float32))
        labels = Variable(labels.to(device=device, dtype=torch.float32))
        
        pred = net(images)
        
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_labels = (pred > 0.5).float()
        y_true.extend(labels.cpu().numpy().ravel().tolist())
        y_pred.extend(pred_labels.cpu().detach().numpy().ravel().tolist())
        
    loss_list.append(epoch_loss)
    print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {acc:.4f},  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    epoch_loss /= len(train_loader)
    
    acc = accuracy_score(y_true, y_pred)
    acc_list.append(acc)
    
    prec = precision_score(y_true, y_pred)
    prec_list.append(prec)
    
    rec = recall_score(y_true, y_pred)
    rec_list.append(rec)
    
    f1 = f1_score(y_true, y_pred)
    f1_list.append(f1)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_auc_list.append(roc_auc)
    
# 保存模型的权重
torch.save(net.state_dict(), 'best_model_weights.pth')
# 绘制损失函数曲线
plt.plot(loss_list)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 绘制准确率曲线
plt.plot(acc_list)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 绘制精确度曲线
plt.plot(prec_list)
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.show()

# 绘制召回率曲线
plt.plot(rec_list)
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.show()

# 绘制F1分数曲线
plt.plot(f1_list)
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.show()

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
```

    Epoch 1: Loss: 1.3376, Acc: 0.9852,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9686, ROC AUC: 0.9988
    Epoch 2: Loss: 1.1733, Acc: 0.9846,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9672, ROC AUC: 0.9792
    Epoch 3: Loss: 1.1809, Acc: 0.9863,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9707, ROC AUC: 0.9814
    Epoch 4: Loss: 1.1723, Acc: 0.9864,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9708, ROC AUC: 0.9816
    Epoch 5: Loss: 1.1359, Acc: 0.9864,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9711, ROC AUC: 0.9816
    Epoch 6: Loss: 1.1366, Acc: 0.9867,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9716, ROC AUC: 0.9821
    Epoch 7: Loss: 1.1235, Acc: 0.9868,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9716, ROC AUC: 0.9821
    Epoch 8: Loss: 1.1296, Acc: 0.9869,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9718, ROC AUC: 0.9823
    Epoch 9: Loss: 1.1216, Acc: 0.9868,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9719, ROC AUC: 0.9823
    Epoch 10: Loss: 1.1131, Acc: 0.9868,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9719, ROC AUC: 0.9822
    Epoch 11: Loss: 1.1168, Acc: 0.9870,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9723, ROC AUC: 0.9825
    Epoch 12: Loss: 1.0860, Acc: 0.9870,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9722, ROC AUC: 0.9825
    Epoch 13: Loss: 1.0968, Acc: 0.9872,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9728, ROC AUC: 0.9828
    Epoch 14: Loss: 1.0926, Acc: 0.9871,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9723, ROC AUC: 0.9825
    Epoch 15: Loss: 1.0960, Acc: 0.9871,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9724, ROC AUC: 0.9825
    Epoch 16: Loss: 1.1251, Acc: 0.9872,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9725, ROC AUC: 0.9827
    Epoch 17: Loss: 1.0668, Acc: 0.9869,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9720, ROC AUC: 0.9824
    Epoch 18: Loss: 1.0814, Acc: 0.9874,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9732, ROC AUC: 0.9832
    Epoch 19: Loss: 1.0772, Acc: 0.9873,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9730, ROC AUC: 0.9830
    Epoch 20: Loss: 1.0527, Acc: 0.9874,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9732, ROC AUC: 0.9830
    Epoch 21: Loss: 1.0527, Acc: 0.9876,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9734, ROC AUC: 0.9833
    Epoch 22: Loss: 1.0660, Acc: 0.9875,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9732, ROC AUC: 0.9830
    Epoch 23: Loss: 1.0346, Acc: 0.9875,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9732, ROC AUC: 0.9832
    Epoch 24: Loss: 1.0225, Acc: 0.9879,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9739, ROC AUC: 0.9836
    Epoch 25: Loss: 1.0576, Acc: 0.9879,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9742, ROC AUC: 0.9837
    Epoch 26: Loss: 1.0493, Acc: 0.9876,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9734, ROC AUC: 0.9833
    Epoch 27: Loss: 1.0292, Acc: 0.9877,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9737, ROC AUC: 0.9835
    Epoch 28: Loss: 1.0450, Acc: 0.9878,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9738, ROC AUC: 0.9836
    Epoch 29: Loss: 1.0231, Acc: 0.9876,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9737, ROC AUC: 0.9834
    Epoch 30: Loss: 1.0343, Acc: 0.9878,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9739, ROC AUC: 0.9837
    Epoch 31: Loss: 1.0920, Acc: 0.9878,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9739, ROC AUC: 0.9837
    Epoch 32: Loss: 1.0336, Acc: 0.9873,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9728, ROC AUC: 0.9829
    Epoch 33: Loss: 1.0122, Acc: 0.9878,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9739, ROC AUC: 0.9837
    Epoch 34: Loss: 1.0107, Acc: 0.9881,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9745, ROC AUC: 0.9840
    Epoch 35: Loss: 1.0076, Acc: 0.9881,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9744, ROC AUC: 0.9840
    Epoch 36: Loss: 1.0202, Acc: 0.9880,  Precision: 0.9674, Recall: 0.9697, F1 Score: 0.9744, ROC AUC: 0.9839

## Test model

```python
test_dataset_noaug = Dataset(
    x_train_dir, 
    y_train_dir,
    augmentation=get_test_augmentation(),
    )
```

```python
image, mask = test_dataset_noaug[120]  # 从测试数据集中取出第120张图片和对应的掩码
show_image = image  # 将图片赋值给变量show_image
with torch.no_grad():
    image = image/255.  # 将像素值归一化到[0,1]区间
    image = image.astype('float32')  # 将图片转换为float32类型
    image = torch.from_numpy(image)  # 将图片转换为PyTorch张量
    image = image.permute(2,0,1)  # 调整维度顺序为通道-高度-宽度
    image = image.to()  # 将张量移动到指定设备上（默认为CPU）
    print(image.shape)  # 打印张量的形状
    
    pred = net(image.unsqueeze(0).cuda())  # 将张量输入到神经网络中，得到预测结果
    pred = pred.cpu()  # 将预测结果移动到CPU上
    
pred = pred>0.5  # 将预测结果二值化为0或1，阈值为0.5
visualize(image=show_image,GT=mask[0,:,:],Pred=pred[0,0,:,:])  # 调用visualize函数，将原始图片、掩码和预测结果可视化
    
plt.figure('mask')
print(mask.shape)
print(np.min(mask),np.max(mask))
plt.imshow(mask[0,:,:])  # 显示掩码
plt.show()
pred = pred>0.8  # 将预测结果二值化为0或1，阈值为0.8
print(np.min(pred.numpy()),np.max(pred.numpy()))
plt.imshow(pred[0,0,:,:])  # 显示预测结果
plt.show()

```

    torch.Size([3, 320, 320])

![png](output_31_1.png)

    (1, 320, 320)
    0.0 1.0

![png](output_31_3.png)

    False True

![png](output_31_5.png)

```python
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Stack the original image and predicted mask side-by-side
stacked = torch.cat((image, pred.float()), dim=3)

# Create a grid of images
grid = vutils.make_grid(stacked, normalize=True, scale_each=True)

# Visualize the stacked image
plt.figure('Stacked Image')
plt.imshow(grid.permute(1,2,0))
plt.axis('off')
plt.show()

# Load the pre-trained model
#net = UNet()
net = UNet(n_channels=3, n_classes=1)
net.load_state_dict(torch.load('best1_model_weights.pth'))

# Load the test image
#image = Image.open('/root/Test/camvid/camvid/test_images/0006R0_f03300.png')
image = Image.open('test.png')

# Preprocess the image
# 将numpy数组类型的图像转换为0~1之间的浮点数
image = np.array(image)
image = image / 255.
# 将图像数据类型转换为float32
image = image.astype('float32')
# 将numpy数组类型的图像转换为PyTorch张量类型
image = torch.from_numpy(image)
# 改变张量的维度顺序，从(H,W,C)变为(C,H,W)
image = image.permute(2,0,1)
# 在张量的第0维添加一个维度，变为(1,C,H,W)
image = image.unsqueeze(0)
# Make a prediction on the preprocessed image using the loaded model
with torch.no_grad():
    pred = net(image)

# Threshold the predicted output at 0.5
pred = pred > 0.5

# Visualize the original image and predicted mask
plt.figure('Original Image')
plt.imshow(image.squeeze().permute(1,2,0))
plt.axis('off')

plt.figure('Predicted Mask')
plt.imshow(pred.squeeze().cpu().numpy())
plt.axis('off')

plt.show()

```

    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[36], line 9
          6 import torchvision.utils as vutils
          8 # Stack the original image and predicted mask side-by-side
    ----> 9 stacked = torch.cat((image, pred.float()), dim=3)
         11 # Create a grid of images
         12 grid = vutils.make_grid(stacked, normalize=True, scale_each=True)


    IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)

```python
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
#from net import UNet  # Assuming you have defined the UNet model in a separate file

# Load the pre-trained model
net = UNet(n_channels=3, n_classes=1)
net.load_state_dict(torch.load('best1_model_weights.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()

# Load the test image
image = Image.open('test.png')

# Preprocess the image
image = np.array(image)
image = image / 255.
image = image.astype('float32')
image = torch.from_numpy(image)
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)
image = image.to(device)

# Make a prediction on the preprocessed image using the loaded model
with torch.no_grad():
    pred = net(image)

# Threshold the predicted output at 0.5
pred = pred > 0.5

# Visualize the original image and predicted mask
plt.figure('Original Image')
plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
plt.axis('off')

plt.figure('Predicted Mask')
plt.imshow(pred.squeeze().cpu().numpy())
plt.axis('off')

plt.show()

# Stack the original image and predicted mask side-by-side
stacked = torch.cat((image.squeeze(), pred.float().squeeze()), dim=2)

# Create a grid of images
grid = vutils.make_grid(stacked, normalize=True, scale_each=True)

# Visualize the stacked image
plt.figure('Stacked Image')
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()

```

![png](output_33_0.png)

![png](output_33_1.png)

    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[41], line 47
         44 plt.show()
         46 # Stack the original image and predicted mask side-by-side
    ---> 47 stacked = torch.cat((image.squeeze(), pred.float().squeeze()), dim=2)
         49 # Create a grid of images
         50 grid = vutils.make_grid(stacked, normalize=True, scale_each=True)


    RuntimeError: Tensors must have same number of dimensions: got 2 and 3

```python

```
