## Loading data

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



<img title="" src="file:///C:/Users/Administrator/Downloads/unet/Unet/road-segmentation-task-based-on-unet/img/output_11_1.png" alt="png" style="zoom:50%;">

## Create model and train



```python
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
```

```python
class DoubleConv(nn.Module):
    """
    DoubleConv模块：两个连续的卷积操作，每个卷积操作后都跟着批归一化和ReLU激活函数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3的卷积操作
            nn.BatchNorm2d(out_channels),  # 批归一化操作
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3的卷积操作
            nn.BatchNorm2d(out_channels),  # 批归一化操作
            nn.ReLU(inplace=True)  # ReLU激活函数
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Down模块：通过最大池化操作下采样，然后使用DoubleConv模块进行卷积操作
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化操作进行下采样
            DoubleConv(in_channels, out_channels)  # 使用DoubleConv模块进行卷积操作
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Up模块：上采样操作，然后使用DoubleConv模块进行卷积操作
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)  # 使用DoubleConv模块进行卷积操作

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样操作
        # 计算x2和x1在高度和宽度上的差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 对x1进行填充，使其与x2具有相同的高度和宽度
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 将上采样后的x1和x2在通道维度上拼接起来
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    OutConv模块：使用1x1卷积操作生成最终的分割输出
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1的卷积操作

    def forward(self, x):
        return self.conv(x)
```

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入图像的通道数
        self.n_classes = n_classes  # 分割的类别数
        self.bilinear = bilinear  # 是否使用双线性插值进行上采样

        # 定义各层模块
        self.inc = DoubleConv(n_channels, 32)  # 输入层
        self.down1 = Down(32, 64)  # 下采样层1
        self.down2 = Down(64, 128)  # 下采样层2
        self.down3 = Down(128, 256)  # 下采样层3
        self.down4 = Down(256, 256)  # 下采样层4
        self.up1 = Up(512, 128, bilinear)  # 上采样层1
        self.up2 = Up(256, 64, bilinear)  # 上采样层2
        self.up3 = Up(128, 32, bilinear)  # 上采样层3
        self.up4 = Up(64, 32, bilinear)  # 上采样层4
        self.outc = OutConv(32, n_classes)  # 输出层
        self.out = torch.sigmoid  # 输出层激活函数

    def forward(self, x):
        # 前向传播过程
        x1 = self.inc(x)  # 输入层
        x2 = self.down1(x1)  # 下采样层1
        x3 = self.down2(x2)  # 下采样层2
        x4 = self.down3(x3)  # 下采样层3
        x5 = self.down4(x4)  # 下采样层4
        x = self.up1(x5, x4)  # 上采样层1
        x = self.up2(x, x3)  # 上采样层2
        x = self.up3(x, x2)  # 上采样层3
        x = self.up4(x, x1)  # 上采样层4
        logits = self.outc(x)  # 输出层
        logits = self.out(logits)  # 输出层激活函数

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
    """
    DoubleConv模块：两个连续的卷积操作，每个卷积操作后都跟着批归一化和ReLU激活函数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3的卷积操作
            nn.BatchNorm2d(out_channels),  # 批归一化操作
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3的卷积操作
            nn.BatchNorm2d(out_channels),  # 批归一化操作
            nn.ReLU(inplace=True)  # ReLU激活函数
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Down模块：通过最大池化操作下采样，然后使用DoubleConv模块进行卷积操作
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化操作进行下采样
            DoubleConv(in_channels, out_channels)  # 使用DoubleConv模块进行卷积操作
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Up模块：上采样操作，然后使用DoubleConv模块进行卷积操作
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)  # 使用DoubleConv模块进行卷积操作

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样操作
        # 计算x2和x1在高度和宽度上的差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 对x1进行填充，使其与x2具有相同的高度和宽度
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 将上采样后的x1和x2在通道维度上拼接起来
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    OutConv模块：使用1x1卷积操作生成最终的分割输出
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1的卷积操作

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入图像的通道数
        self.n_classes = n_classes  # 分割的类别数
        self.bilinear = bilinear  # 是否使用双线性插值进行上采样

        # 定义各层模块
        self.inc = DoubleConv(n_channels, 32)  # 输入层
        self.down1 = Down(32, 64)  # 下采样层1
        self.down2 = Down(64, 128)  # 下采样层2
        self.down3 = Down(128, 256)  # 下采样层3
        self.down4 = Down(256, 256)  # 下采样层4
        self.up1 = Up(512, 128, bilinear)  # 上采样层1
        self.up2 = Up(256, 64, bilinear)  # 上采样层2
        self.up3 = Up(128, 32, bilinear)  # 上采样层3
        self.up4 = Up(64, 32, bilinear)  # 上采样层4
        self.outc = OutConv(32, n_classes)  # 输出层
        self.out = torch.sigmoid  # 输出层激活函数

    def forward(self, x):
        # 前向传播过程
        x1 = self.inc(x)  # 输入层
        x2 = self.down1(x1)  # 下采样层1
        x3 = self.down2(x2)  # 下采样层2
        x4 = self.down3(x3)  # 下采样层3
        x5 = self.down4(x4)  # 下采样层4
        x = self.up1(x5, x4)  # 上采样层1
        x = self.up2(x, x3)  # 上采样层2
        x = self.up3(x, x2)  # 上采样层3
        x = self.up4(x, x1)  # 上采样层4
        logits = self.outc(x)  # 输出层
        logits = self.out(logits)  # 输出层激活函数

        return logits
# 可视化模型结构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
summary(model, input_size=(3, 320, 320))

```



```python
```python
train_dataset = Dataset(
    x_train_dir,  # 训练图像的文件夹路径
    y_train_dir,  # 对应的标签图像的文件夹路径
    augmentation=get_training_augmentation(),  # 数据增强操作
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 创建训练数据加载器，每个批次包含16个样本，且每个epoch都对数据进行重新洗牌
```

```

```python
net = UNet(n_channels=3, n_classes=1)
```

```python
image, mask = train_dataset[1]  # 从训练数据集中获取图像和掩码数据

with torch.no_grad():  # 在评估阶段，不需要计算梯度
    net.to('cuda')  # 将网络模型移到GPU上进行加速计算
    image = image/255.  # 对图像进行归一化，将像素值缩放到0-1范围
    image = image.astype('float32')  # 将图像的数据类型转换为float32
    image = torch.from_numpy(image)  # 将numpy数组转换为PyTorch张量
    image = image.permute(2,0,1)  # 重排图像通道的顺序，将通道维度放在最前面
    image = image.to()  # 将图像数据移到GPU上进行加速计算
    print(image.shape)  # 打印图像张量的形状信息

    #pred = torch.argmax(pred, dim=1).squeeze().numpy()
    pred = net(image.unsqueeze(0).cuda())  # 将图像输入网络进行预测，并将结果移到GPU上进行加速计算
    pred = pred.cpu()  # 将预测结果移到CPU上进行后续处理

plt.figure('mask')  # 创建名为'mask'的图像窗口
print(mask.shape)  # 打印掩码张量的形状信息
print(np.min(mask),np.max(mask))  # 打印掩码张量的最小值和最大值
plt.imshow(mask[0,:,:])  # 在图像窗口中显示掩码图像
plt.show()  # 显示图像窗口

plt.figure('pred')  # 创建名为'pred'的图像窗口
# pred = pred.view(720,960)
print(pred.shape)  # 打印预测结果张量的形状信息
print(np.min(pred.numpy()),np.max(pred.numpy()))  # 打印预测结果张量的最小值和最大值
plt.imshow(pred[0,0,:,:])  # 在图像窗口中显示预测结果图像
plt.show()  # 显示图像窗口
```

    torch.Size([3, 320, 320])
    (1, 320, 320)
    0.0 1.0



```python
```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

net.cuda()  # 将网络模型移动到GPU上进行加速计算
optimizer = optim.RMSprop(net.parameters(), lr=0.003, weight_decay=1e-8)  # 定义优化器
criterion = nn.BCELoss()  # 定义损失函数为二分类交叉熵

# 创建空列表
loss_list = []
device = 'cuda'

for epoch in range(100):
    net.train()  # 设置网络为训练模式
    epoch_loss = 0

    y_true = []  # 用于存储真实标签
    y_pred = []  # 用于存储预测标签

    for data in train_loader:
        images, labels = data
        images = images.permute(0, 3, 1, 2)  # 重排图像通道的顺序，将通道维度放在第2个位置
        images = images / 255.  # 对图像进行归一化，将像素值缩放到0-1范围
        images = Variable(images.to(device=device, dtype=torch.float32))  # 将图像数据移到GPU上进行加速计算，并将其封装成变量
        labels = Variable(labels.to(device=device, dtype=torch.float32))  # 将标签数据移到GPU上进行加速计算，并将其封装成变量

        pred = net(images)  # 将图像输入网络进行预测
        loss = criterion(pred, labels)  # 计算预测结果与真实标签之间的损失
        epoch_loss += loss.item()  # 累计损失值

        optimizer.zero_grad()  # 清空优化器中的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新网络权重

        y_true.extend(labels.cpu().detach().numpy().flatten())  # 将真实标签转移到CPU上，并添加到y_true列表中
        y_pred.extend(pred.cpu().detach().numpy().flatten())  # 将预测标签转移到CPU上，并添加到y_pred列表中

    # 计算准确率、精确度、召回率、F1分数
    acc = accuracy_score(y_true, (np.array(y_pred) > 0.5).astype(int))  # 计算准确率
    precision = precision_score(y_true, (np.array(y_pred) > 0.5).astype(int))  # 计算精确度
    recall = recall_score(y_true, (np.array(y_pred) > 0.5).astype(int))  # 计算召回率
    f1 = f1_score(y_true, (np.array(y_pred) > 0.5).astype(int))  # 计算F1分数

    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  # 计算ROC曲线的假正例率、真正例率和阈值
    roc_auc = auc(fpr, tpr)  # 计算ROC曲线下的面积

    loss_list.append(epoch_loss)  # 将每个epoch的损失值添加到loss_list中

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
    #输出过长省略10-99次epoch





```python
```python
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

net.cuda() # 将网络模型移动到GPU上进行计算
optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-8) # 定义优化器
criterion = nn.BCELoss() # 定义损失函数为二分类交叉熵

# 创建空列表，用于存储训练过程中的指标
loss_list = []
acc_list = []
prec_list = []
rec_list = []
f1_list = []
roc_auc_list = []

device = 'cuda' # 设置设备为GPU
for epoch in range(100):

    net.train() # 设置网络为训练模式
    epoch_loss = 0

    y_true = [] # 存储真实标签
    y_pred = [] # 存储预测标签

    for data in train_loader:

        images,labels = data
        images = images.permute(0,3,1,2) # 调整图片维度顺序
        images = images/255. # 对图片进行归一化处理
        images = Variable(images.to(device=device, dtype=torch.float32)) # 将数据移动到GPU上
        labels = Variable(labels.to(device=device, dtype=torch.float32))

        pred = net(images) # 前向传播

        loss = criterion(pred, labels) # 计算损失
        epoch_loss += loss.item() # 累加损失值

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        pred_labels = (pred > 0.5).float() # 将预测值转换为二分类标签
        y_true.extend(labels.cpu().numpy().ravel().tolist()) # 将真实标签转换为一维列表并存储
        y_pred.extend(pred_labels.cpu().detach().numpy().ravel().tolist()) # 将预测标签转换为一维列表并存储

    loss_list.append(epoch_loss)
    epoch_loss /= len(train_loader)

    acc = accuracy_score(y_true, y_pred) # 计算准确率
    acc_list.append(acc)

    prec = precision_score(y_true, y_pred) # 计算精确率
    prec_list.append(prec)

    rec = recall_score(y_true, y_pred) # 计算召回率
    rec_list.append(rec)

    f1 = f1_score(y_true, y_pred) # 计算F1 Score
    f1_list.append(f1)

    fpr, tpr, _ = roc_curve(y_true, y_pred) # 计算ROC曲线的假正率和真正率
    roc_auc = auc(fpr, tpr) # 计算ROC曲线下的面积
    roc_auc_list.append(roc_auc)

# 保存模型的权重
torch.save(net.state_dict(), 'best_model_weights.pth')


```

```

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



```python
```python
import cv2  
import torch  
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import torchvision.utils as vutils  
#from net import UNet  # 假设你在另一个文件中定义了UNet模型

# 加载预训练模型
net = UNet(n_channels=3, n_classes=1)  # 创建UNet模型对象
net.load_state_dict(torch.load('best1_model_weights.pth'))  # 加载模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为GPU或CPU
net.to(device)  # 将模型移动到指定设备上
net.eval()  # 设置模型为评估模式

# 加载测试图像
image = Image.open('test.png')  # 使用PIL的Image.open()函数加载图像

# 图像预处理
image = np.array(image)  # 将图像转换为NumPy数组
image = image / 255.  # 归一化图像像素值到0-1范围
image = image.astype('float32')  # 将图像像素值数据类型转换为float32
image = torch.from_numpy(image)  # 将NumPy数组转换为PyTorch张量
image = image.permute(2, 0, 1)  # 修改张量的维度顺序
image = image.unsqueeze(0)  # 在第0维上添加一个维度，将图像扩展为4维张量
image = image.to(device)  # 将图像张量移动到指定设备上

# 使用加载的模型对预处理图像进行预测
with torch.no_grad():  # 在预测阶段不需要计算梯度
    pred = net(image)  # 对图像进行预测

# 将预测输出阈值化为0.5
pred = pred > 0.5

# 可视化原始图像和预测的掩膜
plt.figure('原始图像')
plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())  # 可视化原始图像
plt.axis('off')  # 关闭坐标轴

plt.figure('预测的掩膜')
plt.imshow(pred.squeeze().cpu().numpy())  # 可视化预测的掩膜
plt.axis('off')  # 关闭坐标轴

plt.show()  # 显示图像

# 将原始图像和预测的掩膜并排堆叠在一起
stacked = torch.cat((image.squeeze(), pred.float().squeeze()), dim=2)  # 将原始图像和预测的掩膜在第2维上进行堆叠

# 创建一个图像网格
grid = vutils.make_grid(stacked, normalize=True, scale_each=True)  # 创建一个图像网格

# 可视化堆叠的图像
plt.figure('堆叠的图像')
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # 可视化堆叠的图像
plt.axis('off')  # 关闭坐标轴
plt.show()  # 显示图像
```



```
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入预训练模型
net = UNet(n_channels=3, n_classes=1)
net.load_state_dict(torch.load('test_best_model_weights.pth'))

# 加载测试图像
image = Image.open('test.png')

# 预处理图像
image = np.array(image)
image = image / 255.
image = image.astype('float32')
image = torch.from_numpy(image)
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)

# 使用加载的模型对预处理图像进行预测
with torch.no_grad():
    pred = net(image)

# 将预测输出阈值化为0.5
pred = pred > 0.5

# 将预测的掩膜转换为numpy数组
pred_np = pred.squeeze().cpu().numpy()

# 找到预测掩膜的轮廓
contours, _ = cv2.findContours(pred_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 在原始图像上绘制轮廓
image_contours = cv2.drawContours(np.array(image.squeeze().permute(1, 2, 0) * 255, dtype=np.uint8), contours, -1, (0, 255, 0), 2)

# Visualize the original image with contours and predicted mask
plt.figure('Original Image with Contours')
plt.imshow(image_contours)
plt.axis('off')

plt.figure('Predicted Mask')
plt.imshow(pred_np)
plt.axis('off')

# Save the predicted mask as a JPEG image
pred_image = (pred_np * 255).astype(np.uint8)
cv2.imwrite('predicted_mask.jpg', pred_image)

plt.show()
```






