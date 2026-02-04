import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """简单的CNN模型，用于CIFAR10分类"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状: [batch_size, 3, 32, 32]

        # 卷积层1 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 16, 16]

        # 卷积层2 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 8, 8]

        # 卷积层3 + 激活 + 池化
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch, 128, 4, 4]

        # 展平特征图
        x = x.view(x.size(0), -1)  # -> [batch, 128*4*4]

        # 全连接层1 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 全连接层2（输出层）
        x = self.fc2(x)  # -> [batch, num_classes]

        return x
