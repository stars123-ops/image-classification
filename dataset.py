import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64, num_workers=0):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # 归一化
                             (0.2023, 0.1994, 0.2010))
    ])

    # 数据转换（测试集）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # 下载CIFAR10数据集
    train_dataset = datasets.CIFAR10(
        root='./data',  # 保存路径
        train=True,  # 训练集
        download=True,  # 自动下载
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,  # 测试集
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers
    )

    # CIFAR10的类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    return train_loader, test_loader, class_names