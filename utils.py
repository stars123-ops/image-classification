import torch
import os
import time


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        accuracy: 当前准确率
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }

    torch.save(checkpoint, filepath)
    print(f"✅ 检查点已保存: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载模型检查点

    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        model: 加载后的模型
        optimizer: 加载后的优化器
        epoch: 加载的epoch
        accuracy: 加载的准确率
    """
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        accuracy = checkpoint['accuracy']
        print(f"📥 加载检查点: epoch={epoch}, accuracy={accuracy:.2f}%")

        return model, optimizer, epoch, accuracy
    else:
        print(f"⚠️ 检查点不存在: {filepath}")
        return model, optimizer, 0, 0.0


class AverageMeter:
    """计算和存储平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count