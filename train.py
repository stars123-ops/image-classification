import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
from tqdm import tqdm

# 导入自定义模块
from model import SimpleCNN
from dataset import get_dataloaders
from utils import save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch图像分类训练')

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数 (Windows建议设为0)')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=10,
                        help='分类类别数 (默认: 10)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='训练设备 (默认: cuda)')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./outputs/weights',
                        help='模型保存目录 (默认: ./outputs/weights)')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs',
                        help='日志保存目录 (默认: ./outputs/logs)')

    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()  # 切换到训练模式
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc=repr('train'), leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 将数据移到指定设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 统计信息
        loss_meter.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    accuracy = 100. * correct / total
    return loss_meter.avg, accuracy


def validate(model, dataloader, criterion, device):

    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='验证', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return loss_meter.avg, accuracy


def main():
    """主函数"""
    # 1. 解析参数
    args = parse_args()
    print("=" * 60)
    print("训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # 2. 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = 'cpu'

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 3. 准备数据
    print("加载数据...")
    train_loader, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)}张图片")
    print(f"测试集: {len(test_loader.dataset)}张图片")
    print(f"类别: {class_names}")

    # 4. 创建模型
    print("创建模型...")
    model = SimpleCNN(num_classes=args.num_classes).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. 训练循环
    print("\n开始训练...")
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )

        # 计算时间
        epoch_time = time.time() - epoch_start

        # 打印结果
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        print(f"  时间: {epoch_time:.1f}秒")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, best_path)
            print(f"  新的最佳准确率! 模型已保存")

        # 定期保存检查点
        if epoch % 5 == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)

        # 7. 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"训练完成!")
    print(f"总时间: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存在: {args.save_dir}")
    print("=" * 60)

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, val_acc, final_path)


if __name__ == '__main__':
    main()
