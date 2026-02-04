import torch
import argparse
from model import SimpleCNN
from dataset import get_dataloaders
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='测试模型')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str,
                        default='./outputs/weights/best_model.pth',
                        help='模型路径')
    return parser.parse_args()


def main():
    args = parse_args()
    print("测试配置:")
    print(f"  设备: {args.device}")
    print(f"  模型: {args.model_path}")

    # 设置设备
    device = torch.device(args.device)

    # 加载数据
    _, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=0
    )
    print(f"测试集: {len(test_loader.dataset)}张图片")

    # 创建并加载模型
    model = SimpleCNN(num_classes=10).to(device)
    model, _, _, loaded_acc = load_checkpoint(args.model_path, model)

    # 测试
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print("\n" + "=" * 50)
    print(f"测试结果:")
    print(f"  正确/总数: {correct}/{total}")
    print(f"  测试准确率: {accuracy:.2f}%")
    if loaded_acc > 0:
        print(f"  训练时准确率: {loaded_acc:.2f}%")
    print("=" * 50)

    # 各类别准确率
    print("\n各类别准确率:")
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            c = predicted.eq(targets).squeeze()

            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {class_names[i]:15s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")


if __name__ == '__main__':
    main()