import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from swin_backbone import create_swin_segnet


class KvasirSEGDataset(Dataset):
    """Kvasir-SEG Dataset for medical image segmentation"""

    def __init__(self, root_dir, split='train', img_size=224, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # 设置图像和掩码路径
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')

        # 获取所有图像文件名
        self.image_names = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])

        # 设置变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.image_names[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度掩码

        # 应用变换
        if self.transform:
            image = self.transform(image)

        mask = self.mask_transform(mask)
        # 将掩码二值化 (0或1)
        mask = (mask > 0.5).float()

        return image, mask


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 应用sigmoid激活
        inputs = torch.sigmoid(inputs)

        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算Dice系数
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """结合BCE和Dice损失"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(pred, target, threshold=0.5):
    """计算IoU指标"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 前向传播
        if hasattr(model, 'deep_supervision') and model.deep_supervision:
            output, aux_outputs = model(data)
            # 主损失
            main_loss = criterion(output, target)
            # 辅助损失
            aux_loss = sum([criterion(aux_out, target) for aux_out in aux_outputs])
            loss = main_loss + 0.4 * aux_loss
        else:
            output = model(data)
            loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算指标
        iou = calculate_iou(output, target)
        total_loss += loss.item()
        total_iou += iou

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{iou:.4f}'
        })

        # 记录到TensorBoard
        if writer and batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/BatchIoU', iou, global_step)

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    return avg_loss, avg_iou


def validate_epoch(model, val_loader, criterion, device, epoch, writer=None):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')

        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # 前向传播
            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                output, _ = model(data)  # 验证时只使用主输出
            else:
                output = model(data)

            loss = criterion(output, target)
            iou = calculate_iou(output, target)

            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}'
            })

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/IoU', avg_iou, epoch)

    return avg_loss, avg_iou


def main():
    parser = argparse.ArgumentParser(description='SwinSegNet Training')
    parser.add_argument('--data_root', type=str, default='../Kasir-SEG',
                       help='Path to Kasir-SEG dataset')
    parser.add_argument('--model_name', type=str, default='swin_tiny',
                       choices=['swin_tiny', 'swin_small', 'swin_base', 'swin_large'],
                       help='Swin model variant')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--deep_supervision', action='store_true',
                       help='Enable deep supervision')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained Swin Transformer weights')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights during training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建模型
    model = create_swin_segnet(
        model_name=args.model_name,
        ch=64,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
        deep_supervision=args.deep_supervision,
        input_size=args.img_size
    )
    model = model.to(device)

    print(f'Model: {args.model_name}')
    print(f'Pretrained: {args.pretrained}')
    print(f'Freeze Encoder: {args.freeze_encoder}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # 创建数据集
    train_dataset = KvasirSEGDataset(args.data_root, 'train', args.img_size)
    val_dataset = KvasirSEGDataset(args.data_root, 'validation', args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # 创建损失函数和优化器
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 创建TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'logs/{args.model_name}_{timestamp}'
    writer = SummaryWriter(log_dir)

    # 恢复训练
    start_epoch = 0
    best_iou = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        print(f'Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}')

    # 训练循环
    print('Starting training...')

    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_loss, val_iou = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )

        # 更新学习率
        scheduler.step()

        # 记录到TensorBoard
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/EpochIoU', train_iou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存检查点
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'args': args
        }

        # 保存最新检查点
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pth'))

        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pth'))
            print(f'  New best IoU: {best_iou:.4f} - Model saved!')

        print('-' * 50)

    print(f'Training completed! Best IoU: {best_iou:.4f}')
    writer.close()


if __name__ == '__main__':
    main()

