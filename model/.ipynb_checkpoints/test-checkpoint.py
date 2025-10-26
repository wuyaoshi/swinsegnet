import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from swin_backbone import create_swin_segnet

def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    # 加载检查点 - 设置weights_only=False来兼容旧版本保存的模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']

    # 创建模型
    model = create_swin_segnet(
        model_name=args.model_name,
        ch=64,
        pretrained=False,  # 测试时不需要预训练权重
        freeze_encoder=False,
        deep_supervision=args.deep_supervision,
        input_size=args.img_size
    )

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型加载成功!")
    print(f"最佳IoU: {checkpoint['best_iou']:.4f}")
    print(f"训练轮次: {checkpoint['epoch'] + 1}")

    return model, args
    
def preprocess_image(image_path, img_size=224):
    """预处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    return image_tensor, original_image

def postprocess_prediction(prediction, original_size):
    """后处理预测结果"""
    # 应用sigmoid激活函数
    prediction = torch.sigmoid(prediction)

    # 调整大小到原始图像尺寸
    prediction = F.interpolate(prediction, size=original_size, mode='bilinear', align_corners=False)

    # 转换为numpy数组
    prediction = prediction.squeeze().cpu().numpy()

    # 二值化
    prediction_binary = (prediction > 0.5).astype(np.uint8) * 255

    return prediction, prediction_binary

def visualize_results(original_image, ground_truth, prediction, prediction_binary, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # 真实掩码
    if ground_truth is not None:
        axes[0, 1].imshow(ground_truth, cmap='gray')
        axes[0, 1].set_title('真实掩码')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, '无真实掩码', ha='center', va='center')
        axes[0, 1].set_title('真实掩码')
        axes[0, 1].axis('off')

    # 预测概率图
    axes[1, 0].imshow(prediction, cmap='hot')
    axes[1, 0].set_title('预测概率图')
    axes[1, 0].axis('off')

    # 二值化预测结果
    axes[1, 1].imshow(prediction_binary, cmap='gray')
    axes[1, 1].set_title('二值化预测结果')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")

    plt.show()

def calculate_metrics(pred_binary, gt_mask):
    """计算评估指标"""
    if gt_mask is None:
        return None

    # 转换为二进制掩码
    pred_binary = (pred_binary > 127).astype(np.uint8)
    gt_binary = (np.array(gt_mask) > 127).astype(np.uint8)

    # 计算交并比(IoU)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union

    # 计算Dice系数
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0

    return {'IoU': iou, 'Dice': dice}

def main():
    # 配置
    checkpoint_path = 'checkpoints/best.pth'
    data_root = '../Kasir-SEG'  # 根据你的main.py中的默认路径
    device = 'cuda'  # 强制使用CPU

    print("开始测试分割模型...")
    print(f"使用设备: {device}")

    # 加载模型
    try:
        model, args = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 获取第一个训练图像
    train_images_dir = os.path.join(data_root, 'train', 'images')
    train_masks_dir = os.path.join(data_root, 'train', 'masks')

    if not os.path.exists(train_images_dir):
        print(f"训练数据目录不存在: {train_images_dir}")
        return

    # 获取第一个图像文件
    image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
    if not image_files:
        print("没有找到训练图像文件")
        return

    first_image_name = image_files[0]
    image_path = os.path.join(train_images_dir, first_image_name)
    mask_path = os.path.join(train_masks_dir, first_image_name)

    print(f"测试图像: {first_image_name}")

    # 预处理图像
    try:
        image_tensor, original_image = preprocess_image(image_path, args.img_size)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return

    # 加载真实掩码（如果存在）
    ground_truth = None
    if os.path.exists(mask_path):
        ground_truth = Image.open(mask_path).convert('L')
        ground_truth = ground_truth.resize(original_image.size)

    # 模型推理
    print("开始推理...")
    try:
        with torch.no_grad():
            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                prediction, _ = model(image_tensor)
            else:
                prediction = model(image_tensor)

        print("推理完成!")
    except Exception as e:
        print(f"模型推理失败: {e}")
        return

    # 后处理
    try:
        prediction_prob, prediction_binary = postprocess_prediction(
            prediction, original_image.size[::-1]  # PIL size是(width, height)
        )
    except Exception as e:
        print(f"后处理失败: {e}")
        return

    # 计算评估指标
    if ground_truth is not None:
        metrics = calculate_metrics(prediction_binary, ground_truth)
        if metrics:
            print(f"评估指标:")
            print(f"  IoU: {metrics['IoU']:.4f}")
            print(f"  Dice: {metrics['Dice']:.4f}")

    # 可视化结果
    try:
        save_path = f"test_result_{first_image_name.replace('.jpg', '.png')}"
        visualize_results(
            original_image,
            ground_truth,
            prediction_prob,
            prediction_binary,
            save_path
        )
    except Exception as e:
        print(f"可视化失败: {e}")
        print("但推理已完成，模型可以正常工作")


def run_inference(image_path: str, result_path: str, checkpoint_path='checkpoints/best.pth', device='cuda'):

    model, args = load_model(checkpoint_path, device)

        # 预处理
    image_tensor, original_image = preprocess_image(image_path, args.img_size)
    image_tensor = image_tensor.to(device)

        # 推理
    with torch.no_grad():
        if hasattr(model, 'deep_supervision') and model.deep_supervision:
            prediction, _ = model(image_tensor)
        else:
            prediction = model(image_tensor)

        # 后处理
    prediction_prob, prediction_binary = postprocess_prediction(prediction, original_image.size[::-1])

        # 保存结果（二值化 mask）
    Image.fromarray(prediction_binary).save(result_path)

    return result_path



if __name__ == '__main__':
    main()
    
