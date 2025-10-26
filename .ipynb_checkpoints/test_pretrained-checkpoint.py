#!/usr/bin/env python3
"""
测试预训练Swin Transformer模型加载功能
"""

import torch
import sys
import os
sys.path.append('model')

from model.swin_backbone import create_swin_segnet

def test_pretrained_loading():
    """测试预训练权重加载功能"""
    print("=== 测试预训练Swin Transformer模型加载 ===\n")

    # 测试不同的模型配置
    models_to_test = ['swin_tiny', 'swin_small', 'swin_base']

    for model_name in models_to_test:
        print(f"测试 {model_name.upper()} 模型:")
        print("-" * 40)

        try:
            # 创建不使用预训练权重的模型
            print("1. 创建不使用预训练权重的模型...")
            model_no_pretrain = create_swin_segnet(
                model_name=model_name,
                pretrained=False,
                deep_supervision=False,
                input_size=224
            )

            # 创建使用预训练权重的模型
            print("2. 创建使用预训练权重的模型...")
            model_pretrained = create_swin_segnet(
                model_name=model_name,
                pretrained=True,  # 启用预训练权重
                deep_supervision=False,
                input_size=224
            )

            # 比较参数数量
            params_no_pretrain = sum(p.numel() for p in model_no_pretrain.parameters())
            params_pretrained = sum(p.numel() for p in model_pretrained.parameters())

            print(f"   无预训练权重参数量: {params_no_pretrain:,}")
            print(f"   有预训练权重参数量: {params_pretrained:,}")

            # 测试前向传播
            print("3. 测试前向传播...")
            with torch.no_grad():
                x = torch.randn(2, 3, 224, 224)

                output_no_pretrain = model_no_pretrain(x)
                output_pretrained = model_pretrained(x)

                print(f"   无预训练输出形状: {output_no_pretrain.shape}")
                print(f"   有预训练输出形状: {output_pretrained.shape}")

                # 检查输出是否不同
                diff = torch.mean(torch.abs(output_no_pretrain - output_pretrained))
                print(f"   输出差异 (平均绝对差): {diff:.6f}")

                if diff > 1e-6:
                    print("   ✓ 预训练权重加载成功 (输出有差异)")
                else:
                    print("   ⚠ 预训练权重可能未正确加载 (输出无差异)")

            print("   ✓ 测试通过\n")

        except Exception as e:
            print(f"   ✗ 测试失败: {str(e)}\n")
            continue

def test_freeze_encoder():
    """测试编码器冻结功能"""
    print("=== 测试编码器冻结功能 ===\n")

    try:
        # 创建冻结编码器的模型
        print("1. 创建冻结编码器的模型...")
        model_frozen = create_swin_segnet(
            model_name='swin_tiny',
            pretrained=True,
            freeze_encoder=True,
            deep_supervision=False,
            input_size=224
        )

        # 创建不冻结编码器的模型
        print("2. 创建不冻结编码器的模型...")
        model_trainable = create_swin_segnet(
            model_name='swin_tiny',
            pretrained=True,
            freeze_encoder=False,
            deep_supervision=False,
            input_size=224
        )

        # 计算可训练参数
        trainable_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        trainable_normal = sum(p.numel() for p in model_trainable.parameters() if p.requires_grad)

        total_params = sum(p.numel() for p in model_frozen.parameters())

        print(f"   总参数量: {total_params:,}")
        print(f"   冻结编码器可训练参数: {trainable_frozen:,}")
        print(f"   不冻结编码器可训练参数: {trainable_normal:,}")

        if trainable_frozen < trainable_normal:
            print("   ✓ 编码器冻结功能正常工作")
        else:
            print("   ⚠ 编码器冻结可能未正确实现")

    except Exception as e:
        print(f"   ✗ 测试失败: {str(e)}")

def test_deep_supervision():
    """测试深度监督功能"""
    print("\n=== 测试深度监督功能 ===\n")

    try:
        # 创建带深度监督的模型
        print("1. 创建带深度监督的模型...")
        model_ds = create_swin_segnet(
            model_name='swin_tiny',
            pretrained=False,
            deep_supervision=True,
            input_size=224
        )

        # 创建不带深度监督的模型
        print("2. 创建不带深度监督的模型...")
        model_no_ds = create_swin_segnet(
            model_name='swin_tiny',
            pretrained=False,
            deep_supervision=False,
            input_size=224
        )

        # 测试输出
        print("3. 测试输出格式...")
        with torch.no_grad():
            x = torch.randn(2, 3, 224, 224)

            output_ds = model_ds(x)
            output_no_ds = model_no_ds(x)

            print(f"   深度监督输出类型: {type(output_ds)}")
            print(f"   无深度监督输出类型: {type(output_no_ds)}")

            if isinstance(output_ds, tuple):
                main_out, aux_outs = output_ds
                print(f"   主输出形状: {main_out.shape}")
                print(f"   辅助输出数量: {len(aux_outs)}")
                print(f"   辅助输出形状: {[aux.shape for aux in aux_outs]}")
                print("   ✓ 深度监督功能正常工作")
            else:
                print("   ⚠ 深度监督可能未正确实现")

    except Exception as e:
        print(f"   ✗ 测试失败: {str(e)}")

if __name__ == "__main__":
    # 运行所有测试
    test_pretrained_loading()
    test_freeze_encoder()
    test_deep_supervision()

    print("=== 测试完成 ===")
