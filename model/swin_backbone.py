import math

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer
import timm
from collections import OrderedDict
import warnings


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True,
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(init_channels),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            nn.BatchNorm2d(new_channels),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]


class SwinSegNet(nn.Module):
    """
    基于 Swin Transformer 的分割网络，类似于 EffiSegNet 的架构
    """

    # 预定义的Swin Transformer配置
    MODEL_CONFIGS = {
        'swin_tiny': {
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': (7, 7),
            'mlp_ratio': 4.0,
        },
        'swin_small': {
            'embed_dim': 96,
            'depths': (2, 2, 18, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': (7, 7),
            'mlp_ratio': 4.0,
        },
        'swin_base': {
            'embed_dim': 128,
            'depths': (2, 2, 18, 2),
            'num_heads': (4, 8, 16, 32),
            'window_size': (7, 7),
            'mlp_ratio': 4.0,
        },
        'swin_large': {
            'embed_dim': 192,
            'depths': (2, 2, 18, 2),
            'num_heads': (6, 12, 24, 48),
            'window_size': (7, 7),
            'mlp_ratio': 4.0,
        },
        'custom': {
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
            'window_size': (7, 7),
            'mlp_ratio': 4.0,
        }
    }

    def __init__(
        self,
        ch=64,
        pretrained=False,
        freeze_encoder=False,
        deep_supervision=False,
        model_name="swin_tiny",
        input_size=224,
        in_chans=3,
        custom_config=None,
        **kwargs
    ):
        super(SwinSegNet, self).__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.deep_supervision = deep_supervision

        # 获取配置
        if model_name == 'custom' and custom_config is not None:
            config = custom_config
        else:
            config = self.MODEL_CONFIGS[model_name].copy()

        # 更新配置
        config.update(kwargs)
        self.config = config

        # 创建 Swin Transformer 编码器
        self.encoder = SwinTransformer(
            in_chans=in_chans,
            embed_dim=config['embed_dim'],
            window_size=config['window_size'],
            patch_size=(2, 2),  # 使用2x2 patch size以获得更大的特征图
            depths=config['depths'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_checkpoint=False,
            spatial_dims=2
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 获取各个stage的通道数
        embed_dim = config['embed_dim']
        channels_per_output = [
            embed_dim,       # Stage 0: 112x112
            embed_dim * 2,   # Stage 1: 56x56
            embed_dim * 4,   # Stage 2: 28x28
            embed_dim * 8,   # Stage 3: 14x14
            embed_dim * 16   # Stage 4: 7x7
        ]

        # 上采样层 - 都上采样到输入尺寸
        self.up1 = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)
        self.up3 = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)
        self.up5 = nn.Upsample(size=input_size, mode="bilinear", align_corners=False)

        # 通道调整卷积层
        self.conv1 = nn.Conv2d(
            channels_per_output[0], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(
            channels_per_output[1], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(ch)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(ch)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(ch)

        self.relu = nn.ReLU(inplace=True)

        # 深度监督分支
        if self.deep_supervision:
            self.conv7 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn7 = nn.BatchNorm2d(ch)
            self.conv8 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn8 = nn.BatchNorm2d(ch)
            self.conv9 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn9 = nn.BatchNorm2d(ch)
            self.conv10 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn10 = nn.BatchNorm2d(ch)
            self.conv11 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn11 = nn.BatchNorm2d(ch)

        # 最终层
        self.bn6 = nn.BatchNorm2d(ch)
        self.ghost1 = GhostModule(ch, ch)
        self.ghost2 = GhostModule(ch, ch)
        self.conv6 = nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # 预训练权重加载
        if pretrained:
            self.load_pretrained_weights(model_name)

    def forward(self, x):
        # 通过Swin Transformer编码器获取多尺度特征
        features = self.encoder(x, normalize=True)

        # 如果返回的特征数量不是5个，需要处理
        if len(features) == 4:
            # 如果只有4个特征，复制最后一个作为第5个
            x0, x1, x2, x3 = features
            x4 = x3  # 使用相同的特征
        else:
            x0, x1, x2, x3, x4 = features

        # 通道调整和激活
        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.bn1(x0)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)

        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        # 上采样到相同尺寸
        x0 = self.up1(x0)
        x1 = self.up2(x1)
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)

        # 特征融合
        x = x0 + x1 + x2 + x3 + x4
        x = self.bn6(x)
        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.conv6(x)

        # 深度监督输出
        if self.deep_supervision:
            x0 = self.bn7(x0)
            x0 = self.conv7(x0)

            x1 = self.bn8(x1)
            x1 = self.conv8(x1)

            x2 = self.bn9(x2)
            x2 = self.conv9(x2)

            x3 = self.bn10(x3)
            x3 = self.conv10(x3)

            x4 = self.bn11(x4)
            x4 = self.conv11(x4)

            return x, [x0, x1, x2, x3, x4]

        return x

    def get_model_info(self):
        """获取模型配置信息"""
        embed_dim = self.config['embed_dim']
        output_shapes = []

        # 根据实际可能的输出层数动态生成形状信息
        expected_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8, embed_dim*16]
        spatial_sizes = [self.input_size//2, self.input_size//4, self.input_size//8, self.input_size//16, self.input_size//32]

        for i, (ch, size) in enumerate(zip(expected_channels, spatial_sizes)):
            output_shapes.append(f'[B, {ch}, {size}, {size}]')

        return {
            'config': self.config,
            'input_size': self.input_size,
            'embed_dim': embed_dim,
            'output_channels': expected_channels,
            'output_shapes': output_shapes
        }

    def load_pretrained_weights(self, model_name):
        """加载预训练权重"""
        try:
            # 获取对应的timm模型名称
            timm_model_name = self.get_timm_model_name(model_name)
            print(f"正在加载预训练模型: {timm_model_name}")

            # 加载预训练模型
            pretrained_model = timm.create_model(timm_model_name, pretrained=True)
            pretrained_state_dict = pretrained_model.state_dict()

            # 获取当前模型的encoder state dict
            current_encoder_dict = self.encoder.state_dict()

            # 尝试匹配和加载权重
            matched_keys = 0
            total_keys = len(current_encoder_dict)

            for key in current_encoder_dict.keys():
                # 尝试不同的key映射策略
                possible_keys = [
                    key,  # 直接匹配
                    f"features.{key}",  # timm模型通常使用features前缀
                    f"backbone.{key}",  # 一些模型使用backbone前缀
                ]

                for possible_key in possible_keys:
                    if possible_key in pretrained_state_dict:
                        if current_encoder_dict[key].shape == pretrained_state_dict[possible_key].shape:
                            current_encoder_dict[key] = pretrained_state_dict[possible_key]
                            matched_keys += 1
                            break

            # 加载匹配的权重到encoder
            self.encoder.load_state_dict(current_encoder_dict, strict=True)

            print(f"成功加载 {model_name} 的预训练权重: {matched_keys}/{total_keys} 层匹配")

            if matched_keys == 0:
                print("警告: 没有匹配的预训练权重层，将使用随机初始化的权重")

        except Exception as e:
            print(f"加载预训练权重时发生错误: {e}")
            print("将使用随机初始化的权重")

    def get_timm_model_name(self, model_name):
        """根据模型名称获取timm库中对应的模型名称"""
        model_name_mapping = {
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224',
            'swin_large': 'swin_large_patch4_window7_224',
        }

        return model_name_mapping.get(model_name, model_name)


def create_swin_segnet(model_name='swin_tiny', **kwargs):
    """便捷函数：创建Swin分割网络"""
    return SwinSegNet(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # 测试不同配置的模型
    model_names = ['swin_tiny', 'swin_small', 'swin_base']

    for model_name in model_names:
        print(f"\n=== 测试 {model_name.upper()} ===")

        # 创建模型
        model = create_swin_segnet(
            model_name=model_name,
            ch=64,
            pretrained=False,
            freeze_encoder=False,
            deep_supervision=True,
            input_size=224
        )
        model.eval()

        # 打印模型信息
        info = model.get_model_info()
        print("配置:", info['config'])
        print("预期编码器输出形状:", info['output_shapes'])

        # 测试前向传播
        with torch.no_grad():
            x = torch.randn(8, 3, 224, 224)  # batch_size=8
            result = model(x)

            if isinstance(result, tuple):
                output, aux_outputs = result
                print("最终输出形状:", output.shape)
                print("辅助输出形状:", [aux.shape for aux in aux_outputs])
            else:
                output = result
                print("最终输出形状:", output.shape)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params:,}")

    # 测试自定义配置
    print(f"\n=== 测试自定义配置 ===")
    custom_config = {
        'embed_dim': 64,
        'depths': (2, 2, 4, 2),
        'num_heads': (2, 4, 8, 16),
        'window_size': (7, 7),
        'mlp_ratio': 4.0,
    }

    custom_model = create_swin_segnet(
        model_name='custom',
        custom_config=custom_config,
        ch=64,
        deep_supervision=False,
        input_size=224
    )
    custom_model.eval()

    with torch.no_grad():
        x = torch.randn(8, 3, 224, 224)
        output = custom_model(x)

        print("自定义模型输出形状:", output.shape)

