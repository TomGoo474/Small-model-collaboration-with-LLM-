import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, length = x.size()
        channels_per_group = channels // self.groups
        # Reshape: [batch, channels, length] -> [batch, groups, channels_per_group, length]
        x = x.view(batch, self.groups, channels_per_group, length)
        # Transpose: [batch, groups, channels_per_group, length] -> [batch, channels_per_group, groups, length]
        x = x.transpose(1, 2).contiguous()
        # Flatten: [batch, channels_per_group * groups, length] -> [batch, channels, length]
        x = x.view(batch, channels, length)
        return x

class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=4):
        super(ShuffleBlock, self).__init__()
        # Multi-branch structure (RepVGG-like)
        self.branches = nn.ModuleList([
            # Main branch: Grouped convolutions with Channel Shuffle
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
                nn.BatchNorm1d(out_channels),
                ChannelShuffle(groups),
            ),
            # 1x1 branch for additional feature mixing
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        ])

        # Shortcut (identity or projection)
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential(
                nn.Identity(),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = sum(branch(x) for branch in self.branches)
        shortcut = self.shortcut(x)
        out = out + shortcut
        return self.relu(out)

    def fuse_for_inference(self):
        in_channels = self.branches[0][0].in_channels
        out_channels = self.branches[0][4].out_channels
        stride = self.branches[0][0].stride[0]

        fused_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        fused_bn = nn.BatchNorm1d(out_channels)
        fused_weights = 0
        fused_bias = 0

        for branch in [self.branches[0], self.branches[1], self.shortcut]:
            if isinstance(branch[0], nn.Identity):
                identity_kernel = torch.eye(out_channels, in_channels).unsqueeze(-1).to(branch[1].weight.device)
                kernel_padded = F.pad(identity_kernel, (1, 1), mode='constant', value=0)
                weight_bn_scale = (branch[1].weight / torch.sqrt(branch[1].running_var + branch[1].eps)).view(-1, 1, 1)
                fused_weights += kernel_padded * weight_bn_scale
                fused_bias += (branch[1].bias - branch[1].running_mean * weight_bn_scale.squeeze()).view(-1)
            elif len(branch) > 0:
                conv = branch[0]
                bn = branch[1]
                weight_bn_scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1)
                kernel = conv.weight if conv.kernel_size[0] == 3 else F.pad(conv.weight, (1, 1), mode='constant', value=0)
                fused_weights += kernel * weight_bn_scale
                fused_bias += (bn.bias - bn.running_mean * weight_bn_scale.squeeze()).view(-1)

        fused_conv.weight.data.copy_(fused_weights)
        fused_conv.bias.data.copy_(fused_bias)
        return nn.Sequential(fused_conv, fused_bn, self.relu)

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.in_channels = 64  # Keep original initial channels
        self.initial_conv = nn.Conv1d(configs.input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 layers with ShuffleBlocks
        self.layer1 = self._make_layer(64, 2, stride=1, groups=4)
        self.layer2 = self._make_layer(128, 2, stride=2, groups=4)
        self.layer3 = self._make_layer(256, 2, stride=2, groups=4)
        final_out_channels = getattr(configs, 'final_out_channels', 128)
        self.layer4 = self._make_layer(final_out_channels, 2, stride=2, groups=4)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Compute encoder_flattened_dim dynamically
        seq_len = getattr(configs, 'seq_len', 300)
        with torch.no_grad():
            dummy_input = torch.zeros(1, configs.input_channels, seq_len)
            x = self._compute_features(dummy_input)
            configs.encoder_flattened_dim = x.reshape(1, -1).shape[1]

        # Logits layer
        self.logits = nn.Linear(configs.encoder_flattened_dim, configs.num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, groups):
        layers = []
        layers.append(ShuffleBlock(self.in_channels, out_channels, stride, groups))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ShuffleBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _compute_features(self, x_in):
        x = self.initial_conv(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

    def forward(self, x_in):
        x = self._compute_features(x_in)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

    def reparameterize(self):
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for j, block in enumerate(layer):
                if isinstance(block, ShuffleBlock):
                    layer[j] = block.fuse_for_inference()