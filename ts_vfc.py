# models/ts_vfc.py (Modified to be the full self-supervised model)

import torch
import torch.nn as nn
import numpy as np
from .transformer import Seq_Transformer # Import Seq_Transformer
from .encoder import base_Model       # Import base_Model (encoder)

class TS_VFC(nn.Module):
    def __init__(self, configs, device):
        super(TS_VFC, self).__init__()
        self.configs = configs
        self.device = device

        # Instantiate the encoder (base_Model)
        self.encoder = base_Model(configs)

        # Transformer for the weak view (or primary path)
        self.seq_transformer = Seq_Transformer(
            patch_size=configs.final_out_channels, # This is the feature dimension from encoder
            dim=configs.hidden_dim,
            depth=configs.depth,
            heads=configs.heads,
            mlp_dim=configs.mlp_dim,
            channels=1 # Assuming input to transformer is (batch, sequence_length, feature_channels)
        )

        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=configs.final_out_channels, out_channels=128, kernel_size=1), # 1x1 conv to reduce features
            nn.AdaptiveAvgPool1d(1), # Pool across length dimension to get (batch, 128, 1)
            nn.Flatten() # (batch, 128)
        )
  
        self.projection_head = nn.Sequential(
            nn.Linear(configs.hidden_dim + 128, configs.projection_output_dim // 2),
            nn.BatchNorm1d(configs.projection_output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.projection_output_dim // 2, configs.projection_output_dim),
        )

    def forward(self, x_weak, x_strong):
       
        _, features_weak = self.encoder(x_weak) # (batch, final_out_channels, features_len)
        _, features_strong = self.encoder(x_strong) # (batch, final_out_channels, features_len)

       
        features_weak_T = features_weak.transpose(1, 2) # (batch, features_len, final_out_channels)
        v_k_weak = self.seq_transformer(features_weak_T) # (batch, hidden_dim)

        v_k_strong_processed = self.conv_block(features_strong) # (batch, 128)

        # 4. Cross-view Fusion and Projection
        # Concatenate features from weak view (transformer) and strong view (conv_block)
        fused_features = torch.cat((v_k_weak, v_k_strong_processed), dim=1) # (batch, hidden_dim + 128)

        # Apply the projection head
        projected_output = self.projection_head(fused_features) # (batch, projection_output_dim)

       
        z_i = projected_output # This is the result of weak_transformer + strong_conv_block

        
        v_k_strong_via_transformer = self.seq_transformer(features_strong.transpose(1, 2)) # (batch, hidden_dim)
        v_k_weak_via_conv = self.conv_block(features_weak) # (batch, 128)
        fused_features_swapped = torch.cat((v_k_strong_via_transformer, v_k_weak_via_conv), dim=1)
        z_j = self.projection_head(fused_features_swapped)

        # Return the two fused & projected features for NTXentLoss
        return z_i, z_j # (batch, projection_output_dim), (batch, projection_output_dim)