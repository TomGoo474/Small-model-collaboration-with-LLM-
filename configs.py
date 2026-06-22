# configs.py

import argparse
import torch.nn as nn
import torch

def get_encoder_output_dimensions(input_length, input_channels, kernel_size, stride, dropout, final_out_channels):
    """
    Helper function to calculate the output dimensions of the base_Model (encoder).
    This is crucial for determining the input size to the final linear layer (logits)
    and the sequence length for the transformer.
    """

    # Create a dummy encoder to trace dimensions
    class DummyEncoder(nn.Module):
        def __init__(self, input_channels, kernel_size, stride, dropout, final_out_channels):
            super().__init__()
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                          stride=stride, bias=False, padding=(kernel_size // 2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(dropout)
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=4, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
            self.conv_block3 = nn.Sequential(
                nn.Conv1d(64, final_out_channels, kernel_size=4, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            )

        def forward(self, x):
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return x

    dummy_input = torch.randn(1, input_channels, input_length)
    dummy_encoder = DummyEncoder(input_channels, kernel_size, stride, dropout, final_out_channels)
    with torch.no_grad():
        output = dummy_encoder(dummy_input)

    # Calculate flattened dimension for logits
    flattened_dim = output.numel() // output.shape[0]  # output.numel() is total elements, output.shape[0] is batch_size

    # Calculate sequence length for transformer (which is the last dim of output, if it's (Batch, Channels, Length))
    features_len = output.shape[2]  # Length dimension after convolutions/pooling

    return flattened_dim, features_len


def get_args():
    parser = argparse.ArgumentParser(description='Self-supervised learning for Seed Spectra Classification')

    # Data arguments
    parser.add_argument('--input_channels', type=int, default=1, help='Input channels for spectra (1 for 1D spectra)')
    parser.add_argument('--sequence_length', type=int, default=300, help='Length of the spectral sequence')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for seed spectra classification')

    # Encoder (base_Model) arguments - these will be used to calculate dimensions
    parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size for Conv1d in encoder')
    parser.add_argument('--stride', type=int, default=1, help='Stride for Conv1d in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in encoder')
    parser.add_argument('--final_out_channels', type=int, default=128,
                        help='Output channels of the last conv block in encoder')

    # Parse args temporarily to calculate dynamic parameters
    temp_args, _ = parser.parse_known_args()

    # Calculate encoder output dimensions
    flattened_dim, features_len = get_encoder_output_dimensions(
        temp_args.sequence_length,
        temp_args.input_channels,
        temp_args.kernel_size,
        temp_args.stride,
        temp_args.dropout,
        temp_args.final_out_channels
    )
    parser.add_argument('--encoder_flattened_dim', type=int, default=flattened_dim,
                        help='Calculated flattened output dimension of encoder')
    parser.add_argument('--features_len', type=int, default=features_len,
                        help='Calculated sequence length after encoder (input to Transformer)')

    # Transformer (Seq_Transformer) arguments
    # patch_size for Seq_Transformer will be final_out_channels as each time step is a 'patch' of features
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Embedding dimension for Transformer')  # dim in Seq_Transformer
    parser.add_argument('--depth', type=int, default=4, help='Number of Transformer blocks')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=64,
                        help='Hidden dimension for FeedForward in Transformer')  # mlp_dim in Seq_Transformer

    # TS_VFC arguments (specific to the cross-view fusion)
    parser.add_argument('--timesteps', type=int, default=10,
                        help='Number of timesteps to sample in TS_VFC')  # Corresponds to `timestep` in TS_VFC
    # The output dimension of the projection head. Used as output of TS_VFC for NTXentLoss
    parser.add_argument('--projection_output_dim', type=int, default=256,
                        help='Output dimension of the projection head in TS_VFC')

    # Loss (NTXentLoss) arguments
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for NTXentLoss')
    parser.add_argument('--use_cosine_similarity', type=bool, default=True, help='Use cosine similarity in NTXentLoss')

    # Training arguments
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs for pre-training')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How many batches to wait before logging training status')
    parser.add_argument('--save_dir', type=str, default='checkpoints1', help='Directory to save model checkpoints')

    return parser.parse_args()


# Example of how to use this helper to verify:
if __name__ == '__main__':
    args = get_args()
    print(f"Calculated encoder_flattened_dim: {args.encoder_flattened_dim}")
    print(f"Calculated features_len: {args.features_len}")
    # Expected output for 1x300, k=8, s=1, final_out_channels=128:
    # features_len should be 42 (from previous manual calculation)
    # flattened_dim should be 128 * 42 = 5376
    # If your encoder.py uses 1152, then this suggests a discrepancy.
    # The `base_Model` in `encoder.py` needs its `nn.Linear(1152, ...)` updated
    # to `nn.Linear(configs.encoder_flattened_dim, ...)`