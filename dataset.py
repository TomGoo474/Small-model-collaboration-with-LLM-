import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


# --- 数据加载函数 ---
def load_and_preprocess_data(seed=42):
    """
    Load and preprocess spectral data with fixed seed for determinism.

    Args:
        seed (int): Seed for random operations (e.g., train-test split).

    Returns:
        train_data_df, test_data_df, global_mean, global_std
    """
    # Fix seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    try:
        train_spectra_data = pd.read_excel('Train_Spectra_svi.xlsx')
        test_spectra_data = pd.read_excel('Test_Spectra_svi.xlsx')
    except FileNotFoundError as e:
        print(
            f"Error: Data file not found! Please ensure 'Train_Spectra_ttc.xlsx' and 'Test_Spectra_ttc.xlsx' are in the same directory. {e}")
        exit(1)

    train_spectra_data = train_spectra_data.set_index('ID')
    test_spectra_data = test_spectra_data.set_index('ID')

    for spectra_df in [train_spectra_data, test_spectra_data]:
        if 'Class' in spectra_df.columns:
            if spectra_df['Class'].dtype == object:
                if spectra_df['Class'].astype(str).str.match(r'^\d+$').all():
                    spectra_df['Class'] = spectra_df['Class'].astype(int)
                else:
                    class_mapping = {val: idx for idx, val in enumerate(sorted(spectra_df['Class'].unique()))}
                    spectra_df['Class'] = spectra_df['Class'].map(class_mapping)
                    print(f"Class mapping for {spectra_df.columns[0]}:", class_mapping)
            else:
                spectra_df['Class'] = spectra_df['Class'].astype(int)
        else:
            print("Warning: 'Class' column not found in data. Ensure your data has a 'Class' column for labels.")

    # Compute global mean and std for training data
    all_spectra_values = train_spectra_data.iloc[:, 1:].values
    global_mean = all_spectra_values.mean()
    global_std = all_spectra_values.std()
    if global_std == 0:
        global_std = 1.0  # Avoid division by zero

    print(f"Data loading complete. Train samples: {len(train_spectra_data)}, Test samples: {len(test_spectra_data)}")
    print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")

    return train_spectra_data, test_spectra_data, global_mean, global_std


class SeedSpectralDataset(Dataset):
    def __init__(self, spectra_data, augmentation_mode='none', mean=None, std=None, seed=None):
        """
        Dataset for spectral data with deterministic augmentations.

        Args:
            spectra_data (pd.DataFrame): DataFrame with 'Class' and spectral columns.
            augmentation_mode (str): 'pretrain' for self-supervised, 'finetune' or 'none' for supervised.
            mean (float): Global mean for standardization.
            std (float): Global standard deviation for standardization.
            seed (int): Seed for random augmentations.
        """
        self.spectra_values = spectra_data.iloc[:, 1:].values.astype(np.float32)  # Spectra from second column
        self.labels = torch.LongTensor(spectra_data['Class'].values)  # Labels
        self.seed = seed
        self.rng = torch.Generator()  # PyTorch Generator for random operations
        if seed is not None:
            self.rng.manual_seed(seed)  # Fix seed for reproducibility

        # Standardize
        if mean is not None and std is not None:
            self.spectra_raw_normalized = torch.FloatTensor((self.spectra_values - mean) / (std + 1e-8))
        else:
            local_mean = self.spectra_values.mean()
            local_std = self.spectra_values.std()
            if local_std == 0:
                local_std = 1.0
            self.spectra_raw_normalized = torch.FloatTensor((self.spectra_values - local_mean) / local_std)

        self.augmentation_mode = augmentation_mode
        self.sequence_length = self.spectra_raw_normalized.shape[1]

        print(f"Dataset mode: {augmentation_mode}, Spectra shape: {self.spectra_raw_normalized.shape}, "
              f"Min: {self.spectra_raw_normalized.min():.4f}, Max: {self.spectra_raw_normalized.max():.4f}")

    def __len__(self):
        return len(self.spectra_raw_normalized)

    def _random_masking(self, spectrum, mask_ratio=0.2):
        """
        Randomly mask a percentage of spectral points.

        Args:
            spectrum (torch.Tensor): Input spectrum, shape (sequence_length,)
            mask_ratio (float): Percentage of points to mask.

        Returns:
            torch.Tensor: Masked spectrum.
        """
        masked_spectrum = spectrum.clone()
        num_mask_points = int(self.sequence_length * mask_ratio)
        if num_mask_points == 0 and mask_ratio > 0:
            num_mask_points = 1

        # Use torch.randperm with fixed generator
        mask_indices = torch.randperm(self.sequence_length, generator=self.rng)[:num_mask_points]
        masked_spectrum[mask_indices] = 0.0
        return masked_spectrum

    def _augment(self, spectrum, mode):
        """
        Apply augmentation to spectrum.

        Args:
            spectrum (torch.Tensor): Input spectrum, shape (sequence_length,)
            mode (str): 'weak', 'strong', or 'none'.

        Returns:
            torch.Tensor: Augmented spectrum.
        """
        if mode == 'weak':
            # Weak augmentation: Gaussian noise
            noise = torch.randn(spectrum.shape, generator=self.rng) * 0.3
            return spectrum + noise
        elif mode == 'strong':
            # Strong augmentation: Random masking
            return self._random_masking(spectrum, mask_ratio=0.3)
        else:
            return spectrum

    def __getitem__(self, idx):
        # Set seed for this specific index to ensure reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)  # Per-sample seed
            self.rng.manual_seed(self.seed + idx)

        spectrum = self.spectra_raw_normalized[idx]
        label = self.labels[idx]

        if self.augmentation_mode == 'pretrain':
            return {
                'spectra_weak': self._augment(spectrum, 'weak').unsqueeze(0),  # (1, sequence_length)
                'spectra_strong': self._augment(spectrum, 'strong').unsqueeze(0),  # (1, sequence_length)
                'label': label
            }
        else:
            return {
                'spectra': spectrum.unsqueeze(0),  # (1, sequence_length)
                'label': label
            }


# --- 数据加载测试 ---
if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    train_data_df, test_data_df, global_mean, global_std = load_and_preprocess_data(seed=seed)

    # Pretrain dataset
    print("\n--- Testing Pretrain Mode ---")
    train_dataset_pretrain = SeedSpectralDataset(
        train_data_df,
        augmentation_mode='pretrain',
        mean=global_mean,
        std=global_std,
        seed=seed
    )
    train_loader_pretrain = DataLoader(
        train_dataset_pretrain,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    for i, batch in enumerate(train_loader_pretrain):
        print(f"Pretrain Batch {i}:")
        print(f"  Spectra Weak shape: {batch['spectra_weak'].shape}")
        print(f"  Spectra Strong shape: {batch['spectra_strong'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Sample 0 Weak (first 10): {batch['spectra_weak'][0, 0, :10].tolist()}")
        print(f"  Sample 0 Strong (first 10): {batch['spectra_strong'][0, 0, :10].tolist()}")
        break

    # Finetune dataset
    print("\n--- Testing Finetune Mode ---")
    train_dataset_finetune = SeedSpectralDataset(
        train_data_df,
        augmentation_mode='finetune',
        mean=global_mean,
        std=global_std,
        seed=seed
    )
    train_loader_finetune = DataLoader(
        train_dataset_finetune,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    for i, batch in enumerate(train_loader_finetune):
        print(f"Finetune Batch {i}:")
        print(f"  Spectra shape: {batch['spectra'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Spectra: {batch['spectra'][0, 0, :10].tolist()}")
        break