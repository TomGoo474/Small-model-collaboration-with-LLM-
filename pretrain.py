import torch
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm

from configs import get_args
from dataset import load_and_preprocess_data, SeedSpectralDataset, DataLoader
from models.ts_vfc import TS_VFC
from models.encoder import base_Model
from loss import NTXentLoss
from utils import save_checkpoint

def set_seed(seed=42):
    """Set all random seeds for deterministic results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def pretrain():
    # Set CuBLAS workspace config for deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Fix CuBLAS non-determinism

    # 1. Initialize
    configs = get_args()
    seed = configs.seed if hasattr(configs, 'seed') else 42
    set_seed(seed)

    device = torch.device(configs.device)

    # 2. Data loading
    train_data_df, _, global_mean, global_std = load_and_preprocess_data(seed=seed)
    train_dataset_pretrain = SeedSpectralDataset(
        train_data_df,
        augmentation_mode='pretrain',
        mean=global_mean,
        std=global_std,
        seed=seed
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset_pretrain,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Single-threaded for determinism
        generator=g
    )

    # 3. Model initialization
    model = TS_VFC(configs, device).to(device)
    print(f"Self-supervised Pre-training Model (TS_VFC):\n{model}")

    # 4. Loss and optimizer
    criterion = NTXentLoss(device, configs.batch_size, configs.use_cosine_similarity)
  

    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)

    # 5. Training loop
    print("Starting self-supervised pre-training...")
    for epoch in range(1, configs.pretrain_epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{configs.pretrain_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            spectra_weak = batch['spectra_weak'].to(device)  # (batch_size, 1, 300)
            spectra_strong = batch['spectra_strong'].to(device)  # (batch_size, 1, 300)

            zis, zjs = model(spectra_weak, spectra_strong)
            loss = criterion(zis, zjs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % configs.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average Pre-train Loss: {avg_loss:.6f}")

        if epoch % 10 == 0 or epoch == configs.pretrain_epochs:
            save_checkpoint(model, optimizer, epoch, configs.save_dir, 'self_supervised_ts_vfc')

    print("Pre-training finished.")

if __name__ == '__main__':
    pretrain()