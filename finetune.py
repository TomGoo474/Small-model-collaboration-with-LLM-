import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import numpy as np
import random
import time

from configs import get_args
from dataset import load_and_preprocess_data, SeedSpectralDataset, DataLoader
from models.encoder import base_Model
from models.ts_vfc import TS_VFC
from utils import save_checkpoint

def finetune(experiment_id=0, current_seed=42):
    configs = get_args()
    device = torch.device(configs.device)

    # Set random seeds
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    if configs.device == 'cuda':
        torch.cuda.manual_seed(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"--- 启动微调实验 {experiment_id + 1}/10，使用种子 {current_seed} (加载预训练权重) ---")

    # Data Loading
    train_data_df, test_data_df, global_mean, global_std = load_and_preprocess_data()
    train_dataset_finetune = SeedSpectralDataset(train_data_df, augmentation_mode='finetune', mean=global_mean, std=global_std)
    test_dataset_finetune = SeedSpectralDataset(test_data_df, augmentation_mode='finetune', mean=global_mean, std=global_std)
    train_loader = DataLoader(train_dataset_finetune, batch_size=configs.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_finetune, batch_size=configs.batch_size, shuffle=False)

    # Model Initialization
    model = base_Model(configs).to(device)
    print(f"微调监督模型 (实验 {experiment_id + 1}):\n{model}")

    # Load pre-trained weights
    pretrain_checkpoint_path = os.path.join(configs.save_dir, f'self_supervised_ts_vfc_epoch_{configs.pretrain_epochs}.pth')
    if os.path.exists(pretrain_checkpoint_path):
        temp_pretrain_model = TS_VFC(configs, device).to(device)
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            temp_pretrain_model.load_state_dict(checkpoint['model_state_dict'])
            model.load_state_dict(temp_pretrain_model.encoder.state_dict(), strict=False)
            print(f"为实验 {experiment_id + 1} 成功加载了预训练编码器权重，来自 {pretrain_checkpoint_path}")
        else:
            try:
                model.load_state_dict(checkpoint, strict=True)
                print(f"为实验 {experiment_id + 1} 成功加载了预训练编码器权重 (直接加载)，来自 {pretrain_checkpoint_path}")
            except RuntimeError as e:
                print(f"警告：直接加载预训练权重失败，尝试使用 strict=False。错误: {e}")
                model.load_state_dict(checkpoint, strict=False)
                print(f"为实验 {experiment_id + 1} 成功加载了部分预训练编码器权重 (strict=False)，来自 {pretrain_checkpoint_path}")
    else:
        print(f"未找到预训练模型：{pretrain_checkpoint_path}。实验 {experiment_id + 1} 将从头开始微调。")

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)

    # Training Loop
    print(f"开始对实验 {experiment_id + 1} 进行监督微调...")
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, configs.finetune_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"实验 {experiment_id + 1} - 周期 {epoch}/{configs.finetune_epochs} (训练)", leave=False)
        for batch_idx, batch in enumerate(pbar):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(spectra)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % configs.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"实验 {experiment_id + 1} - 周期 {epoch} 训练损失: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"实验 {experiment_id + 1} - 周期 {epoch}/{configs.finetune_epochs} (评估)", leave=False)
            for batch_idx, batch in enumerate(pbar):
                spectra = batch['spectra'].to(device)
                labels = batch['label'].to(device)
                logits, _ = model(spectra)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"实验 {experiment_id + 1} - 周期 {epoch} 测试准确率: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, configs.save_dir, f'finetuned_model_exp{experiment_id + 1}_seed{current_seed}_pretrain_best')
            print(f"实验 {experiment_id + 1} - 新的最佳准确率: {best_accuracy:.4f} (周期 {best_epoch})。模型已保存。")

    print(f"实验 {experiment_id + 1} 的微调完成。")
    print(f"实验 {experiment_id + 1} 达到的最佳测试准确率: {best_accuracy:.4f} (周期 {best_epoch})")
    return best_accuracy

if __name__ == '__main__':
    num_experiments = 10
    start_seed = 42
    pretrain_accuracies = []

    print("\n" + "=" * 50)
    print("====== 使用预训练权重进行微调 ======")
    print("=" * 50 + "\n")
    start_time = time.time()
    for i in range(num_experiments):
        current_seed = start_seed + i
        current_best_accuracy = finetune(experiment_id=i, current_seed=current_seed)
        pretrain_accuracies.append(current_best_accuracy)
        print(f"--- 实验 {i + 1}/{num_experiments} 完成 (预训练)。最佳准确率: {current_best_accuracy:.4f} ---")
    end_time = time.time()

    print("\n" + "=" * 50)
    print("====== 所有微调实验完成，结果汇总 ======")
    print("=" * 50 + "\n")
    print(f"预训练模式下的最佳准确率列表: {pretrain_accuracies}")
    print(f"预训练模式下平均最佳准确率: {np.mean(pretrain_accuracies):.4f}")
    print(f"预训练模式下最佳准确率的标准差: {np.std(pretrain_accuracies):.4f}")
    print(f"预训练模式总耗时: {(end_time - start_time) / 60:.2f} 分钟")