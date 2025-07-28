import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os
import numpy as np
from collections import Counter

from configs import get_args
from dataset import load_and_preprocess_data, SeedSpectralDataset, DataLoader
from models.encoder import base_Model

# Qwen-0.6B 相关函数
def load_qwen_model():
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_path = "./checkpoint-400_svi_0.6/checkpoint-400_svi_0.6"  # 更新为 SVI 路径
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, quantization_config=quantization_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    return model, tokenizer

def build_prompt(instruction):
    return f"Input: {instruction} Output:"

def generate_qwen_response(model, tokenizer, instruction, max_new_tokens=2):
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.01,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    first_word = response.split()[0].capitalize() if response else ""
    return first_word if first_word in ["Vigor", "Non"] else "Invalid output"

# 小模型相关函数
def load_small_model(configs, device, checkpoint_path):
    model = base_Model(configs).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"成功加载小模型权重：{checkpoint_path}")
    else:
        raise FileNotFoundError(f"未找到小模型权重：{checkpoint_path}")
    return model

def evaluate_hybrid_model():
    configs = get_args()
    device = torch.device(configs.device)
    test_file_path = "./data_cat/data_cat/Test_svi_output.json"  # 更新为 SVI 数据集

    # 加载小模型
    checkpoint_path = os.path.join(configs.save_dir, 'finetuned_model_exp5_seed46_pretrain_best_epoch_11.pth')
    small_model = load_small_model(configs, device, checkpoint_path)
    small_model.eval()

    # 加载 Qwen-0.6B 模型
    qwen_model, qwen_tokenizer = load_qwen_model()

    # 加载测试数据集
    _, test_data_df, global_mean, global_std = load_and_preprocess_data()
    test_dataset = SeedSpectralDataset(test_data_df, augmentation_mode='finetune', mean=global_mean, std=global_std)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

    # 加载 Qwen 测试数据集
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"测试文件 {test_file_path} 不存在")
    qwen_test_dataset = load_dataset('json', data_files=test_file_path)["train"]

    # 检查样本数量是否匹配
    if len(test_dataset) != len(qwen_test_dataset):
        raise ValueError(f"测试数据集样本数量不匹配：小模型数据集 {len(test_dataset)}，Qwen 数据集 {len(qwen_test_dataset)}")

    # 预测逻辑
    predictions, true_labels, model_used = [], [], []
    diff_threshold = 0.8  # 最大概率与次大概率的差值阈值，可调

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中")):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)
            batch_size = spectra.size(0)

            # 小模型预测
            logits, _ = small_model(spectra)
            softmax_probs = F.softmax(logits, dim=1)
            top2_probs, top2_indices = torch.topk(softmax_probs, k=2, dim=1)  # 获取前两个最大概率
            max_probs = top2_probs[:, 0].cpu().numpy()  # 最大概率
            second_probs = top2_probs[:, 1].cpu().numpy()  # 次大概率
            prob_diffs = max_probs - second_probs  # 计算差值
            preds = torch.argmax(softmax_probs, dim=1).cpu().numpy()

            # 遍历批次中的每个样本
            for i in range(batch_size):
                true_label = labels[i].item()
                true_label_str = "Vigor" if true_label == 1 else "Non"  # 假设 1=Vigor, 0=Non
                sample_idx = batch_idx * configs.batch_size + i

                if prob_diffs[i] > diff_threshold:
                    # 差值大，小模型自信，使用其预测
                    pred = "Vigor" if preds[i] == 1 else "Non"
                    model_used.append("Small")
                else:
                    # 差值小，小模型不自信，使用 Qwen-0.6B
                    if sample_idx < len(qwen_test_dataset):
                        instruction = qwen_test_dataset[sample_idx]["instruction"]
                        pred = generate_qwen_response(qwen_model, qwen_tokenizer, instruction)
                        model_used.append("Qwen")
                    else:
                        print(f"警告：样本索引 {sample_idx} 超出 Qwen 数据集范围，使用小模型预测")
                        pred = "Vigor" if preds[i] == 1 else "Non"
                        model_used.append("Small")

                predictions.append(pred)
                true_labels.append(true_label_str)

    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)
    print("\n=== 混合模型评估结果 ===")
    print(f"整体准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=["Non", "Vigor"], zero_division=0))

    # 统计模型使用情况
    model_usage = Counter(model_used)
    print("\n=== 模型使用统计 ===")
    print(f"小模型预测次数: {model_usage['Small']}")
    print(f"Qwen-0.6B 预测次数: {model_usage['Qwen']}")

    # 输出错误分类样本
    print("\n=== 错误分类样本 ===")
    misclassified = False
    for i, (pred, true, used) in enumerate(zip(predictions, true_labels, model_used)):
        if pred != true:
            misclassified = True
            instruction = qwen_test_dataset[i]["instruction"][:50] + "..." if i < len(qwen_test_dataset) else "未知指令"
            print(f"样本 {i+1}:")
            print(f"输入: {instruction}")
            print(f"预测: {pred}, 真实: {true}, 使用模型: {used}")
    if not misclassified:
        print("无错误分类样本。")

if __name__ == "__main__":
    evaluate_hybrid_model()