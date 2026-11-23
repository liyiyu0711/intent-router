import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from router import AgentRouter

# ================= 1. 加载模型和数据 =================
print("加载模型中...")
router = AgentRouter() 
test_df = pd.read_csv("real_test.csv")
texts = test_df["text"].tolist()
true_labels = np.array(test_df["label"].tolist())

# ================= 2. 收集所有预测结果与置信度 =================
print("正在推理全部测试集...")
nb_preds, nb_top1_probs, nb_margins = [], [], []
cnn_preds = []

for text in texts:
    # 获取 NB 预测分布（严格按照路由逻辑）
    x_vec = router.vectorizer.transform([text])
    probs = router.nb_model.predict_proba(x_vec)[0]
    
    # 排序拿 Top1 和 Top2
    sorted_indices = probs.argsort()[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]
    
    top1_conf = probs[top1_idx]
    top2_conf = probs[top2_idx]
    margin = top1_conf / (top2_conf + 1e-6)
    
    nb_preds.append(router.nb_model.classes_[top1_idx])
    nb_top1_probs.append(top1_conf)
    nb_margins.append(margin)
    
    # 获取 CNN 预测
    cnn_intent, _ = router.predict_textcnn(text)
    cnn_preds.append(cnn_intent)

nb_preds = np.array(nb_preds)
nb_top1_probs = np.array(nb_top1_probs)
nb_margins = np.array(nb_margins)
cnn_preds = np.array(cnn_preds)

# ================= 3. 遍历绝对阈值，计算指标 =================
# 设定固定的 Margin 门槛（和 router.py 里保持一致，比如 2.0）
FIXED_MARGIN_THRESHOLD = 2.0
thresholds = np.linspace(0.01, 0.15, 50) 

coverage_rates = [] 
nb_accuracies = []  
cnn_accuracies = [] 
total_accuracies = [] 

for th in thresholds:
    # 核心：必须同时满足绝对概率达标 和 Margin 达标
    passed_mask = (nb_top1_probs >= th) & (nb_margins >= FIXED_MARGIN_THRESHOLD)
    
    # 1. 覆盖率
    coverage = passed_mask.sum() / len(texts)
    coverage_rates.append(coverage)
    
    # 2. NB 放行部分的准确率
    if passed_mask.sum() > 0:
        nb_acc = accuracy_score(true_labels[passed_mask], nb_preds[passed_mask])
    else:
        nb_acc = np.nan
    nb_accuracies.append(nb_acc)
    
    # 3. CNN 复核部分的准确率 (剩下没放行的)
    review_mask = ~passed_mask
    if review_mask.sum() > 0:
        cnn_acc = accuracy_score(true_labels[review_mask], cnn_preds[review_mask])
    else:
        cnn_acc = np.nan
    cnn_accuracies.append(cnn_acc)
    
    # 4. 系统整体准确率 (放行的信 NB，没放行的信 CNN)
    final_preds = np.where(passed_mask, nb_preds, cnn_preds)
    total_acc = accuracy_score(true_labels, final_preds)
    total_accuracies.append(total_acc)

# ================= 4. 绘制曲线图 =================
plt.figure(figsize=(10, 6))
plt.plot(thresholds, coverage_rates, label="Coverage Rate (Passed %)", color="blue", linewidth=2)
plt.plot(thresholds, nb_accuracies, label="NB Acc (Passed Set)", color="green", linestyle="--")
plt.plot(thresholds, cnn_accuracies, label="CNN Acc (Review Set)", color="orange", linestyle="--")
plt.plot(thresholds, total_accuracies, label="System Total Accuracy", color="red", linewidth=2)

plt.title(f"Router Analysis (Fixed Margin >= {FIXED_MARGIN_THRESHOLD})", fontsize=14)
plt.xlabel("NB Absolute Threshold", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.legend(loc="best")
plt.grid(True, linestyle=":", alpha=0.7)

# 标注一下你最终选的 0.04 阈值
current_th = 0.04
# 找到最接近 0.04 的 index
idx = np.argmin(np.abs(thresholds - current_th))
plt.axvline(x=thresholds[idx], color='gray', linestyle='-.', label=f'Current Threshold ({current_th})')
plt.scatter([thresholds[idx]], [coverage_rates[idx]], color='black', zorder=5)
plt.text(thresholds[idx] + 0.002, coverage_rates[idx], f"  Cov: {coverage_rates[idx]:.1%}", verticalalignment='bottom')

plt.tight_layout()
plt.savefig("router_evaluation.png", dpi=300)
plt.show()