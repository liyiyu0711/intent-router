import json, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")
le = LabelEncoder()
y = le.fit_transform(df["label"])

# 字级词表（中文不需要分词）:统计所有出现过的汉字，给每个字一个整数ID。<PAD> 占0号，专门用来补齐。
vocab = {"<PAD>": 0}
for text in df["text"]:
    for ch in text:
        if ch not in vocab:
            vocab[ch] = len(vocab)
#一句话变成长度固定为 20 的整数列表：编码+截断+补齐（用0）
MAX_LEN = 20
def encode(text):
    seq = [vocab.get(ch, 0) for ch in text][:MAX_LEN]
    return seq + [0] * (MAX_LEN - len(seq))

#划分训练集与测试集
X = [encode(t) for t in df["text"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#把 Python 列表转成 PyTorch 张量，再交给 DataLoader 按 batch 取。shuffle=True 每个 epoch 打乱顺序，防止模型学“样本顺序”这种假规律
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=16)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=(2,3,4), num_filters=32):
        super().__init__()
        # Embedding：字 ID → 向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #创建 3 个一维卷积层，卷积核宽度分别是 2、3、4
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # (B, L, E)
        #把“序列长度”和“特征维度”换个位置，nn.Conv1d 要求输入形状是 (batch, channels, length)
        emb = emb.permute(0, 2, 1)           # (B, E, L) -> Conv1d
        pooled = []
        #卷积 + ReLU + 最大池化
        for conv in self.convs:
            c = torch.relu(conv(emb))        # (B, F, L-k+1)
            c = torch.max(c, dim=2).values   # (B, F)  max pooling over time
            pooled.append(c)
        #Dropout（防止过拟合）+全连接
        h = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(h)

model = TextCNN(len(vocab), embed_dim=32, num_classes=len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#训练循环
for epoch in range(20):
    #切到训练模式
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()#清空上一步的梯度。PyTorch 默认累加梯度，不清零会越积越大
        loss = criterion(model(bx), by)#前向传播算 logits，再和真实标签算交叉熵
        loss.backward()#BP，自动算每个参数的梯度
        optimizer.step()#按 Adam 的规则更新参数

#评估阶段
model.eval()#关闭 Dropout
correct = total = 0
with torch.no_grad():
    for bx, by in test_loader:
        pred = model(bx).argmax(dim=1)#5 个 logits 里取最大的那个下标作为预测类别
        correct += (pred == by).sum().item()
        total += by.size(0)

print(f"TextCNN 测试准确率: {correct/total:.4f}")



import os, json
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/textcnn.pth")#存模型参数
with open("checkpoints/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)#存字到ID的映射，不然router无法把新句子转成相同索引
with open("checkpoints/label_map.json", "w", encoding="utf-8") as f:
    json.dump({i: name for i, name in enumerate(le.classes_)}, f, ensure_ascii=False)#把输出的下标映射回类别名

# 同时保存模型结构参数，加载时用来重建模型
with open("checkpoints/config.json", "w") as f:
    json.dump({
        "embed_dim": 32, "num_classes": len(le.classes_),
        "kernel_sizes": [2,3,4], "num_filters": 32, "max_len": MAX_LEN
    }, f)