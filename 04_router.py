import json, torch, torch.nn as nn, torch.nn.functional as F
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

#1. 重新定义 TextCNN:PyTorch 的 torch.save(model.state_dict()) 只保存了“参数（数字）”，没有保存“结构（代码）
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=(1,2,3), num_filters=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.max(torch.relu(conv(emb)), dim=2).values for conv in self.convs]
        h = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(h)

#2. 加载所有模型
class AgentRouter:
    def __init__(self, ckpt_dir="checkpoints"):
        print("正在加载 NB 模型...")
        self.vectorizer = joblib.load(f"{ckpt_dir}/tfidf_vectorizer.pkl")
        self.nb_model = joblib.load(f"{ckpt_dir}/nb_model.pkl")

        print("正在加载 TextCNN 模型...")
        with open(f"{ckpt_dir}/config.json") as f:
            cfg = json.load(f)
        with open(f"{ckpt_dir}/vocab.json", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(f"{ckpt_dir}/label_map.json", encoding="utf-8") as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

        self.max_len = cfg["max_len"]
        self.model = TextCNN(
            len(self.vocab), cfg["embed_dim"], cfg["num_classes"],
            tuple(cfg["kernel_sizes"]), cfg["num_filters"]
        )
        self.model.load_state_dict(torch.load(f"{ckpt_dir}/textcnn.pth", map_location="cpu"))
        self.model.eval()
        print("所有模型加载完毕！\n")

    def _encode(self, text):
        seq = [self.vocab.get(word, 1) for word in text.lower().split()][:self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    @torch.no_grad()
    def predict_textcnn(self, text):
        x = torch.tensor([self._encode(text)])
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        return self.label_map[idx], probs[idx].item()

    def predict_nb(self, text):
        x_vec = self.vectorizer.transform([text])
        probs = self.nb_model.predict_proba(x_vec)[0]
        idx = probs.argmax()
        # 获取原始标签名
        label = self.nb_model.classes_[idx]
        return label, probs[idx]

    #3. 级联路由策略
    def route(self, text, nb_threshold=0.04, margin_threshold=2.0, cnn_threshold=0.60):
        result = {"text": text, "intent": None, "confidence": 0.0, "action": None, "note": ""}

        # 第一级：朴素贝叶斯粗筛
        
        # 为了算 Margin，我们需要拿到所有的概率分布
        x_vec = self.vectorizer.transform([text])
        probs_nb = self.nb_model.predict_proba(x_vec)[0]
        
        # 找出排名前 2 的概率
        sorted_indices = probs_nb.argsort()[::-1]
        top1_conf = probs_nb[sorted_indices[0]]
        top2_conf = probs_nb[sorted_indices[1]]
        nb_intent = self.nb_model.classes_[sorted_indices[0]]
        
        # 相对置信度：第一名是第二名的几倍？
        margin = top1_conf / (top2_conf + 1e-6) # 加个极小值防除以0

        # 判定条件：不仅绝对概率要过关，还得比第二名高出一大截（比如 2 倍以上）
        if top1_conf >= nb_threshold and margin >= margin_threshold:
            result.update({
                "intent": nb_intent, 
                "confidence": top1_conf, 
                "action": "direct_nb", 
                "note": f"NB高置信分流 (绝对概率:{top1_conf:.3f}, 远超第二名 {margin:.1f}倍)"
            })
            return result

        # 第二级：贝叶斯拿不准，交给 TextCNN 复核
        cnn_intent, cnn_conf = self.predict_textcnn(text)
        
        if cnn_conf >= cnn_threshold:
            result.update({"intent": cnn_intent, "confidence": cnn_conf, "action": "review_cnn", "note": f"NB({top1_conf:.2f})拿不准，CNN复核为 {cnn_intent} ({cnn_conf:.2f})"})
        else:
            # 第三级：两个模型都拿不准，走兜底（闲聊或转人工）
            result.update({"intent": "fallback_chat", "confidence": cnn_conf, "action": "fallback", "note": f"两模型均无把握 (NB:{top1_conf:.2f}, CNN:{cnn_conf:.2f})，转兜底"})
            
        return result

#4. 测试运行
if __name__ == "__main__":
    router = AgentRouter()
    
    test_sentences = [
        "What is the weather like in New York?",  # 简单句，NB 应该直接判断
        "Can you remind me to buy milk",           # 简单句
        "I want to know if it's going to rain tomorrow afternoon" # 复杂句，可能触发 CNN 复核
    ]
    
    for text in test_sentences:
        res = router.route(text)
        print(f"用户输入: {text}")
        print(f"-> 路由决策: {res['action']} | 意图: {res['intent']} | 置信度: {res['confidence']:.2f}")
        print(f"-> 决策说明: {res['note']}\n")