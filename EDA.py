import pandas as pd
df = pd.read_csv("real_train.csv")
# 统计每个句子按空格切分后的单词数量
word_counts = df['text'].str.split().str.len()
print("90%的句子单词数在:", word_counts.quantile(0.9))
print("最长单词数:", word_counts.max())

#拿到所有label
all_labels = df['label'].unique()
all_labels.sort()
print("所有意图标签:", all_labels)