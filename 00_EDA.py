import pandas as pd
df = pd.read_csv("real_train.csv")
# 统计每个句子按空格切分后的单词数量
word_counts = df['text'].str.split().str.len()
print("90%的句子单词数在:", word_counts.quantile(0.9))
print("最长单词数:", word_counts.max())
