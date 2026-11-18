from datasets import load_dataset
import pandas as pd

print("正在从 HuggingFace 拉取数据（首次运行可能需要几分钟）...")
dataset = load_dataset("FastFit/clinc_150") 

# 1. 原始数据
print("训练集大小:", len(dataset['train']))
print("数据样例:", dataset['train'][0])


# 3. 转成 pandas DataFrame
def to_df(split_name):
    data = dataset[split_name]
    # 转成 DataFrame，只保留 text 和 intent 两列
    df = pd.DataFrame({
        'text': data['text'],
        'label': data['label']
    })
    return df

df_train = to_df('train')
df_val = to_df('validation')
df_test = to_df('test')

# 4. 合并（直接合并训练和验证，测试集单独留着）
df_train_all = pd.concat([df_train, df_val]).sample(frac=1, random_state=42).reset_index(drop=True)
#sample(frac=1,random_state=42)把数据 100% 抽样，但是打乱顺序
#reset_index(drop=True)重排索引

# 5. 保存为csv 格式：一个训练集，一个测试集
df_train_all.to_csv("real_train.csv", index=False, encoding="utf-8")
df_test.to_csv("real_test.csv", index=False, encoding="utf-8")
print(f"处理完成！共生成 {len(df_train_all)} 条真实数据，已保存")