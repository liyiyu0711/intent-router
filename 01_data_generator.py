import pandas as pd, random
random.seed(42)

templates = {
    "weather": [
        "今天{city}天气怎么样", "{city}明天会下雨吗", "查一下{city}的温度",
        "外面冷不冷", "需要带伞吗", "{city}这周末适合出门吗"
    ],
    "math": [
        "{a}加{b}等于多少", "计算{a}乘以{b}", "{a}减{b}是多少",
        "帮我算一下{a}除以{b}", "这题答案是多少：{a}+{b}"
    ],
    "database": [
        "查询{table}里所有的记录", "{table}中最近的数据", "查看{table}内容",
        "把{table}的数据拉出来", "帮我看看{table}里有什么"
    ],
    "chat": [
        "讲个笑话", "你是谁", "今天心情不好",
        "随便聊点什么", "你最近怎么样"
    ],
    "reminder": [
        "{time}提醒我{thing}", "帮我定个{time}的闹钟",
        "别忘了{time}要{thing}", "到{time}叫我一下"
    ]
}

cities = ["北京","上海","广州","深圳","成都"]
tables = ["用户表","订单表","商品表"]
times = ["明天早上8点","今晚9点","下午3点"]
things = ["开会","吃药","交作业"]

data = []
for label, tpls in templates.items():
    for _ in range(100):
        t = random.choice(tpls)
        text = t.format(
            city=random.choice(cities), a=random.randint(1,50),
            b=random.randint(1,9), table=random.choice(tables),
            time=random.choice(times), thing=random.choice(things)
        )
        data.append({"text": text, "label": label})

df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
df.to_csv("data.csv", index=False, encoding="utf-8")
print("数据生成完毕，共", len(df), "条\n")

print("=== 数据集预览 ===")
print(df.head())

print("\n=== 各类别数据分布 ===")
print(df['label'].value_counts())