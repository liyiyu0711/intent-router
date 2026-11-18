import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from router import AgentRouter  

# ================= 1. 加载环境变量与初始化 LLM =================
# 从 .env 文件读取 API Key
load_dotenv()

# 初始化智谱 GLM 客户端（兼容 OpenAI 格式）
llm_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ================= 2. 初始化你的级联路由模型 =================
print("正在启动系统，加载本地模型...")
router = AgentRouter()
print("系统启动完毕！")

# ================= 3. LLM 执行器：大模型统筹，小模型决策 =================
def execute_with_llm(intent, user_text):
    """
    将小模型识别出的意图和原始文本，发给大模型进行具体任务执行。
    这里使用了 Prompt Engineering，让 LLM 扮演特定工具的角色。
    """
    # 针对不同意图，给大模型设定不同的“角色扮演”
    system_prompts = {
        "weather": "你是一个精准的天气助手。请根据用户提供的城市，模拟返回当地的天气信息（晴朗/下雨等），字数简短。",
        "translate": "你是一个专业的翻译官。请将用户的话翻译成中文（如果原文是中文则翻译成英文），字数简短。",
        "calculator": "你是一个计算器。请计算出用户表达式的准确结果，直接回复答案，不要废话。",
        "greeting": "你是一个友好的智能助手。请友善地回复用户的问候。"
    }
    # 如果没有专门定义，给一个通用的系统设定
    sys_prompt = system_prompts.get(intent, f"你是一个全能助手。系统识别出用户的意图是【{intent}】。请根据意图，为用户提供合理的回答或模拟执行结果。")

    try:
        response = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ 大模型调用失败: {str(e)}"

# ================= 4. 主交互逻辑 =================
def chat_interface(text):
    if not text.strip():
        return "请输入内容", "", "", "", ""

    # 第一步：小模型预测与路由
    # 这里调用的是我们之前写的级联路由逻辑，它会输出：intent, confidence, action, note
    res = router.route(text)
    
    intent = res["intent"]
    action = res["action"]
    conf = f"{res['confidence']:.4f}"
    note = res["note"]
    
    # 第二步：根据路由结果，决定是否呼叫大模型
    tool_result = ""
    
    if action in ["direct_nb", "review_cnn"]:
        # 如果小模型自信能判断出意图，进入 LLM 工具执行阶段
        tool_result = f"🤖 [LLM 执行结果]:\n{execute_with_llm(intent, text)}"
        
    elif action == "fallback":
        # 如果兜底，就不用 LLM 去查天气算数了，直接让 LLM 闲聊回复
        tool_result = f"💬 [兜底闲聊]:\n{execute_with_llm('chat', text)}"
    else:
        tool_result = "🛠️ [执行器]: 意图不明确，无法调用具体工具。"

    # 返回给 Gradio 界面的五列数据
    return intent, conf, action, tool_result, note

# ================= 5. 构建 Gradio UI =================
with gr.Blocks(title="Agent 意图路由与 LLM 执行系统") as demo:
    gr.Markdown("## 🤖 Agent 智能意图识别与路由系统")
    gr.Markdown("**架构**：小模型 (NB + TextCNN) 级联路由 ➡️ 大模型 (GLM-4.7-Flash) 任务执行")
    
    with gr.Row():
        # 左侧输入区
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="用户输入", 
                lines=4, 
                placeholder="请输入英文指令，例如：\n- What is the weather like in New York?\n- Translate 'apple' to Chinese\n- Can you remind me to buy milk"
            )
            btn = gr.Button("发送请求", variant="primary")
        
        # 右侧输出区
        with gr.Column(scale=2):
            out_intent = gr.Textbox(label="1. 预测意图 (Intent)")
            out_conf = gr.Textbox(label="2. 置信度 (Confidence)")
            out_action = gr.Textbox(label="3. 路由动作 (Action)")
            out_tool = gr.Textbox(label="4. 执行结果 (LLM Execution)", lines=4)
            out_note = gr.Textbox(label="5. 决策依据 (Decision Note)", lines=2)
            
    # 绑定按钮点击和回车事件
    btn.click(chat_interface, inputs=inp, outputs=[out_intent, out_conf, out_action, out_tool, out_note])
    inp.submit(chat_interface, inputs=inp, outputs=[out_intent, out_conf, out_action, out_tool, out_note])

# ================= 6. 启动服务 =================
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())