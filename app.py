import gradio as gr
from router import AgentRouter

# ================= 1. 初始化级联路由系统 =================
print("正在启动系统，加载模型...")
router = AgentRouter()
print("系统启动完毕！")

# ================= 2. 主交互逻辑 =================
def chat_interface(text):
    if not text.strip():
        return "请输入内容", "", "", ""

    # 调用路由引擎
    res = router.route(text)
    
    intent = res["intent"]
    action = res["action"]
    conf = f"{res['confidence']:.4f}"
    note = res["note"]
    
    # 如果遇到 fallback，给一个友好的界面提示
    if action == "fallback":
        intent = "未知 / 闲聊 (OOD)"

    return intent, conf, action, note

# ================= 3. 构建 Gradio UI =================
with gr.Blocks(title="Agent 意图路由系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 Agent 智能意图识别与级联路由系统")
    gr.Markdown("**架构**：基于相对置信度(Margin)的 **NB(粗筛) ➡️ TextCNN(复核) ➡️ 拦截兜底** 级联架构。")
    
    with gr.Row():
        # 左侧输入区
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="用户输入", 
                lines=4, 
                placeholder="请输入英文指令，例如：\n- What is the weather like in New York?\n- Can you remind me to buy milk\n- asdfghjkl"
            )
            btn = gr.Button("执行路由决策", variant="primary")
        
        # 右侧输出区 (纯路由信息)
        with gr.Column(scale=2):
            out_intent = gr.Textbox(label="1. 预测意图 (Intent)")
            out_conf = gr.Textbox(label="2. 置信度 (Confidence)")
            out_action = gr.Textbox(label="3. 路由动作 (Action)")
            out_note = gr.Textbox(label="4. 决策依据 (Decision Note)", lines=3)
            
    # 绑定事件
    btn.click(chat_interface, inputs=inp, outputs=[out_intent, out_conf, out_action, out_note])
    inp.submit(chat_interface, inputs=inp, outputs=[out_intent, out_conf, out_action, out_note])

# ================= 4. 启动服务 =================
if __name__ == "__main__":
    demo.launch()