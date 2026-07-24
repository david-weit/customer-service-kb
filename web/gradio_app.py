import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_agent


def build_knowledge_base():
    """构建离线知识库并返回 agent。"""
    try:
        _, agent = setup_agent()
        return agent, "✅ 离线知识库构建完成，可以开始问答。"
    except Exception as exc:
        return None, f"❌ 知识库构建失败：{exc}"


def chat_with_kb(message, history, agent):
    """聊天回调：优先检查知识库状态，再执行问答。"""
    if history is None:
        history = []

    if not message.strip():
        return history, "", agent

    if agent is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "请先点击「构建离线知识库」按钮。"},
        ]
        return history, "", agent

    try:
        answer = agent.invoke(message)
    except Exception as exc:
        answer = f"系统处理问题时发生错误：{exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, "", agent


with gr.Blocks(title="客服知识库问答") as demo:
    gr.Markdown("## AI 客服知识库问答")
    gr.Markdown("先点击“构建离线知识库”，完成后再开始聊天提问。")

    agent_state = gr.State(value=None)

    with gr.Row():
        build_btn = gr.Button("构建离线知识库", variant="primary")
        build_status = gr.Textbox(
            label="知识库状态",
            value="未构建，请先点击按钮构建离线知识库。",
            interactive=False,
        )

    chatbot = gr.Chatbot(label="问答记录", height=420)
    user_input = gr.Textbox(label="输入问题", placeholder="例如：退换货政策是什么？")
    send_btn = gr.Button("发送", variant="secondary")

    build_btn.click(
        fn=build_knowledge_base,
        inputs=[],
        outputs=[agent_state, build_status],
    )

    send_btn.click(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state],
        outputs=[chatbot, user_input, agent_state],
    )
    user_input.submit(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state],
        outputs=[chatbot, user_input, agent_state],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)