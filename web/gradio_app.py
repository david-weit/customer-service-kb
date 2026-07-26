import sys
from pathlib import Path
from uuid import uuid4

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_agent


def build_knowledge_base():
    """构建离线知识库并返回 agent 与新 thread_id。"""
    try:
        _, agent = setup_agent()
        thread_id = str(uuid4())
        return (
            agent,
            thread_id,
            "✅ 离线知识库构建完成，可以开始问答。",
        )
    except Exception as exc:
        return None, str(uuid4()), f"❌ 知识库构建失败：{exc}"


def chat_with_kb(message, history, agent, thread_id):
    """聊天回调：优先检查知识库状态，再按 thread_id 续聊。"""
    if history is None:
        history = []
    if not thread_id:
        thread_id = str(uuid4())

    if not message.strip():
        return history, "", agent, thread_id

    if agent is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "请先点击「构建离线知识库」按钮。"},
        ]
        return history, "", agent, thread_id

    try:
        answer = agent.invoke(message, thread_id=thread_id)
    except Exception as exc:
        answer = f"系统处理问题时发生错误：{exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, "", agent, thread_id


def start_new_chat():
    """开启新对话：换新 thread_id，清空界面历史。"""
    return [], str(uuid4())


with gr.Blocks(title="客服知识库问答") as demo:
    gr.Markdown("## AI 客服知识库问答")
    gr.Markdown(
        "先点击“构建离线知识库”，完成后再开始聊天。"
        "多轮对话由 Postgres Checkpointer 按 thread_id 续聊；点「新对话」切换会话。"
    )

    agent_state = gr.State(value=None)
    thread_state = gr.State(value=str(uuid4()))

    with gr.Row():
        build_btn = gr.Button("构建离线知识库", variant="primary")
        new_chat_btn = gr.Button("新对话", variant="secondary")
        build_status = gr.Textbox(
            label="知识库状态",
            value="未构建，请先点击按钮构建离线知识库。",
            interactive=False,
        )

    chatbot = gr.Chatbot(label="问答记录", height=420)
    user_input = gr.Textbox(
        label="输入问题",
        placeholder="例如：查一下我的订单 / 退换货政策是什么？",
    )
    send_btn = gr.Button("发送", variant="secondary")

    build_btn.click(
        fn=build_knowledge_base,
        inputs=[],
        outputs=[agent_state, thread_state, build_status],
    )

    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chatbot, thread_state],
    )

    send_btn.click(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state, thread_state],
        outputs=[chatbot, user_input, agent_state, thread_state],
    )
    user_input.submit(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state, thread_state],
        outputs=[chatbot, user_input, agent_state, thread_state],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
