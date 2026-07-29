"""客服知识库 Gradio 界面（按设计稿布局）。"""

import sys
from pathlib import Path
from uuid import uuid4

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_agent

ORANGE = "#F0783C"
ORANGE_HOVER = "#E06728"
BG = "#F7EDE4"
PANEL_BG = "#FFF8F2"
SUCCESS_BG = "#E8F6EC"
SUCCESS_TEXT = "#1F7A3A"

WELCOME_MSG = (
    "你好！欢迎来到客服中心~ 😊\n\n"
    "请问有什么可以帮您的吗？比如：\n"
    "- 查询订单物流\n"
    "- 咨询退换货政策\n\n"
    "期待为您服务！✨"
)

CSS = f"""
.gradio-container {{
  background: {BG} !important;
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif !important;
  max-width: 980px !important;
  margin: 0 auto !important;
}}
.main-title h1 {{
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: #2b2b2b !important;
  margin: 0.2rem 0 0.4rem 0 !important;
}}
.main-sub {{
  color: #6b6b6b !important;
  font-size: 0.95rem !important;
  line-height: 1.55 !important;
  margin-bottom: 0.8rem !important;
}}
.build-btn button {{
  background: {ORANGE} !important;
  border: none !important;
  color: white !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  min-height: 52px !important;
  box-shadow: 0 6px 16px rgba(240, 120, 60, 0.28) !important;
}}
.build-btn button:hover {{
  background: {ORANGE_HOVER} !important;
}}
.circle-btn button {{
  background: {ORANGE} !important;
  border: none !important;
  color: white !important;
  border-radius: 999px !important;
  width: 64px !important;
  min-width: 64px !important;
  height: 64px !important;
  padding: 0 !important;
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  line-height: 1.15 !important;
  box-shadow: 0 6px 16px rgba(240, 120, 60, 0.28) !important;
}}
.circle-btn button:hover {{
  background: {ORANGE_HOVER} !important;
}}
.status-card {{
  background: {PANEL_BG} !important;
  border: 1px solid #f0dcc8 !important;
  border-radius: 16px !important;
  padding: 14px 16px !important;
  min-height: 120px !important;
}}
.status-card .label-wrap {{
  margin-bottom: 6px !important;
}}
.chat-wrap {{
  background: transparent !important;
}}
.chat-wrap .bubble-wrap {{
  border-radius: 16px !important;
}}
.input-row textarea {{
  border-radius: 18px !important;
  border: 1px solid #e8d5c4 !important;
  background: #fff !important;
  min-height: 56px !important;
}}
.hint-text {{
  color: #9a8a7a !important;
  font-size: 0.85rem !important;
  margin-top: 0.25rem !important;
}}
"""


def _status_html(
    message: str,
    thread_id: str,
    *,
    ok: bool = False,
    checkpointer: str = "未连接",
) -> str:
    badge_bg = SUCCESS_BG if ok else "#FFF1E8"
    badge_color = SUCCESS_TEXT if ok else "#9A5B2E"
    cp_color = SUCCESS_TEXT if checkpointer == "运行中" else "#9A5B2E"
    icon = "✅" if ok else "ℹ️"
    short_tid = (thread_id or "")[:8] or "-"
    return f"""
<div style="line-height:1.55;color:#333;">
  <div style="font-weight:700;margin-bottom:8px;color:#444;">知识库状态</div>
  <div style="background:{badge_bg};color:{badge_color};border-radius:10px;
              padding:8px 10px;margin-bottom:10px;font-size:0.92rem;">
    {icon} {message}
  </div>
  <div style="font-size:0.9rem;color:#555;">
    当前线程：<code style="background:#f3e7dc;padding:1px 6px;border-radius:6px;">
    thread_id: {short_tid}</code>
  </div>
  <div style="font-size:0.9rem;color:#555;margin-top:4px;">
    Postgres Checkpointer：
    <span style="color:{cp_color};font-weight:700;">{checkpointer}</span>
  </div>
</div>
"""


def _welcome_history():
    return [{"role": "assistant", "content": WELCOME_MSG}]


def build_knowledge_base(thread_id: str):
    """构建离线知识库并返回 agent 与状态面板。"""
    if not thread_id:
        thread_id = str(uuid4())
    try:
        _, agent = setup_agent()
        status = _status_html(
            "离线知识库构建完成，可以开始问答。",
            thread_id,
            ok=True,
            checkpointer="运行中",
        )
        return agent, thread_id, status
    except Exception as exc:
        status = _status_html(
            f"知识库构建失败：{exc}",
            thread_id,
            ok=False,
            checkpointer="未连接",
        )
        return None, thread_id, status


def chat_with_kb(message, history, agent, thread_id):
    """聊天回调：优先检查知识库状态，再按 thread_id 续聊。"""
    if history is None:
        history = _welcome_history()
    if not thread_id:
        thread_id = str(uuid4())

    if not message or not str(message).strip():
        return history, "", agent, thread_id

    text = str(message).strip()

    if agent is None:
        history = history + [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "请先点击「构建离线知识库」按钮，完成后再开始提问。",
            },
        ]
        return history, "", agent, thread_id

    try:
        answer = agent.invoke(text, thread_id=thread_id)
    except Exception as exc:
        answer = f"系统处理问题时发生错误：{exc}"

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return history, "", agent, thread_id


def start_new_chat(agent):
    """开启新对话：换新 thread_id，保留欢迎语。"""
    thread_id = str(uuid4())
    if agent is None:
        status = _status_html(
            "未构建，请先点击按钮构建离线知识库。",
            thread_id,
            ok=False,
            checkpointer="未连接",
        )
    else:
        status = _status_html(
            "离线知识库已就绪，已开启新对话。",
            thread_id,
            ok=True,
            checkpointer="运行中",
        )
    return _welcome_history(), thread_id, status


INITIAL_THREAD = str(uuid4())

with gr.Blocks(title="AI 客服知识库问答") as demo:
    agent_state = gr.State(value=None)
    thread_state = gr.State(value=INITIAL_THREAD)

    gr.Markdown("# AI 客服知识库问答", elem_classes=["main-title"])
    gr.Markdown(
        "先点击「构建离线知识库」，完成后再开始聊天。"
        "多轮对话由 Postgres Checkpointer 按 thread_id 续聊；点「新对话」切换会话。",
        elem_classes=["main-sub"],
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                build_btn = gr.Button(
                    "🗄  构建离线知识库",
                    elem_classes=["build-btn"],
                    scale=4,
                )
                new_chat_btn = gr.Button(
                    "＋\n新对话",
                    elem_classes=["circle-btn"],
                    scale=1,
                    min_width=72,
                )
        with gr.Column(scale=4):
            build_status = gr.HTML(
                value=_status_html(
                    "未构建，请先点击按钮构建离线知识库。",
                    INITIAL_THREAD,
                    ok=False,
                    checkpointer="未连接",
                ),
                elem_classes=["status-card"],
            )

    chatbot = gr.Chatbot(
        label="问答记录",
        value=_welcome_history(),
        height=440,
        avatar_images=(
            None,
            "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png",
        ),
        elem_classes=["chat-wrap"],
        buttons=["copy"],
        layout="bubble",
    )

    with gr.Row(equal_height=True):
        user_input = gr.Textbox(
            label="输入问题...",
            placeholder="例如：查一下我的订单 / 退换货政策是什么？",
            lines=2,
            scale=8,
            container=True,
        )
        send_btn = gr.Button(
            "✈\n发送",
            elem_classes=["circle-btn"],
            scale=1,
            min_width=72,
        )

    gr.Markdown(
        "例如：查一下我的订单 / 退换货政策是什么？",
        elem_classes=["hint-text"],
    )

    build_btn.click(
        fn=build_knowledge_base,
        inputs=[thread_state],
        outputs=[agent_state, thread_state, build_status],
    )

    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[agent_state],
        outputs=[chatbot, thread_state, build_status],
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

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.Color(
                c50="#FFF7F0",
                c100="#FFE8D6",
                c200="#FFD0AD",
                c300="#FFB784",
                c400="#F78E52",
                c500=ORANGE,
                c600=ORANGE_HOVER,
                c700="#C9551A",
                c800="#A34414",
                c900="#82360F",
                c950="#5C250A",
            ),
            secondary_hue="stone",
            neutral_hue="stone",
        ),
    )
