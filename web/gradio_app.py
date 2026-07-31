"""客服知识库 Gradio 界面：导入文档与全量构建知识库分离。"""

import sys
from pathlib import Path
from uuid import uuid4

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import setup_agent
from src.document_loader import list_upload_files, load_raw_documents, save_uploads

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
.chat-wrap {{
  background: transparent !important;
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
    extra: str = "",
) -> str:
    badge_bg = SUCCESS_BG if ok else "#FFF1E8"
    badge_color = SUCCESS_TEXT if ok else "#9A5B2E"
    cp_color = SUCCESS_TEXT if checkpointer == "运行中" else "#9A5B2E"
    icon = "✅" if ok else "ℹ️"
    short_tid = (thread_id or "")[:8] or "-"
    extra_html = (
        f'<div style="font-size:0.88rem;color:#666;margin-top:8px;">{extra}</div>'
        if extra
        else ""
    )
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
  {extra_html}
</div>
"""


def _welcome_history():
    return [{"role": "assistant", "content": WELCOME_MSG}]


def _uploads_summary() -> str:
    files = list_upload_files()
    if not files:
        return "uploads：暂无导入文件"
    names = "、".join(p.name for p in files[:8])
    more = f" 等 {len(files)} 个" if len(files) > 8 else f"（共 {len(files)} 个）"
    return f"uploads：{names}{more}"


def init_agent(thread_id: str, agent):
    """仅初始化 LLM + 空知识库 Agent（不解析、不入库）。"""
    if not thread_id:
        thread_id = str(uuid4())
    if agent is not None:
        status = _status_html(
            "Agent 已就绪。请导入文档后点击「构建离线知识库」。",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status
    try:
        _, agent = setup_agent()
        status = _status_html(
            "Agent 已初始化。请先导入文档，再构建知识库。",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status
    except Exception as exc:
        status = _status_html(
            f"初始化失败：{exc}",
            thread_id,
            ok=False,
            checkpointer="未连接",
        )
        return None, thread_id, status


def import_documents(files, agent, thread_id: str, kb_ready: bool):
    """导入文档：仅落盘到 uploads，不解析入库。"""
    if not thread_id:
        thread_id = str(uuid4())
    if agent is None:
        status = _status_html(
            "请先点击「初始化 Agent」，再导入文档。",
            thread_id,
            ok=False,
            checkpointer="未连接",
        )
        return agent, thread_id, status, kb_ready

    paths = []
    if files:
        for f in files:
            if f is None:
                continue
            paths.append(f if isinstance(f, (str, Path)) else getattr(f, "name", f))

    if not paths:
        status = _status_html(
            "未选择文件。",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status, kb_ready

    try:
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        saved = save_uploads(paths)
        # 新导入后需要重新全量构建
        status = _status_html(
            f"已导入 {len(saved)} 个文件（仅落盘）。有未入库文件，请重新构建知识库。",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status, False
    except Exception as exc:
        status = _status_html(
            f"导入失败：{exc}",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status, kb_ready


def build_knowledge_base(agent, thread_id: str):
    """全量重建知识库：reset → 解析全部原始目录 → 入库。"""
    if not thread_id:
        thread_id = str(uuid4())
    if agent is None:
        try:
            _, agent = setup_agent()
        except Exception as exc:
            status = _status_html(
                f"初始化失败：{exc}",
                thread_id,
                ok=False,
                checkpointer="未连接",
            )
            return agent, thread_id, status, False

    try:
        raw_docs = load_raw_documents()
        if not raw_docs:
            status = _status_html(
                "未找到可构建的文档。请先导入文件，或将文件放入 data/raw/policies|products|uploads。",
                thread_id,
                ok=False,
                checkpointer="运行中",
                extra=_uploads_summary(),
            )
            return agent, thread_id, status, False

        agent.kb.reset_collection()
        agent.kb.add_documents(raw_docs)
        stats = agent.kb.get_collection_stats()
        status = _status_html(
            f"全量重建完成，可以开始问答。（解析单元 {len(raw_docs)}，库内约 {stats.get('total_documents', '?')} 条）",
            thread_id,
            ok=True,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status, True
    except Exception as exc:
        status = _status_html(
            f"知识库构建失败：{exc}",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
        return agent, thread_id, status, False


def chat_with_kb(message, history, agent, thread_id, kb_ready):
    """聊天回调：需已构建知识库。"""
    if history is None:
        history = _welcome_history()
    if not thread_id:
        thread_id = str(uuid4())

    if not message or not str(message).strip():
        return history, "", agent, thread_id, kb_ready

    text = str(message).strip()

    if agent is None:
        history = history + [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "请先初始化 Agent，导入文档并构建知识库后再提问。",
            },
        ]
        return history, "", agent, thread_id, kb_ready

    if not kb_ready:
        history = history + [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "知识库尚未构建或有新导入未入库。请点击「构建离线知识库」（全量重建）后再提问。",
            },
        ]
        return history, "", agent, thread_id, kb_ready

    try:
        answer = agent.invoke(text, thread_id=thread_id)
    except Exception as exc:
        answer = f"系统处理问题时发生错误：{exc}"

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return history, "", agent, thread_id, kb_ready


def start_new_chat(agent, kb_ready):
    """开启新对话：换新 thread_id，保留欢迎语。"""
    thread_id = str(uuid4())
    if agent is None:
        status = _status_html(
            "请先初始化 Agent，导入文档并构建知识库。",
            thread_id,
            ok=False,
            checkpointer="未连接",
            extra=_uploads_summary(),
        )
    elif not kb_ready:
        status = _status_html(
            "Agent 已就绪，但知识库未构建或有新文件待入库。",
            thread_id,
            ok=False,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
    else:
        status = _status_html(
            "离线知识库已就绪，已开启新对话。",
            thread_id,
            ok=True,
            checkpointer="运行中",
            extra=_uploads_summary(),
        )
    return _welcome_history(), thread_id, status


INITIAL_THREAD = str(uuid4())

with gr.Blocks(title="AI 客服知识库问答") as demo:
    agent_state = gr.State(value=None)
    thread_state = gr.State(value=INITIAL_THREAD)
    kb_ready_state = gr.State(value=False)

    gr.Markdown("# AI 客服知识库问答", elem_classes=["main-title"])
    gr.Markdown(
        "流程：① 初始化 Agent → ② 导入文档（仅落盘）→ ③ 构建知识库（全量重建）→ ④ 问答。"
        "再次导入后需重新构建。多轮对话由 Postgres Checkpointer 按 thread_id 续聊。",
        elem_classes=["main-sub"],
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                init_btn = gr.Button(
                    "⚡  初始化 Agent",
                    elem_classes=["build-btn"],
                    scale=2,
                )
                build_btn = gr.Button(
                    "🗄  构建离线知识库",
                    elem_classes=["build-btn"],
                    scale=3,
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
                    "请先初始化 Agent，再导入文档并全量构建知识库。",
                    INITIAL_THREAD,
                    ok=False,
                    checkpointer="未连接",
                    extra=_uploads_summary(),
                ),
                elem_classes=["status-card"],
            )

    with gr.Row():
        file_input = gr.File(
            label="导入文档（pdf/docx/pptx/xlsx/csv/json/txt/md/图片等）",
            file_count="multiple",
            type="filepath",
        )
        import_btn = gr.Button("📥 导入文档", elem_classes=["build-btn"])

    chatbot = gr.Chatbot(
        label="问答记录",
        value=_welcome_history(),
        height=420,
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

    init_btn.click(
        fn=init_agent,
        inputs=[thread_state, agent_state],
        outputs=[agent_state, thread_state, build_status],
    )

    import_btn.click(
        fn=import_documents,
        inputs=[file_input, agent_state, thread_state, kb_ready_state],
        outputs=[agent_state, thread_state, build_status, kb_ready_state],
    )

    build_btn.click(
        fn=build_knowledge_base,
        inputs=[agent_state, thread_state],
        outputs=[agent_state, thread_state, build_status, kb_ready_state],
    )

    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[agent_state, kb_ready_state],
        outputs=[chatbot, thread_state, build_status],
    )

    send_btn.click(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state, thread_state, kb_ready_state],
        outputs=[chatbot, user_input, agent_state, thread_state, kb_ready_state],
    )
    user_input.submit(
        fn=chat_with_kb,
        inputs=[user_input, chatbot, agent_state, thread_state, kb_ready_state],
        outputs=[chatbot, user_input, agent_state, thread_state, kb_ready_state],
    )

if __name__ == "__main__":
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
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
