# pages/chat.py - fully fixed

import os
import sys
import io
import json
import re
import warnings
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Silence everything ───────────────────────────────────
os.environ["SILENT_MODE"]                     = "true"
os.environ["TOKENIZERS_PARALLELISM"]          = "false"
os.environ["TRANSFORMERS_VERBOSITY"]          = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONWARNINGS"]                  = "ignore"
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Chat Widget Demo",
    page_icon="💬",
    layout="wide"
)

# ─── Load Config ──────────────────────────────────────────
def load_config() -> dict:
    with open("admin_config.json", "r") as f:
        return json.load(f)

config    = load_config()
bot_color = config.get("bot_color", "#4a9eff")
bot_name  = config.get("bot_name", "AI Assistant")
welcome   = config.get("bot_welcome_message", "Hi! How can I help?")

# ─── Session State ────────────────────────────────────────
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_processed" not in st.session_state:
    st.session_state.last_processed = ""
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

# ─── Load Agent ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        from framework.agents.file_agent import (
            FileRAGAgent, FileAgentConfig
        )

        agent_config = FileAgentConfig(
            llm_model=config["model"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            chunk_size=config["chunk_size"],
            system_prompt=config["system_prompt"],
            persist_dir="vector_store",
        )

        agent = FileRAGAgent(agent_config)

        upload_dir = "uploaded_files"
        if os.path.exists(upload_dir):
            files = [
                os.path.join(upload_dir, f)
                for f in os.listdir(upload_dir)
                if os.path.isfile(os.path.join(upload_dir, f))
            ]
            if files:
                agent.add_files(files)
                agent.index_files()
                return agent

        return None

    except Exception:
        return None

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ─── Helper ───────────────────────────────────────────────
def html(content: str):
    st.markdown(content, unsafe_allow_html=True)

def make_links_clickable(text: str) -> str:
    """Convert URLs in text to clickable HTML links"""
    text = text.replace("\n", "<br>")
    url_pattern = r'(https?://[^\s<>"]+)'
    return re.sub(
        url_pattern,
        r'<a href="\1" target="_blank" '
        r'style="color:#4a9eff;word-break:break-all;">\1</a>',
        text
    )

# ─── CSS ──────────────────────────────────────────────────
html(f"""
<style>
    #MainMenu                        {{display: none !important;}}
    footer                           {{display: none !important;}}
    header                           {{display: none !important;}}
    .stAppDeployButton               {{display: none !important;}}
    div[data-testid="stSpinner"]     {{display: none !important;}}
    div[data-testid="stStatusWidget"]{{display: none !important;}}
    div[data-testid="stToast"]       {{display: none !important;}}
    .stSpinner                       {{display: none !important;}}

    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }}

    /* Fake website */
    .fake-navbar {{
        background: #222;
        color: white;
        padding: 16px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: Arial, sans-serif;
    }}
    .fake-navbar-logo  {{ font-size: 20px; font-weight: bold; }}
    .fake-navbar-links {{
        display: flex;
        gap: 24px;
        font-size: 14px;
        color: #ccc;
    }}
    .fake-hero {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 90px 32px;
        text-align: center;
        font-family: Arial, sans-serif;
    }}
    .fake-hero h1 {{ font-size: 44px; margin-bottom: 14px; }}
    .fake-hero p  {{ font-size: 18px; opacity: 0.9; }}
    .fake-content {{
        padding: 50px 32px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        max-width: 1100px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }}
    .fake-card {{
        background: white;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    .fake-card h3 {{ color: #333; margin-bottom: 10px; }}
    .fake-card p  {{ color: #666; font-size: 14px; line-height: 1.6; }}

    /* Chat header */
    .chat-header-bar {{
        background: {bot_color};
        color: white;
        padding: 12px 16px;
        border-radius: 12px 12px 0 0;
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: Arial, sans-serif;
    }}
    .chat-avatar {{
        width: 34px;
        height: 34px;
        background: rgba(255,255,255,0.25);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }}
    .chat-name   {{ font-weight: bold; font-size: 15px; }}
    .chat-status {{
        font-size: 11px;
        opacity: 0.85;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 2px;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        background: #44ff88;
        border-radius: 50%;
    }}

    /* Messages */
    .bot-msg {{
        background: white;
        color: #333;
        padding: 10px 14px;
        border-radius: 18px 18px 18px 4px;
        max-width: 85%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.09);
        font-size: 14px;
        line-height: 1.5;
        word-wrap: break-word;
        margin-bottom: 8px;
    }}
    .user-msg {{
        background: {bot_color};
        color: white;
        padding: 10px 14px;
        border-radius: 18px 18px 4px 18px;
        max-width: 85%;
        margin-left: auto;
        font-size: 14px;
        line-height: 1.5;
        word-wrap: break-word;
        margin-bottom: 8px;
    }}

    /* Typing indicator */
    .typing-indicator {{
        display: flex;
        gap: 4px;
        padding: 10px 14px;
        background: white;
        border-radius: 18px 18px 18px 4px;
        width: fit-content;
        box-shadow: 0 1px 4px rgba(0,0,0,0.09);
        margin-bottom: 8px;
    }}
    .typing-dot {{
        width: 8px; height: 8px;
        background: #bbb;
        border-radius: 50%;
        animation: typingBounce 1s infinite;
    }}
    .typing-dot:nth-child(2) {{ animation-delay: 0.15s; }}
    .typing-dot:nth-child(3) {{ animation-delay: 0.30s; }}
    @keyframes typingBounce {{
        0%, 100% {{ transform: translateY(0); }}
        50%       {{ transform: translateY(-6px); }}
    }}

    /* Form */
    div[data-testid="stForm"] {{
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        border-top: 1px solid #eee !important;
        margin-top: 0 !important;
    }}
    .stTextInput input {{
        border-radius: 20px !important;
        font-size: 13px !important;
        border: 1px solid #ddd !important;
        background: #f8f9fa !important;
        color: #333 !important;
        padding: 8px 14px !important;
    }}
    .stTextInput input:focus {{
        border-color: {bot_color} !important;
        outline: none !important;
        box-shadow: 0 0 0 2px {bot_color}33 !important;
    }}
    .stTextInput label {{ display: none !important; }}

    div[data-testid="stForm"] button[kind="primaryFormSubmit"],
    div[data-testid="stForm"] button[type="submit"] {{
        background: {bot_color} !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        width: 100% !important;
    }}
    .close-btn button {{
        background: transparent !important;
        color: #999 !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        padding: 4px 10px !important;
        width: 100% !important;
    }}
    .open-btn button {{
        background: {bot_color} !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        font-size: 14px !important;
        padding: 10px !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }}
</style>
""")

# ─── PAGE LAYOUT ──────────────────────────────────────────
left_col, right_col = st.columns([3, 1])

# ─── LEFT: Fake Website ───────────────────────────────────
with left_col:
    html("""
    <div class="fake-navbar">
        <div class="fake-navbar-logo">🏢 MyCompany</div>
        <div class="fake-navbar-links">
            <span>Home</span>
            <span>About</span>
            <span>Services</span>
            <span>Contact</span>
        </div>
    </div>
    <div class="fake-hero">
        <h1>Welcome to MyCompany</h1>
        <p>This is a demo showing how the chat widget works</p>
    </div>
    <div class="fake-content">
        <div class="fake-card">
            <h3>📦 Our Products</h3>
            <p>We offer a wide range of products designed
               to help your business grow and succeed.</p>
        </div>
        <div class="fake-card">
            <h3>🛠️ Our Services</h3>
            <p>Professional services tailored to your needs.
               Our team is ready to help you succeed.</p>
        </div>
        <div class="fake-card">
            <h3>📞 Support</h3>
            <p>24/7 support available. Use the chat widget
               on the right to get instant answers.</p>
        </div>
    </div>
    """)

# ─── RIGHT: Chat Widget ───────────────────────────────────
with right_col:

    if not st.session_state.chat_open:
        html("<br><br><br><br><br><br><br>")
        html('<div class="open-btn">')
        if st.button(
            "💬  Chat with us",
            key="open_chat",
            use_container_width=True
        ):
            st.session_state.chat_open = True
            st.rerun()
        html('</div>')

    else:
        # Header
        html(f"""
        <div class="chat-header-bar">
            <div class="chat-avatar">🤖</div>
            <div>
                <div class="chat-name">{bot_name}</div>
                <div class="chat-status">
                    <span class="status-dot"></span>
                    Online · Ready to help
                </div>
            </div>
        </div>
        """)

        # Close button
        html('<div class="close-btn">')
        if st.button(
            "✕  Close chat",
            key="close_chat",
            use_container_width=True
        ):
            st.session_state.chat_open = False
            st.session_state.is_typing = False
            st.rerun()
        html('</div>')

        # Messages
        messages_container = st.container(height=350)
        with messages_container:
            html(f'<div class="bot-msg">🤖 {welcome}</div>')

            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    html(
                        f'<div class="user-msg">'
                        f'{msg["content"]}'
                        f'</div>'
                    )
                else:
                    formatted = make_links_clickable(
                        msg["content"]
                    )
                    html(
                        f'<div class="bot-msg">'
                        f'{formatted}'
                        f'</div>'
                    )

            # Typing dots - ONLY dots here nothing else
            if st.session_state.is_typing:
                html("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                """)

        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                placeholder=f"Message {bot_name}...",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button(
                "Send ➤",
                use_container_width=True,
                type="primary"
            )

# ─── Handle Submit ────────────────────────────────────────
# OUTSIDE all columns and blocks - correct indent level
if st.session_state.chat_open:
    if submitted and user_input.strip():
        prompt = user_input.strip()

        if (
            prompt != st.session_state.last_processed
            and not st.session_state.is_typing
        ):
            st.session_state.last_processed = prompt
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt
            })
            st.session_state.is_typing = True
            st.rerun()

# ─── Process Agent Response ───────────────────────────────
# OUTSIDE all columns and blocks - correct indent level
# Only runs when is_typing is True
if st.session_state.is_typing:

    messages = st.session_state.chat_messages
    pending_question = None

    # Find last user message
    for msg in reversed(messages):
        if msg["role"] == "user":
            pending_question = msg["content"]
            break

    # Only process if last message is from user
    last_is_user = (
        len(messages) > 0 and
        messages[-1]["role"] == "user"
    )

    if pending_question and last_is_user:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            agent = load_agent()
            if agent:
                response = agent.ask(
                    pending_question,
                    show_sources=False
                )
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.files_used
                })
            else:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": (
                        "I am not available right now. "
                        "Please try again later."
                    ),
                    "sources": []
                })

        except Exception:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Something went wrong. Please try again.",
                "sources": []
            })

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # Turn off typing and show response
    st.session_state.is_typing = False
    st.rerun()