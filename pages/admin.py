# pages/admin.py

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Admin Panel",
    page_icon="⚙️",
    layout="wide"
)

# ─── Load Config ──────────────────────────────────────────
def load_config() -> dict:
    with open("admin_config.json", "r") as f:
        return json.load(f)

def save_config(config: dict):
    with open("admin_config.json", "w") as f:
        json.dump(config, f, indent=4)

# ─── Session State ────────────────────────────────────────
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "agent" not in st.session_state:
    st.session_state.agent = None

# ─── Login Page ───────────────────────────────────────────
def show_login():
    st.title("⚙️ Admin Panel")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Login")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password"
        )

        if st.button("Login", use_container_width=True, type="primary"):
            config = load_config()
            if password == config["password"]:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Wrong password")

# ─── Admin Dashboard ──────────────────────────────────────
def show_dashboard():
    config = load_config()

    st.title("⚙️ Admin Panel")

    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

    st.markdown("---")

    # pages/admin.py
    # Find the tabs section and add a new legal tab

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Files",
        "🤖 Model Settings",
        "🎨 Chat Appearance",
        "⚖️ Legal Settings",       # ← new tab
        "🔐 Security"
    ])
    # ── Tab 1: Files ──────────────────────────────────────
    with tab1:
        st.subheader("📁 Manage Files")

        uploaded_files = st.file_uploader(
            "Upload files for the chatbot to use",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv", "md", "json"],
        )

        if uploaded_files:
            if st.button(
                "🔍 Index Files",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Processing and indexing files..."):
                    try:
                        from framework.agents.file_agent import (
                            FileRAGAgent,
                            FileAgentConfig
                        )

                        # Save files
                        os.makedirs("uploaded_files", exist_ok=True)
                        saved_paths = []

                        for f in uploaded_files:
                            path = os.path.join("uploaded_files", f.name)
                            with open(path, "wb") as file:
                                file.write(f.getbuffer())
                            saved_paths.append(path)

                        # Create and index agent
                        agent_config = FileAgentConfig(
                            llm_model=config["model"],
                            temperature=config["temperature"],
                            top_k=config["top_k"],
                            chunk_size=config["chunk_size"],
                            system_prompt=config["system_prompt"]
                        )

                        agent = FileRAGAgent(agent_config)
                        agent.add_files(saved_paths)
                        agent.index_files()

                        # Save agent to session
                        st.session_state.agent = agent

                        # Update config
                        config["files_indexed"] = True
                        save_config(config)

                        st.success(
                            f"✅ {len(uploaded_files)} files indexed!"
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Show indexed files
        st.markdown("---")
        st.subheader("Currently Indexed Files")

        upload_dir = "uploaded_files"
        if os.path.exists(upload_dir):
            files = os.listdir(upload_dir)
            if files:
                for f in files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"📄 {f}")
                    with col2:
                        if st.button("🗑️", key=f"del_{f}"):
                            os.remove(os.path.join(upload_dir, f))
                            st.rerun()
            else:
                st.info("No files indexed yet")
        else:
            st.info("No files uploaded yet")

    # ── Tab 2: Model Settings ─────────────────────────────
    with tab2:
        st.subheader("🤖 Model Settings")

        new_model = st.selectbox(
            "Model",
            options=[
                "claude-haiku-4-5",
                "claude-sonnet-4-5"
            ],
            index=0 if config["model"] == "claude-haiku-4-5" else 1
        )

        new_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(config["temperature"]),
            step=0.1,
            help="Lower = more precise answers"
        )

        new_top_k = st.slider(
            "Sources to retrieve",
            min_value=1,
            max_value=10,
            value=int(config["top_k"]),
            help="How many document chunks to search"
        )

        new_chunk_size = st.select_slider(
            "Chunk Size",
            options=[256, 512, 1024, 2048],
            value=int(config["chunk_size"]),
            help="Size of document chunks"
        )

        new_system_prompt = st.text_area(
            "System Prompt",
            value=config["system_prompt"],
            height=150,
            help="Instructions that define how the bot behaves"
        )

        if st.button(
            "💾 Save Model Settings",
            type="primary",
            use_container_width=True
        ):
            config["model"] = new_model
            config["temperature"] = new_temperature
            config["top_k"] = new_top_k
            config["chunk_size"] = new_chunk_size
            config["system_prompt"] = new_system_prompt
            save_config(config)
            st.success("✅ Settings saved!")

    # ── Tab 3: Chat Appearance ────────────────────────────
    with tab3:
        st.subheader("🎨 Chat Widget Appearance")

        new_bot_name = st.text_input(
            "Bot Name",
            value=config["bot_name"],
            help="Name shown at top of chat widget"
        )

        new_welcome = st.text_area(
            "Welcome Message",
            value=config["bot_welcome_message"],
            help="First message visitors see"
        )

        new_color = st.color_picker(
            "Chat Color",
            value=config["bot_color"]
        )

        # Preview
        st.markdown("---")
        st.subheader("Preview")
        st.markdown(f"""
        <div style="
            border: 1px solid {new_color};
            border-radius: 15px;
            padding: 20px;
            max-width: 400px;
            background: #1a1f2e;
        ">
            <div style="
                color: {new_color};
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            ">
                🤖 {new_bot_name}
            </div>
            <div style="
                background: #2d3748;
                border-radius: 10px;
                padding: 10px;
                color: white;
                border-left: 3px solid {new_color};
            ">
                {new_welcome}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "💾 Save Appearance",
            type="primary",
            use_container_width=True
        ):
            config["bot_name"] = new_bot_name
            config["bot_welcome_message"] = new_welcome
            config["bot_color"] = new_color
            save_config(config)
            st.success("✅ Appearance saved!")
    
    # Add this new tab content after tab3

    # ── Tab 4: Legal Settings ─────────────────────────────
    with tab4:
        st.subheader("⚖️ Legal Reference Settings")

        st.info(
            "When a visitor asks a legal question the bot "
            "will automatically add a reference section "
            "with the official document link."
        )

        st.markdown("---")

        # Legal keywords preview
        with st.expander("📋 Legal trigger keywords"):
            st.markdown("""
            The bot detects legal questions when these
            words appear in the question:

            `regulation` `law` `legal` `legislation`
            `directive` `compliance` `article` `section`
            `require` `obligation` `prohibited` `penalty`
            `mandatory` `must` `shall` `liability`
            `rights` `jurisdiction` `exemption` `scope`
            """)

        st.markdown("---")

        new_legal_name = st.text_input(
            "Regulation / Law Name",
            value=config.get(
                "legal_reference_name",
                "EU Regulation 2024/1157"
            ),
            help="The official name of the regulation"
        )

        new_legal_url = st.text_input(
            "Official Document URL",
            value=config.get(
                "legal_reference_url",
                "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A02024R1157-20250109&qid=1773755883217"
            ),
            help="Link to the official regulation text"
        )

        new_legal_prompt = st.text_area(
            "Legal System Prompt",
            value=config.get(
                "legal_system_prompt",
                "You are a knowledgeable assistant that answers "
                "legal and regulatory questions clearly and accurately. "
                "When answering legal questions always be precise "
                "and thorough. At the end of your answer always add "
                "a reference section that mentions the relevant "
                "regulation by its proper name. Always include the "
                "official EU regulation link when relevant. "
                "Never mention internal files or excerpts. "
                "Answer clearly so non-lawyers can understand."
            ),
            height=150,
            help="Instructions for how the bot handles legal questions"
        )

        # Preview what the reference will look like
        st.markdown("---")
        st.subheader("Preview")
        st.markdown(f"""
        This is how the legal reference will appear
        at the bottom of legal answers:

        ---
        📋 **Legal Reference:**
        This information is based on **{new_legal_name}**.
        For the full official text visit:
        [{new_legal_url}]({new_legal_url})

        ---
        """)

        if st.button(
            "💾 Save Legal Settings",
            type="primary",
            use_container_width=True
        ):
            config["legal_reference_name"] = new_legal_name
            config["legal_reference_url"]  = new_legal_url
            config["legal_system_prompt"]  = new_legal_prompt
            save_config(config)
            st.success("✅ Legal settings saved!")

    # ── Tab 4: Security ───────────────────────────────────
    with tab4:
        st.subheader("🔐 Change Password")

        current = st.text_input(
            "Current Password",
            type="password"
        )
        new_pass = st.text_input(
            "New Password",
            type="password"
        )
        confirm = st.text_input(
            "Confirm New Password",
            type="password"
        )

        if st.button(
            "🔐 Update Password",
            type="primary",
            use_container_width=True
        ):
            if current != config["password"]:
                st.error("❌ Current password is wrong")
            elif new_pass != confirm:
                st.error("❌ New passwords do not match")
            elif len(new_pass) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                config["password"] = new_pass
                save_config(config)
                st.success("✅ Password updated!")

# ─── Main ─────────────────────────────────────────────────
if not st.session_state.admin_logged_in:
    show_login()
else:
    show_dashboard()