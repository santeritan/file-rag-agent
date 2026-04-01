# run.py - replace everything with this

import os
from dotenv import load_dotenv
load_dotenv()

from framework.agents.file_agent import FileRAGAgent, FileAgentConfig

# ─── Setup ────────────────────────────────────────────────
config = FileAgentConfig(
    llm_model="claude-haiku-4-5",
    top_k=5,
    chunk_size=512,
)

# ─── Create Agent ─────────────────────────────────────────
agent = FileRAGAgent(config)

# ─── Get Project Folder Path ──────────────────────────────
# This makes sure we always find the files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Add Your Files Here ──────────────────────────────────
# Using full path to avoid "file not found" errors
agent.add_files([
    os.path.join(BASE_DIR, "uploaded_files", "test_doc.txt"),
])

# ─── Index Files ──────────────────────────────────────────
agent.index_files()

# ─── Show Loaded Files ────────────────────────────────────
agent.list_files()

# ─── Chat Loop ────────────────────────────────────────────
print("\n💬 Ask questions about your files")
print("   Type 'quit' to exit")
print("   Type 'files' to see loaded files\n")

while True:
    question = input("You: ").strip()

    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    elif question.lower() == "files":
        agent.list_files()
    elif question:
        agent.ask(question)