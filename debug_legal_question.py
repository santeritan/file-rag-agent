# debug_legal_question.py

import os
import json
from dotenv import load_dotenv
load_dotenv()

# Disable silent mode
os.environ["SILENT_MODE"] = "false"

with open("admin_config.json") as f:
    config = json.load(f)

from framework.agents.file_agent import FileRAGAgent, FileAgentConfig

agent_config = FileAgentConfig(
    llm_model=config["model"],
    temperature=config["temperature"],
    top_k=config["top_k"],
    chunk_size=config["chunk_size"],
    system_prompt=config["system_prompt"],
    legal_system_prompt=config.get("legal_system_prompt", ""),
    legal_files=config.get("legal_files", []),
    persist_dir="vector_store"
)

agent = FileRAGAgent(agent_config)

# Ask the exact question
question = "who has to fill the annex vii?"

print(f"\n{'='*60}")
print(f"Question: {question}")
print(f"{'='*60}\n")

response = agent.ask(question, show_sources=True)

print(f"\n{'='*60}")
print(f"ANSWER:\n{response.answer}")
print(f"{'='*60}\n")