# debug_legal.py - run this to see exact filenames

import os
import json
import sys
from dotenv import load_dotenv
load_dotenv()

# Load config
with open("admin_config.json") as f:
    config = json.load(f)

print("=== Legal files in config ===")
for lf in config.get("legal_files", []):
    print(f"  Configured: '{lf['filename']}'")

print("\n=== Files in uploaded_files folder ===")
upload_dir = "uploaded_files"
for f in os.listdir(upload_dir):
    print(f"  Found: '{f}'")

print("\n=== Testing agent and checking sources ===")
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

files = [
    os.path.join(upload_dir, f)
    for f in os.listdir(upload_dir)
    if os.path.isfile(os.path.join(upload_dir, f))
]

agent.add_files(files)
agent.index_files()

# Test with a legal question
response = agent.ask("who signs the annex vii", show_sources=False)

print("\n=== Files used in response ===")
for f in response.files_used:
    print(f"  Used: '{f}'")

print("\n=== Legal refs detected ===")
legal_refs = agent._get_legal_references(response.files_used)
print(f"  Found {len(legal_refs)} legal references")
for ref in legal_refs:
    print(f"  Ref: {ref}")

print("\n=== Answer ===")
print(response.answer)