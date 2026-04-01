# create debug_agent.py and run it

import os
import sys
from dotenv import load_dotenv
load_dotenv()

print("Step 1: Checking uploaded_files folder...")
upload_dir = "uploaded_files"

if os.path.exists(upload_dir):
    files = os.listdir(upload_dir)
    print(f"  Found {len(files)} files: {files}")
else:
    print("  ERROR: uploaded_files folder does not exist")

print("\nStep 2: Checking vector_store folder...")
if os.path.exists("vector_store"):
    print("  vector_store folder exists")
else:
    print("  vector_store folder does not exist yet")

print("\nStep 3: Checking admin_config.json...")
import json
with open("admin_config.json") as f:
    config = json.load(f)
print(f"  model: {config['model']}")
print(f"  files_indexed: {config['files_indexed']}")

print("\nStep 4: Trying to load agent...")
try:
    from framework.agents.file_agent import FileRAGAgent, FileAgentConfig

    agent_config = FileAgentConfig(
        llm_model=config["model"],
        temperature=config["temperature"],
        top_k=config["top_k"],
        chunk_size=config["chunk_size"],
        system_prompt=config["system_prompt"],
        persist_dir="vector_store"
    )

    agent = FileRAGAgent(agent_config)
    print("  Agent created successfully")

    if os.path.exists(upload_dir):
        files = [
            os.path.join(upload_dir, f)
            for f in os.listdir(upload_dir)
            if os.path.isfile(os.path.join(upload_dir, f))
        ]
        print(f"  Files to index: {files}")

        if files:
            agent.add_files(files)
            agent.index_files()
            print("  Files indexed successfully")

            response = agent.ask("hello", show_sources=False)
            print(f"  Test response: {response.answer[:100]}")
        else:
            print("  ERROR: No files found to index")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()