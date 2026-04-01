# api_debug.py - version with better error handling

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import os
import sys
import io
import traceback

load_dotenv()

# Don't suppress errors during startup
# os.environ["SILENT_MODE"] = "true"

from framework.agents.file_agent import FileRAGAgent, FileAgentConfig

app = FastAPI(
    title="File RAG Agent API",
    description="Ask questions about your documents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
print("\n[STARTUP] Loading config...")
try:
    with open("admin_config.json") as f:
        config = json.load(f)
    print("[STARTUP] ✅ Config loaded")
except Exception as e:
    print(f"[STARTUP] ❌ Failed to load config: {e}")
    config = {}

# Cache agent
agent = None
agent_error = None

def get_agent():
    """Get or create agent instance"""
    global agent, agent_error

    if agent_error:
        raise Exception(agent_error)

    if agent is None:
        print("\n[STARTUP] Initializing agent...")
        try:
            agent_config = FileAgentConfig(
                llm_model=config.get("model", "claude-haiku-4-5"),
                temperature=config.get("temperature", 0.1),
                top_k=config.get("top_k", 5),
                chunk_size=config.get("chunk_size", 512),
                system_prompt=config.get("system_prompt", ""),
                legal_system_prompt=config.get(
                    "legal_system_prompt", ""
                ),
                legal_files=config.get("legal_files", []),
                persist_dir="vector_store"
            )

            print("[STARTUP] Creating FileRAGAgent...")
            agent = FileRAGAgent(agent_config)

            print("[STARTUP] ✅ Agent initialized successfully")
            return agent

        except Exception as e:
            error_msg = (
                f"Failed to initialize agent: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            print(f"[STARTUP] ❌ {error_msg}")
            agent_error = error_msg
            raise Exception(agent_error)

    return agent

# Models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    files_used: list

# Endpoints
@app.get("/")
async def root():
    return {
        "status": "running",
        "endpoints": {
            "health": "GET /api/health",
            "info": "GET /api/info",
            "chat": "POST /api/chat",
            "docs": "GET /docs"
        }
    }

@app.get("/api/health")
async def health():
    try:
        agent = get_agent()
        return {
            "status": "healthy",
            "agent_loaded": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "agent_loaded": False,
            "error": str(e)
        }

@app.get("/api/info")
async def info():
    return {
        "bot_name": config.get("bot_name", "AI Assistant"),
        "model": config.get("model", "unknown"),
        "files_indexed": config.get("files_indexed", False)
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        if not config.get("files_indexed"):
            raise HTTPException(
                status_code=503,
                detail="Files not indexed yet"
            )

        agent = get_agent()
        response = agent.ask(
            request.question,
            show_sources=False
        )

        return ChatResponse(
            answer=response.answer,
            files_used=response.files_used
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting File RAG Agent API")
    print("="*60)
    print("Open browser: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )