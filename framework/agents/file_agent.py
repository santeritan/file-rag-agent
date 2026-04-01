# framework/agents/file_agent.py

import os
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

import anthropic
from anthropic.types import TextBlock
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from framework.core.file_handler import FileHandler, FileRecord
from framework.components.ingestion import FileIngestionPipeline


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class FileAgentConfig:
    llm_model: str           = "claude-sonnet-4-5"
    embedding_model: str     = "all-MiniLM-L6-v2"
    temperature: float       = 0.1
    top_k: int               = 10
    chunk_size: int          = 1024
    chunk_overlap: int       = 200
    upload_dir: str          = "uploaded_files"
    persist_dir: str         = "vector_store"
    system_prompt: str       = ""
    legal_system_prompt: str = ""
    # Fix: use field(default_factory=list)
    # instead of None for List types
    legal_files: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def __post_init__(self):
        if self.legal_files is None:
            self.legal_files = []


# ─── Response ─────────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    query:      str
    answer:     str
    sources:    List[Dict]
    files_used: List[str]


# ─── Agent ────────────────────────────────────────────────────────────────────

class FileRAGAgent:
    """
    RAG Agent that answers questions based on files you provide.
    Automatically adds legal disclaimers when legal files are used.

    Usage:
        agent = FileRAGAgent()
        agent.add_files(["report.pdf", "regulation.pdf"])
        agent.index_files()
        response = agent.ask("What are the requirements?")
    """

    def __init__(self, config: Optional[FileAgentConfig] = None):

        self.config = config if config is not None else FileAgentConfig()

        # Check API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "\n❌ ANTHROPIC_API_KEY not found\n"
                "Make sure your .env file contains:\n"
                "ANTHROPIC_API_KEY=sk-ant-..."
            )

        # Core components
        self.file_handler = FileHandler(self.config.upload_dir)
        self.ingestion    = FileIngestionPipeline(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )

        # ─── Load existing vector store from disk ──────────
        self.vector_store = self._load_vector_store()

        self.chat_history: List[Dict] = []

        # Only print in non-silent mode
        if not os.environ.get("SILENT_MODE"):
            print(f"🤖 FileRAGAgent ready")
            print(f"   Model     : {self.config.llm_model}")
            print(f"   Embeddings: {self.config.embedding_model}")
            
            if self.vector_store:
                print(f"   Vector Store: ✅ Loaded from disk")
            else:
                print(f"   Vector Store: ⏳ Not indexed yet")
        

    # ─── File Management ──────────────────────────────────────────────────────

    def add_file(self, file_path: str) -> FileRecord:
        """Add a single file"""
        return self.file_handler.add_file(file_path)

    def add_files(self, file_paths: List[str]) -> List[FileRecord]:
        """Add multiple files at once"""
        return self.file_handler.add_files(file_paths)

    def add_directory(
        self,
        dir_path: str,
        recursive: bool = False
    ) -> List[FileRecord]:
        """Add all supported files from a directory"""
        return self.file_handler.add_directory(dir_path, recursive)

    def remove_file(self, file_id: str) -> None:
        """Remove a file - requires re-indexing after"""
        removed = self.file_handler.remove_file(file_id)
        if removed and not os.environ.get("SILENT_MODE"):
            print("⚠️  Please call index_files() to update the index")

    def list_files(self) -> None:
        """Show all loaded files and their status"""
        self.file_handler.print_status()

    # framework/agents/file_agent.py
    # Add this method to FileRAGAgent class

    def _load_vector_store(self):
        """
        Try to load existing vector store from disk.
        Returns None if vector store doesn't exist yet.
        """
        import os
        from langchain_community.vectorstores import Chroma

        persist_dir = self.config.persist_dir

        # Check if persist directory exists
        if not os.path.exists(persist_dir):
            return None

        # Check if it has data
        db_file = os.path.join(persist_dir, "chroma.sqlite3")
        if not os.path.exists(db_file):
            return None

        try:
            # Try to load from disk
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )

            # Test if it has documents
            try:
                # Do a dummy search to verify it's working
                results = vector_store.similarity_search("test", k=1)
                # If we got here, it's working
                return vector_store
            except Exception as e:
                # Vector store exists but is empty or corrupted
                return None

        except Exception as e:
            # Could not load vector store
            return None


    # ─── Indexing ─────────────────────────────────────────────────────────────

    def index_files(self, force_reindex: bool = False) -> None:
        """
        Index files into vector store.
        Loads existing index from disk if available.
        Only indexes new files unless force_reindex=True.
        """
        persist_dir = self.config.persist_dir

        # Load existing vector store from disk if it exists
        if (
            not force_reindex
            and os.path.exists(persist_dir)
            and os.listdir(persist_dir)
        ):
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
                # Mark all files as indexed
                for record in self.file_handler.list_files():
                    self.file_handler.mark_indexed(record.file_id, 0)

                if not os.environ.get("SILENT_MODE"):
                    print("✅ Loaded existing index from disk")
                return

            except Exception:
                pass

        # Decide which files to index
        if force_reindex:
            files_to_index = self.file_handler.list_files()
            if not os.environ.get("SILENT_MODE"):
                print("🔄 Force re-indexing all files...")
        else:
            files_to_index = self.file_handler.get_unindexed_files()

        if not files_to_index:
            if not os.environ.get("SILENT_MODE"):
                print("✅ All files already indexed")
            return

        # Process files into chunks
        chunks = self.ingestion.process_records(files_to_index)

        if not chunks:
            if not os.environ.get("SILENT_MODE"):
                print("❌ No chunks generated - check your files")
            return

        # Build or update vector store
        if self.vector_store is None or force_reindex:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
        else:
            self.vector_store.add_documents(chunks)

        # Mark files as indexed
        chunk_counts = self._count_chunks_per_file(chunks)
        for record in files_to_index:
            count = chunk_counts.get(record.file_id, 0)
            self.file_handler.mark_indexed(record.file_id, count)

        if not os.environ.get("SILENT_MODE"):
            print(
                f"🎯 Index updated: {len(chunks)} chunks "
                f"across {len(files_to_index)} files\n"
            )

    # ─── Querying ─────────────────────────────────────────────────────────────

    # framework/agents/file_agent.py
    # Find the ask() method and add debug output

    def ask(
        self,
        question: str,
        show_sources: bool = True
    ) -> AgentResponse:

        if self.vector_store is None:
            raise RuntimeError(
                "No files indexed. "
                "Call add_files() then index_files() first."
            )

        # Retrieve
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.top_k}
        )

        relevant_chunks = retriever.invoke(question)

        if len(relevant_chunks) < 3:
            broader = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.top_k * 2}
            )
            relevant_chunks = broader.invoke(question)

        if not relevant_chunks:
            return AgentResponse(
                query=question,
                answer="I could not find relevant information.",
                sources=[],
                files_used=[]
            )

        # Build context
        context, sources = self._build_context(relevant_chunks)

        # Get files used
        files_used = list({s["file_name"] for s in sources})

        # ─── DEBUG: Print what we found ───────────────────
        print(f"\n[DEBUG ask()]")
        print(f"  Question: {question}")
        print(f"  Files used: {files_used}")
        print(f"  Legal files config: {[lf.get('filename') for lf in self.config.legal_files]}")

        # Check legal
        legal_refs = self._get_legal_references(files_used)

        # ─── DEBUG: Print legal detection ─────────────────
        print(f"  Legal refs found: {len(legal_refs)}")
        for ref in legal_refs:
            print(f"    - {ref.get('reference_name')}")

        prompt = self._build_prompt(
            question=question,
            context=context,
            is_legal=len(legal_refs) > 0
        )

        answer = self._call_claude(prompt)

        if legal_refs:
            disclaimer = self._build_legal_disclaimer(legal_refs)
            answer = answer + "\n\n" + disclaimer

        self.chat_history.append({
            "question": question,
            "answer": answer,
            "files_used": files_used
        })

        if show_sources:
            self._display_response(answer, sources, show_sources)

        return AgentResponse(
            query=question,
            answer=answer,
            sources=sources,
            files_used=files_used
        )

    def ask_with_history(self, question: str) -> AgentResponse:
        """Ask a question with conversation context from last 3 exchanges"""

        history_context = ""
        if self.chat_history:
            recent = self.chat_history[-3:]
            history_context = "\n".join([
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in recent
            ])

        if history_context:
            augmented = (
                f"Previous conversation:\n{history_context}\n\n"
                f"Current question: {question}"
            )
        else:
            augmented = question

        return self.ask(augmented)

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.chat_history = []
        if not os.environ.get("SILENT_MODE"):
            print("🧹 Conversation history cleared")

    # ─── Legal Handling ───────────────────────────────────────────────────────

    # framework/agents/file_agent.py
    # Add debug to the matching function

    # framework/agents/file_agent.py
    # Replace _get_legal_references completely

    def _get_legal_references(
        self,
        files_used: List[str]
    ) -> List[Dict]:
        """
        Check if any files used are legal documents.
        Uses fuzzy matching to handle filename variations.
        """
        if not self.config.legal_files:
            return []

        legal_refs = []

        for file_used in files_used:
            file_used_lower = file_used.lower().strip()

            for legal_file in self.config.legal_files:
                legal_filename = legal_file.get(
                    "filename", ""
                ).lower().strip()

                if not legal_filename:
                    continue

                # Already added this reference
                if legal_file in legal_refs:
                    continue

                # Strategy 1: Exact match
                if file_used_lower == legal_filename:
                    legal_refs.append(legal_file)
                    continue

                # Strategy 2: Either contains the other
                if (legal_filename in file_used_lower or 
                    file_used_lower in legal_filename):
                    legal_refs.append(legal_file)
                    continue

                # Strategy 3: Extract just the filename (no path/hash)
                # and compare those
                file_used_name = os.path.basename(
                    file_used_lower
                )
                legal_name = os.path.basename(legal_filename)

                if file_used_name == legal_name:
                    legal_refs.append(legal_file)
                    continue

                # Strategy 4: Match on key identifier
                # Like "2024_1157" should match any file with that
                if "2024" in legal_filename and "1157" in legal_filename:
                    if "2024" in file_used_lower and "1157" in file_used_lower:
                        legal_refs.append(legal_file)
                        continue

        return legal_refs

    def _build_legal_disclaimer(
        self,
        legal_refs: List[Dict]
    ) -> str:
        """
        Build disclaimer text to append to answers
        that use legal documents.
        """
        if not legal_refs:
            return ""

        disclaimer_parts = []

        for ref in legal_refs:
            name       = ref.get("reference_name", "")
            url        = ref.get("reference_url", "")
            disclaimer = ref.get(
                "disclaimer",
                "Always consult a qualified legal professional "
                "before making compliance decisions."
            )

            disclaimer_parts.append(
                f"---\n"
                f"📋 Legal Reference: {name}\n"
                f"🔗 {url}\n"
                f"⚠️ {disclaimer}\n"
                f"---"
            )

        return "\n\n".join(disclaimer_parts)

    # ─── Prompt Building ──────────────────────────────────────────────────────

    # ─── REPLACE WITH THIS ────────────────────────────────────
    def _build_prompt(
        self,
        question: str,
        context: str,
        is_legal: bool = False
    ) -> str:
        """
        Build prompt.
        Uses legal system prompt when answer
        comes from a legal document.
        """

        if is_legal and self.config.legal_system_prompt:
            system = self.config.legal_system_prompt
        elif is_legal:
            system = (
                "You are a knowledgeable assistant. "
                "Answer clearly and professionally. "
                "Anyone should be able to understand your answer. "
                "Never mention internal files or excerpts."
            )
        else:
            system = self.config.system_prompt or (
                "You are a friendly and knowledgeable assistant. "
                "Answer questions in a natural conversational way. "
                "Never mention documents, files, excerpts or pages."
            )

        return f"""{system}

    Use the following information to answer the question.
    Do not mention that you are using documents or files.
    Do not quote directly from the text.
    Do not reference excerpt numbers or page numbers.
    Do not say things like according to or based on the documents.
    Just answer naturally and helpfully as if you already know this.

    INFORMATION:
    {context}

    QUESTION: {question}

    ANSWER:"""

    def _build_context(
        self,
        chunks: List[Document]
    ) -> tuple:
        """
        Build clean context string from chunks.
        No excerpt labels or markers shown to the AI.
        Source metadata tracked separately for citations.
        """
        context_parts = []
        sources       = []

        for i, chunk in enumerate(chunks, 1):
            meta      = chunk.metadata
            file_name = meta.get("file_name", "Unknown")
            page      = meta.get("page", "")

            # Clean text only - no labels
            context_parts.append(chunk.page_content.strip())

            sources.append({
                "index":           i,
                "file_name":       file_name,
                "file_id":         meta.get("file_id", ""),
                "file_type":       meta.get("file_type", ""),
                "page":            page,
                "chunk_index":     meta.get("chunk_index", 0),
                "content_preview": chunk.page_content[:100] + "..."
            })

        return "\n\n".join(context_parts), sources

    # ─── Claude API Call ──────────────────────────────────────────────────────

    def _call_claude(self, prompt: str) -> str:
        """
        Call Claude directly using the Anthropic SDK.
        Returns friendly messages instead of technical errors.
        """
        try:
            message = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract only TextBlock content
            text_parts: List[str] = [
                block.text
                for block in message.content
                if isinstance(block, TextBlock)
            ]

            if text_parts:
                return " ".join(text_parts)
            else:
                return (
                    "I am sorry I could not generate a response. "
                    "Please try again."
                )

        except Exception as e:
            return self._get_friendly_error(str(e).lower())

    def _get_friendly_error(self, error_str: str) -> str:
        """Convert technical API errors into friendly messages"""

        if "rate_limit" in error_str or "429" in error_str:
            return (
                "I am currently handling a lot of requests. "
                "Please wait a moment and try again."
            )
        elif "401" in error_str or "authentication" in error_str:
            return (
                "I am having trouble connecting right now. "
                "Please try again in a moment."
            )
        elif "404" in error_str or "not_found" in error_str:
            return (
                "I am currently unavailable. "
                "Please try again later."
            )
        elif "529" in error_str or "overload" in error_str:
            return (
                "I am very busy right now. "
                "Please try again in a few minutes."
            )
        elif "timeout" in error_str or "timed out" in error_str:
            return (
                "It is taking longer than expected to respond. "
                "Please try asking again."
            )
        elif "token" in error_str and "limit" in error_str:
            return (
                "Your question or the documents are too long "
                "for me to process. Please try a shorter question."
            )
        elif "connection" in error_str or "network" in error_str:
            return (
                "I am having trouble connecting. "
                "Please check your connection and try again."
            )
        else:
            return (
                "I am sorry something went wrong. "
                "Please try again in a moment."
            )

    # ─── Display ──────────────────────────────────────────────────────────────

    def _display_response(
        self,
        answer: str,
        sources: List[Dict],
        show_sources: bool
    ) -> None:
        """Display response in terminal - only used in non-silent mode"""

        if os.environ.get("SILENT_MODE"):
            return

        print("\n" + "="*60)
        print("💬 ANSWER:")
        print("="*60)
        print(answer)

        if show_sources and sources:
            print("\n" + "-"*60)
            print("📚 SOURCES:")
            print("-"*60)
            seen_files: set = set()
            for s in sources:
                if s["file_name"] not in seen_files:
                    page     = s.get("page", "")
                    page_str = (
                        f", p.{int(page) + 1}"
                        if page != "" else ""
                    )
                    print(
                        f"  [{s['index']}] "
                        f"{s['file_name']}{page_str}"
                    )
                    seen_files.add(s["file_name"])

        print("="*60 + "\n")

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _count_chunks_per_file(
        self,
        chunks: List[Document]
    ) -> Dict[str, int]:
        """Count how many chunks came from each file"""
        counts: Dict[str, int] = {}
        for chunk in chunks:
            fid = chunk.metadata.get("file_id", "")
            counts[fid] = counts.get(fid, 0) + 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics"""
        files        = self.file_handler.list_files()
        total_chunks = sum(f.chunk_count for f in files)
        return {
            "total_files":     len(files),
            "indexed_files":   len(
                self.file_handler.get_indexed_files()
            ),
            "total_chunks":    total_chunks,
            "questions_asked": len(self.chat_history),
            "model":           self.config.llm_model
        }