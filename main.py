import os

from framework.agents.file_agent import FileRAGAgent, FileAgentConfig


def main():

    

    # ─── Configure Agent ──────────────────────────────────────

    config = FileAgentConfig(

        llm_model="gpt-4o-mini",

        embedding_model="text-embedding-3-small",

        temperature=0.2,

        top_k=5,

        chunk_size=512,

        chunk_overlap=50,

        upload_dir="uploaded_files",

        persist_dir="vector_store",

        system_prompt=(

            "You are a document analyst. Answer based only "

            "on the provided files. Be concise and cite sources."

        )

    )

    

    agent = FileRAGAgent(config)

    

    # ─── Add Your Files ───────────────────────────────────────

    agent.add_files([

        "docs/annual_report.pdf",

        "docs/financial_data.csv",

        "docs/meeting_notes.docx"

    ])

    

    # Or add a whole directory

    # agent.add_directory("my_documents/", recursive=True)

    

    # See what's loaded

    agent.list_files()

    

    # ─── Index Files ──────────────────────────────────────────

    agent.index_files()

    

    # ─── Ask Questions ────────────────────────────────────────

    agent.ask("What is the main topic of these documents?")

    agent.ask("What are the key financial figures?")

    agent.ask("Summarize the meeting notes")

    

    # ─── Interactive Mode ─────────────────────────────────────

    print("\n💬 Interactive Mode (type 'quit' to exit)\n")

    

    while True:

        question = input("You: ").strip()

        

        if question.lower() in ["quit", "exit", "q"]:

            break

        elif question.lower() == "files":

            agent.list_files()

        elif question.lower() == "stats":

            print(agent.get_stats())

        elif question.lower() == "clear":

            agent.clear_history()

        elif question.lower().startswith("add "):

            file_path = question[4:].strip()

            agent.add_file(file_path)

            agent.index_files()

        elif question:

            agent.ask_with_history(question)


if __name__ == "__main__":

    main()