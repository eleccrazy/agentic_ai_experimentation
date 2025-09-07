"""
File: vector_databases.py
Description:
    This module demonstrates how to work with vector databases and embeddings
    for semantic search and Retrieval-Augmented Generation (RAG) workflows.

    Functionality included:
        - `simple_embedding_usage`: Demonstrates embeddings on small sample documents.
        - `process_document_file`: Processes large log files (e.g., dmesg/auth logs),
          splits into chunks, embeds with HuggingFace, and stores in Chroma.
        - `main`: Runs a semantic search query on a log file to retrieve
          contextually relevant entries.

    Use Cases:
        - Experimenting with vector databases for semantic search.
        - Building RAG pipelines.
        - Querying long log files using natural language (e.g., “Were there USB
          device connection or disconnection events?”).

Author: Gizachew Kassa
Date: 07/09/2025
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from paths import EMBEDDING_FPATH


def simple_embedding_usage() -> None:
    """
    Main demonstration of vector database usage for semantic search.

    Workflow:
        1. Initialize a HuggingFace embedding model (`all-MiniLM-L6-v2`).
        2. Create a small set of sample documents with metadata.
        3. Store the documents in a Chroma vector database.
        4. Perform a semantic similarity search for a given query.
        5. Print the retrieved documents and their similarity scores.

    Notes:
        - In a real-world RAG system, the documents would be loaded
          from external sources (e.g., PDFs, APIs, databases).
        - The similarity score is distance-based:
            Lower scores = higher semantic similarity.
    """
    # 1. Create an embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Example documents and metadata (mocked for demo purposes)
    texts = [
        "Vector databases enable semantic search by storing embeddings.",
        "RAG systems combine retrieval with language model generation.",
        "Embeddings capture semantic meaning in numerical form.",
    ]
    metadatas = [
        {"topic": "databases", "type": "technical"},
        {"topic": "AI", "type": "technical"},
        {"topic": "ML", "type": "technical"},
    ]

    # Wrap texts and metadata into LangChain Document objects
    documents = [
        Document(page_content=text, metadata=metadatas[i])
        for i, text in enumerate(texts)
    ]

    # 3. Create the vector store (Chroma) with the documents and embeddings
    vectorstore = Chroma.from_documents(documents, embeddings)

    # 4. Perform semantic similarity search
    # The query does not need to match exact words, just semantic meaning
    results = vectorstore.similarity_search_with_score("What are Vector Databases", k=2)

    # 5. Print results with scores and metadata
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Text: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 3)


def process_document_file(file_path: str) -> Chroma:
    """
    Complete document processing pipeline for a single log file.

    Steps:
        1. Read the entire file as text.
        2. Split the text into manageable chunks (default: 500 chars with 50 overlap).
        3. Wrap each chunk into a `Document` with metadata: source file and chunk ID.
        4. Generate embeddings for each chunk using HuggingFace `all-MiniLM-L6-v2`.
        5. Store the embedded chunks in a Chroma vector database for semantic search.

    Args:
        file_path (str): Path to the log file.

    Returns:
        Chroma: A searchable vector store containing document chunks and embeddings.
    """
    # Read the document
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split intelligently
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Create documents with metadata
    documents = [
        Document(page_content=chunk, metadata={"source": file_path, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    # Create searchable vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)

    return vectorstore


def main() -> None:
    """
    Demonstrates semantic search on a log file.

    Example query:
        "Were there USB device connection or disconnection events?"

    Prints the top-k matching document chunks and their similarity scores.
    """
    store = process_document_file(EMBEDDING_FPATH)
    results = store.similarity_search_with_score(
        "Were there USB device connection or disconnection events?", k=3
    )

    # Print results
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Text: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 3)


if __name__ == "__main__":
    """
    Entry point for running the module.

    This section demonstrates two workflows:

    1. `simple_embedding_usage`:
        - Demonstrates semantic search on small sample documents.
        - Example query: "What are Vector Databases"
        - Example Output:
            Score: 0.688
            Text: Vector databases enable semantic search by storing embeddings.
            Metadata: {'topic': 'databases', 'type': 'technical'}
            ---
            Score: 1.470
            Text: Embeddings capture semantic meaning in numerical form.
            Metadata: {'topic': 'ML', 'type': 'technical'}
        - Notes: Lower scores indicate higher semantic similarity.

    2. `main` (log file semantic search):
        - Processes a large log file using `process_document_file`:
            * Reads file and splits into 500-char overlapping chunks
            * Embeds chunks with HuggingFace model
            * Stores in Chroma vector store
        - Example query: "Were there USB device connection or disconnection events?"
        - Example Output:
            Score: 1.080
            Text: [   15.224743] kernel: loop3: detected capacity change ...
            Metadata: {'source': 'logfile.log', 'chunk_id': 667}
            ---
            Score: 1.159
            Text: [    0.851907] kernel: usb usb2: Manufacturer: Linux 6.8.0-71-generic ...
            Metadata: {'source': 'logfile.log', 'chunk_id': 879}
            ---
            Score: 1.164
            Text: [    2.284638] kernel: usb 1-7: new full-speed USB device number 6 ...
            Metadata: {'source': 'logfile.log', 'chunk_id': 644}
        - Notes: The system retrieves log entries by semantic meaning rather than exact keywords.
                 Lower scores indicate higher similarity.
    """
    # simple_embedding_usage()
    main()
