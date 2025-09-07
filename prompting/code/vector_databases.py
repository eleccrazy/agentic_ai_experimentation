"""
File: vector_databases.py
Description:
    This module demonstrates how to work with vector databases and embeddings
    as part of a Retrieval-Augmented Generation (RAG) workflow.

    Specifically, it shows how to:
        - Create embeddings using HuggingFace models
        - Store documents in a Chroma vector database
        - Perform semantic similarity search over the stored documents

    Vector databases are key components of modern AI systems that enable
    efficient retrieval of contextually relevant documents based on meaning
    rather than keyword matching.

Author: Gizachew Kassa
Date: 07/09/2025
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def main() -> None:
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


if __name__ == "__main__":
    """
    Entry point for running the demo.

    Example Output:
        Score: 0.688
        Text: Vector databases enable semantic search by storing embeddings.
        Metadata: {'topic': 'databases', 'type': 'technical'}
        ---
        Score: 1.470
        Text: Embeddings capture semantic meaning in numerical form.
        Metadata: {'topic': 'ML', 'type': 'technical'}
        ---

    Explanation:
        The system retrieves documents by meaning, not keywords.
        Lower scores indicate higher similarity (0.688 is more relevant than 1.470).
    """
    main()
