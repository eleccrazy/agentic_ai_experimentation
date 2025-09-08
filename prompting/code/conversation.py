"""
File: conversation.py
Description:
    This module demonstrates a simple conversational example using LangChain LLMs.
    It shows how to send system and human messages to the model, optionally
    using the content of a publication as context for the conversation.

Author: Gizachew Kassa
Date: 08/09/2025
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llms import get_llm
from utils import load_publication


def basic_question(model: str) -> None:
    """
    Sends a publication-contextualized question to the LLM and prints the response.

    Workflow:
        1. Load the specified LLM model.
        2. Load publication content using `load_publication()`.
        3. Define a conversation with a system message and a human message
           that includes the publication content as context.
        4. Invoke the LLM and print its response.

    Args:
        model (str): The LLM model identifier (e.g., "gemini-1.5-flash").
    """
    llm = get_llm(model)

    # Load the content of the publication for context
    publication_content = load_publication()

    # Define conversation messages
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(
            content=f"""
            Based on this publication: {publication_content}

            What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
            """
        ),
    ]

    # Invoke the model and print the response
    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    """
    Entry point for running the demo.

    Demonstrates sending a publication-contextualized question to the LLM
    using the `basic_question` function with the "gemini-1.5-flash" model.
    """
    basic_question("gemini-1.5-flash")
