"""
File: conversation.py
Description:
    This module demonstrates a simple conversational example using LangChain LLMs.
    It shows how to send system and human messages to the model and receive a response.

Author: Gizachew Kassa
Date: 08/09/2025
"""

from langchain_core.messages import HumanMessage, SystemMessage
from llms import get_llm


def basic_question(model: str) -> None:
    """
    Sends a predefined question to the LLM and prints the response.

    Workflow:
        1. Load the specified LLM model.
        2. Define a conversation with a system message and a human message.
        3. Invoke the LLM and print its response.

    Args:
        model (str): The LLM model identifier (e.g., "gemini-1.5-flash").
    """
    llm = get_llm(model)
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(
            content="What are variational autoencoders and list the top 5 applications for them?"
        ),
    ]
    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    """
    Entry point for running the demo.
    Calls the `basic_question` function using the "gemini-1.5-flash" model.
    """
    basic_question("gemini-1.5-flash")
