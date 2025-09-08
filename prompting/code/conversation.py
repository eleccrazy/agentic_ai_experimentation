"""
File: conversation.py
Description:
    This module demonstrates a conversational example using LangChain LLMs
    with publication context and follow-up functionality (f.f). It shows how
    to maintain conversation history so the AI can answer follow-up questions
    naturally.

Author: Gizachew Kassa
Date: 08/09/2025
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llms import get_llm
from utils import load_publication


def basic_question(model: str) -> None:
    """
    Sends publication-contextualized questions to the LLM and prints responses,
    demonstrating follow-up capability.

    Workflow:
        1. Load the specified LLM model.
        2. Load publication content using `load_publication()`.
        3. Initialize conversation with a system message containing the publication.
        4. Send a first human question and print the AI's response.
        5. Add the AI response to conversation history.
        6. Send a follow-up human question; AI uses prior context to answer naturally.

    Args:
        model (str): The LLM model identifier (e.g., "gemini-1.5-flash").
    """
    llm = get_llm(model)

    # Load the content of the publication for context
    publication_content = load_publication()

    # Initialize conversation with publication context
    conversation = [
        SystemMessage(
            content=f"""
You are a helpful AI assistant discussing a research publication.
Base your answers only on this publication content:

{publication_content}
"""
        )
    ]

    # User question 1
    conversation.append(
        HumanMessage(
            content="""
What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
"""
        )
    )

    # AI response to first question
    response1 = llm.invoke(conversation)
    print("AI Response to Question 1:")
    print(response1.content)
    print("\n" + "=" * 50 + "\n")

    # Add AI response to conversation history (f.f)
    conversation.append(AIMessage(content=response1.content))

    # User question 2 (follow-up)
    conversation.append(
        HumanMessage(
            content="""
How does it work in case of anomaly detection?
"""
        )
    )

    # AI response to follow-up question
    response2 = llm.invoke(conversation)
    print("AI Response to Question 2 (Follow-up):")
    print(response2.content)


if __name__ == "__main__":
    """
    Entry point for running the demo.

    Demonstrates:
        - Sending publication-contextualized questions to the LLM.
        - Maintaining conversation history for follow-up questions.
        - Using the `basic_question` function with the "gemini-1.5-flash" model.
    """
    basic_question("gemini-1.5-flash")
