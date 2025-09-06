"""
File: function_chaining.py
Description:
    This module demonstrates prompt templating and function chaining
    concepts using LangChain's `PromptTemplate`.
Author: Gizachew Kassa
Date: 06-09-2025
"""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from llms import get_llm

# Load environment variables (e.g., API keys)
load_dotenv()


def templating(llm: str) -> None:
    """
    Demonstrates prompt templating and simple function chaining with an LLM.

    Args:
        llm (str): The LLM model identifier (e.g., "gemini-1.5-flash").
    """
    # Create a prompt template with a placeholder {company}
    template = PromptTemplate(
        input_variables=["company"],
        template="Who is the CEO of {company}?",
    )

    # Load the specified LLM
    model = get_llm(llm)

    # Iterate over a set of companies, format prompt, and query the model
    for company in ["Meta", "Apple", "Tesla", "Google", "OpenAI"]:
        formatted_prompt = template.format(company=company)
        print("Formatted prompt:", formatted_prompt)

        response = model.invoke(formatted_prompt)
        print("\nResponse:", response.content, "\n----------------")


if __name__ == "__main__":
    """
    Entry point for running the script.
    """
    templating("gemini-1.5-flash")
