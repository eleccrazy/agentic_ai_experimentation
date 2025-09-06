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


def simple_templating(llm: str) -> None:
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


def customer_support_usecase_templating(llm: str) -> None:
    """
    Demonstrates a customer support use case using prompt templating.

    This function builds a reusable `PromptTemplate` tailored for customer
    support scenarios. The template includes placeholders for customer-specific
    and issue-specific details, which are dynamically filled in before
    invoking the LLM.

    Workflow:
        1. Define a structured template with placeholders:
            - customer_name
            - product_name
            - issue_description
            - previous_interactions
            - tone
        2. Format the template with example customer data.
        3. Pass the formatted prompt to the specified LLM.
        4. Print both the formatted prompt and the model’s response.

    Args:
        llm (str): The LLM model identifier (e.g., "gemini-1.5-flash").

    Example:
        Generates an empathetic but technical response for a customer named
        Alex Smith who is troubleshooting WiFi issues on their SmartHome Hub.
    """
    customer_support_template = PromptTemplate(
        input_variables=[
            "customer_name",
            "product_name",
            "issue_description",
            "previous_interactions",
            "tone",
        ],
        template="""
        You are a customer support specialist for {product_name}.
    
        Customer: {customer_name}
        Issue: {issue_description}
        Previous interactions: {previous_interactions}
    
        Respond to the customer in a {tone} tone. If you don't have enough information to resolve their issue,
        ask clarifying questions. Always prioritize customer satisfaction and accurate information.
        """,
    )

    model = get_llm(llm)
    formatted_prompt = customer_support_template.format(
        # This can now handle all types of customer inquiries with appropriate context
        customer_name="Alex Smith",
        product_name="SmartHome Hub",
        issue_description="Device won't connect to WiFi after power outage",
        previous_interactions="Customer has already tried resetting the device twice.",
        tone="empathetic but technical",
    )
    print("Formatted prompt:", formatted_prompt)

    response = model.invoke(formatted_prompt)
    print("\nResponse:", response.content, "\n----------------")


if __name__ == "__main__":
    """
    Entry point for running the script.
    """
    # simple_templating("gemini-1.5-flash")
    customer_support_usecase_templating("gemini-1.5-flash")
