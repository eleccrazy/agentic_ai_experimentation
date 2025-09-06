"""
File: function_chaining.py
Description:
    This module demonstrates prompt templating and function chaining
    concepts using LangChain's `PromptTemplate`.
Author: Gizachew Kassa
Date: 06-09-2025
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
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


def simple_function_chaining(llm: str) -> None:
    """
    Demonstrates a multi-step function chaining example with LangChain.

    Workflow:
        1. Define a prompt template (`product_prompt`) that generates 3 product
           names from a given company.
        2. Define another prompt template (`product_description_prompt`) that
           produces a short description (max 5 words) for each product.
        3. Use `StrOutputParser` to normalize outputs into strings.
        4. Build chains:
            - product_chain: company → products
            - description_chain: product → short description
        5. Combine them into `main_chain` to go from company → products → descriptions.
        6. Invoke the chain with a sample company (Google) and print results.

    Args:
        llm (str): The LLM model identifier (e.g., "gemini-1.5-flash").
    """
    model = get_llm(llm)

    # Create a prompt template for a product
    product_prompt = PromptTemplate(
        input_variables=["company"],
        template="Give me 3 product names from the company {company}:",
    )

    # Create prompt template for product descriptionnnn
    product_description_prompt = PromptTemplate(
        input_variables=["product"],
        template="Can you tell me a short description about {product} in one sentence. Maximum 5 words.",
    )

    # Output parser to convert model output to string
    output_parser = StrOutputParser()

    # Create a chain by binding the prompt to the model
    product_chain = product_prompt | model | output_parser

    description_chain = product_description_prompt | model | output_parser

    # Define a function to create the combined input for the description chain
    def create_description_input(output):
        return {"product": output}

    # Chain everything together
    main_chain = product_chain | create_description_input | description_chain

    result = main_chain.invoke({"company": "Google"})

    print(result)


if __name__ == "__main__":
    """
    Entry point for running the script.
    """
    # simple_templating("gemini-1.5-flash")
    # customer_support_usecase_templating("gemini-1.5-flash")
    simple_function_chaining("gemini-1.5-flash")
