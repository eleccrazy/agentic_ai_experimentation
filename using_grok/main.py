"""
File: main.py
Description: This module contains the code for experimenting different opensource llms.
Author: Gizachew Kassa
Date Created: 09/08/2025
"""

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192")
response = llm.invoke("What is agentic AI?")
print(response.content)
