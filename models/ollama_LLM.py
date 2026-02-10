import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

ollama_model = ChatOllama(
    model="llama3.1:8b",  
    temperature=0.0,
    base_url="http://localhost:11434",
    format="json" 
)