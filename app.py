import streamlit as st
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss


load_dotenv()
genai.configure(api_key= os.getenv("GENAI_API_KEY"))

st.set_page_config(page_title= "RAG chatbot with Gemini", page_icon= "🤖")
st.title("RAG chatbot")

# Read document
with open("employee_data.txt", 'r', encoding="utf-8") as f:
    text = f.read()