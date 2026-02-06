import streamlit as st
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss

TOP_K = 3

load_dotenv()
genai.configure(api_key= os.getenv("GENAI_API_KEY"))

st.set_page_config(page_title= "RAG chatbot with Gemini", page_icon= "🤖")
st.title("RAG chatbot")

# Read document
with open("employee_data.txt", 'r', encoding="utf-8") as f:
    text = f.read()

# Chunks
def chunk_docs(text, chunk_size = 500, overlap = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_docs(text)

# Embedding
@st.cache_resource
def build_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

index, chunks = build_index(chunks)

# Retrieve

def retrieve(query, k = TOP_K):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0]]

# Gemini

def ask_llm(context, question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a helpful assistant for the company.
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context: {context}
Question: {question}

"""

    response = model.generate_content(prompt)
    return response.text

# UI

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

query = st.chat_input("Ask anything about employee data")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})

    retrieved_chunks = retrieve(query)
    context = "\n\n".join(retrieved_chunks)

    with st.spinner('Generating response'):
        answer = ask_llm(context, query)

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role":"assistant", "content":answer})
