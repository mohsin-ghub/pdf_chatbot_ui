import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB (in-memory, or set persist_directory)
client = chromadb.Client(Settings())
collection = client.get_or_create_collection(name="pdf_chatbot")

# Load TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load embedding model (needed for Chroma)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: Read and extract text from PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper: Split text into chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Embed and add chunks to vector DB
def store_chunks(chunks):
    embeddings = embedding_model.encode(chunks).tolist()
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], embeddings=[embeddings[i]], ids=[str(i)])

# Search top relevant chunk(s)
def search_similar_chunks(query, top_k=1):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0][0] if results["documents"] else ""

# Generate response using LLM and retrieved context
def get_response(query):
    context = search_similar_chunks(query)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("📄 PDF Chatbot (RAG-based)")

uploaded_file = st.file_uploader("📂 Drag & drop or browse a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")
    pdf_text = read_pdf(uploaded_file)
    chunks = split_text(pdf_text)
    store_chunks(chunks)
    st.info("✅ Text processed and stored for Q&A.")

    user_input = st.text_input("❓ Ask a question about the document:")
    if user_input:
        response = get_response(user_input)
        st.markdown(f"**🧠 Response:** {response}")
