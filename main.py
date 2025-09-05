import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# LLM
from langchain_perplexity import ChatPerplexity
# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():   # only append if not empty
        urls.append(url)

print(urls)
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# -----------------------------
# Perplexity LLM + HuggingFace Embeddings
# -----------------------------
pplx_key = os.getenv("PPLX_API_KEY")
if not pplx_key:
    st.error("Missing PPLX_API_KEY in .env file")
    st.stop()

llm = ChatPerplexity(model="sonar", temperature=0.7)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"truncate": True}  
)

# -----------------------------
# Process URLs
# -----------------------------
if process_url_clicked:
    print("processing started")
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=200,
        chunk_overlap=30
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(1)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

# -----------------------------
# Question Answering
# -----------------------------
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
    else:
        st.warning(" No FAISS index found. Please process URLs first.")
