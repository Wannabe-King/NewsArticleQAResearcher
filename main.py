import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# LLM imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_perplexity import ChatPerplexity

# Load environment variables
load_dotenv()  # expects OPENAI_API_KEY or PPLX_API_KEY

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# -----------------------------
# Choose backend: OpenAI or Perplexity
# -----------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() 
if LLM_PROVIDER == "perplexity":
    llm = ChatPerplexity(model="sonar", temperature=0.7)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  
# -----------------------------
# Process URLs
# -----------------------------
if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(1)

    # Save index
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

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
