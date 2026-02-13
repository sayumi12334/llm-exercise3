# %%
#Build chromaDB to store Cancer PDFs

"""
Docstring for llm-exercise3-dev.llm-exercise3.build_cancer_db
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings



DOCS_DIR = "cancer_docs"
PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "Cancer"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"


# %%  Load ALL PDFs in the folder

loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()


# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
chunks = splitter.split_documents(documents)


emb = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
)
vectordb.persist()
print("✅ Saved to Chroma collection:", COLLECTION_NAME)


        



