# %% Load presisted chromaDB

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DOCS_DIR = "cancer_docs"
PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "Cancer"
emb = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma(
    embedding_function=emb,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
)




questions = [
    "What are the main screening methods for lung cancer?",
    "How is mammography used in early cancer detection?",
    "What role does AI or machine learning play in cancer imaging diagnosis?",
    "Describe common biomarkers used for cancer diagnosis.",
    "What are the WHO guidelines for early diagnosis of cervical cancer?",
    "Compare different methods for detecting lung cancer early.",
    "Explain liquid biopsy in cancer diagnosis",
]

for q in questions:
    print("\n")
    print("Q:", q)
    results = vectordb.similarity_search(q, k=2)
    for i, doc in enumerate(results, 1):
        print(doc.page_content[:500])

# %%
"""