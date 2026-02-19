# %% Load presisted chromaDB

"""
Docstring for llm-exercise3-dev.llm-exercise3.query_cancer
"""
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

EMB_MODEL = "nomic-embed-text"
PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "Cancer"

# %% retireve the first 20 chunks
BEST_K = 20
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2:3b"

emb = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma(
    embedding_function=emb,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
)

# %%
reranker = CrossEncoder(RERANKER)

llm = ChatOllama(model=LLM_MODEL)

# %% best document using reranking

def rerank_top1(query: str, docs):
 
    pairs = [(query, d.page_content) for d in docs]

    scores = reranker.predict(pairs)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    
    return docs[best_idx], float(scores[best_idx])
# %% 3) Questions

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
