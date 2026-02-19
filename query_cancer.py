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

# %% feed best doc to LLM to generate final answer
for q in questions:
    print("\n" + "=" * 90)
    print("Q:", q)
  
    candidates = vectordb.similarity_search(q, k=BEST_K)

    best_doc, best_score = rerank_top1(q, candidates)

    prompt = f"""
 Answer question using only the context below.
    If the context does not contain the answer, say "I don't know based on the provided documents."

Question:
{q}

Context:
{best_doc.page_content}
""".strip()

    # (D) Ask the LLM to generate the final answer based on that context
    answer = llm.invoke(prompt).content

    # (E) Print result
    print(f"\nBest reranker score: {best_score:.4f}")
    print("\nAnswer:\n", answer)