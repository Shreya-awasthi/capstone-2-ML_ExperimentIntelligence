from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
import json, re, os

experiments = json.load(open("data/experiments.json"))

def tokenise(text): return re.findall(r'\w+', text.lower())

# Embed: algorithm + task + outcome + tags + notes (semantic signal)
def make_doc(e):
    return (f"{e['algorithm']} {e['task']} outcome={e['outcome']} "
            f"leakage={e['leakage_risk']} tags={' '.join(e['tags'])} notes={e['notes']}")

bm25_corpus = [tokenise(make_doc(e)) for e in experiments]
bm25 = BM25Okapi(bm25_corpus)

# ── Persistent ChromaDB — embeddings survive restarts ─────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection("experiments", embedding_function=ef)

# Only embed if the collection is empty — skip on subsequent startups
existing_ids = set(collection.get()["ids"])
new_experiments = [e for e in experiments if e["id"] not in existing_ids]

if new_experiments:
    collection.add(
        documents=[make_doc(e) for e in new_experiments],
        metadatas=[{"id":e["id"],"outcome":e["outcome"],"leakage_risk":e["leakage_risk"],
                    "algorithm":e["algorithm"],"task":e["task"]} for e in new_experiments],
        ids=[e["id"] for e in new_experiments]
    )
    print(f"✓ Embedded {len(new_experiments)} new experiments → {CHROMA_PATH}")
else:
    print(f"✓ ChromaDB loaded from cache ({len(existing_ids)} experiments) — skipping embedding")

print(f"✓ BM25 index built — {len(experiments)} experiments")
