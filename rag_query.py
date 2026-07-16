"""
Minimal RAG (retrieval-augmented generation) demo over this repo's HN summaries.

Retrieval: embed every saved discussion summary in summary-hn/, embed the
user's question, and rank summaries by cosine similarity.

Generation: hand the top-k summaries to an LLM as context and have it answer
the question, citing which post(s) it drew on — instead of answering from
the model's own training data alone.

Usage:
    python3 rag_query.py "What are people saying about AI coding agents?"
"""
import json
import math
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUMMARY_DIR = "summary-hn"
EMBED_CACHE_FILE = "rag-embeddings-cache.json"
EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
TOP_K = 5

HEADER_RE = re.compile(r"^(.*?)\s+\((\d+) pts, (\d+) comments\)$")


def load_corpus():
    docs = []
    for filename in sorted(os.listdir(SUMMARY_DIR)):
        if not filename.endswith(".txt"):
            continue
        item_id = filename[:-4]
        with open(os.path.join(SUMMARY_DIR, filename)) as f:
            header, hn_link, _, summary = f.read().split("\n", 3)
        match = HEADER_RE.match(header)
        title = match.group(1) if match else header
        docs.append({
            "item_id": item_id,
            "title": title,
            "hn_link": hn_link,
            "summary": summary.strip(),
        })
    return docs


def load_embed_cache():
    if not os.path.exists(EMBED_CACHE_FILE):
        return {}
    with open(EMBED_CACHE_FILE) as f:
        return json.load(f)


def save_embed_cache(cache):
    with open(EMBED_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def embed_texts(client, texts):
    vectors = []
    batch_size = 200  # keep batches well under OpenAI's 300k-tokens-per-request cap
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def get_doc_embeddings(client, docs):
    # Embeddings only depend on title+summary text, which is immutable once
    # a summary file is written, so cache by item_id and skip re-embedding
    # docs seen in a prior run.
    cache = load_embed_cache()
    uncached = [d for d in docs if d["item_id"] not in cache]
    if uncached:
        vectors = embed_texts(client, [f"{d['title']}\n{d['summary']}" for d in uncached])
        for doc, vector in zip(uncached, vectors):
            cache[doc["item_id"]] = vector
        save_embed_cache(cache)
    return {doc["item_id"]: cache[doc["item_id"]] for doc in docs}


def retrieve(client, query, docs, embeddings, k=TOP_K):
    query_vector = embed_texts(client, [query])[0]
    scored = [
        (doc, cosine_similarity(query_vector, embeddings[doc["item_id"]]))
        for doc in docs
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def build_prompt(query, retrieved):
    context_block = "\n\n".join(
        f"[{i}] {doc['title']}\n{doc['summary']}"
        for i, (doc, _score) in enumerate(retrieved, start=1)
    )
    return f"""Answer the question using ONLY the numbered Hacker News discussion
summaries below as context. Cite sources inline like [1], [2]. If the context
doesn't contain enough information to answer, say so instead of guessing.

Context:
{context_block}

Question: {query}"""


def generate_answer(client, query, retrieved):
    prompt = build_prompt(query, retrieved)
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def answer_question(query, k=TOP_K):
    client = OpenAI()
    docs = load_corpus()
    embeddings = get_doc_embeddings(client, docs)
    retrieved = retrieve(client, query, docs, embeddings, k=k)
    answer = generate_answer(client, query, retrieved)
    return retrieved, answer


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 rag_query.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    retrieved, answer = answer_question(query)

    print("Retrieved sources:")
    for i, (doc, score) in enumerate(retrieved, start=1):
        print(f"  [{i}] ({score:.2f}) {doc['title']}  {doc['hn_link']}")

    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
