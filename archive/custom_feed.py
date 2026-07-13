import math
import requests

from hn_fetch import get_past_month_hn_posts

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "embeddinggemma"
EMBED_PREFIX = "task: classification | query: "
SIMILARITY_TEMPERATURE = 10.0  # sharpens the softmax over raw cosine similarities

TECH_SEED_TEXTS = [
    "Show HN: I built a new open source database engine in Rust",
    "A new programming language for systems development",
    "Nvidia announces next-generation GPU architecture for AI training",
    "How we scaled our Kubernetes cluster to 10,000 nodes",
    "Understanding how compilers optimize WebAssembly bytecode",
    "OpenAI releases a new API for function calling",
]
NON_TECH_SEED_TEXTS = [
    "Scientists discover new species of deep-sea jellyfish",
    "The history of the Roman Empire's economic collapse",
    "President signs new bill on agricultural subsidies",
    "Study finds link between diet and heart disease",
    "Local team wins championship after dramatic final match",
    "A memoir about growing up in the countryside",
]

def embed_texts(session, texts):
    texts = [EMBED_PREFIX + t for t in texts]
    response = session.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
    response.raise_for_status()
    return response.json()["embeddings"]

def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

def _average_vector(vectors):
    length = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(length)]

def get_tech_probabilities(session, titles):
    tech_proto = _average_vector(embed_texts(session, TECH_SEED_TEXTS))
    non_tech_proto = _average_vector(embed_texts(session, NON_TECH_SEED_TEXTS))

    title_vectors = embed_texts(session, titles)

    probabilities = []
    for vector in title_vectors:
        sim_tech = _cosine_similarity(vector, tech_proto) * SIMILARITY_TEMPERATURE
        sim_non_tech = _cosine_similarity(vector, non_tech_proto) * SIMILARITY_TEMPERATURE
        probabilities.append(1 / (1 + math.exp(sim_non_tech - sim_tech)))  # sigmoid of the sim gap
    return probabilities

def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last month.")

    # Classify tech-relatedness for posts with more than 5 upvotes
    popular_posts = [post for post in all_posts if post.get('points', 0) > 5]

    with requests.Session() as session:
        titles = [post.get('title', '') for post in popular_posts]
        probabilities = get_tech_probabilities(session, titles)

    tech_posts = [
        (post, probability) for post, probability in zip(popular_posts, probabilities)
        if probability >= 0.6
    ]

    for post, probability in tech_posts:
        print(f"- [{post.get('points')}pts] [{probability:.2f} tech] {post.get('title')} by {post.get('author')}")

    print(f"\nTotal posts: {len(tech_posts)}")


if __name__ == "__main__":
    main()