import math
import requests
from dotenv import load_dotenv
from openai import OpenAI

from hn_fetch import get_past_month_hn_posts

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 2000  # OpenAI caps embeddings requests at 2048 inputs
SIMILARITY_TEMPERATURE = 10.0  # sharpens the softmax over raw cosine similarities
AI_THRESHOLD = 0.82  # recalibrated for text-embedding-3-small; 0.70 (tuned for the
                      # old local embeddinggemma model) let too much noise through
                      # after switching embedding providers

AI_SEED_TEXTS = [
    "OpenAI releases a new large language model with improved reasoning",
    "Show HN: I built an autonomous AI agent framework for coding tasks",
    "Understanding how transformer attention mechanisms work in LLMs",
    "Anthropic announces a new Claude model with an extended context window",
    "Fine-tuning open source LLMs for domain-specific tasks",
    "A deep dive into retrieval-augmented generation architectures",
]
NON_AI_SEED_TEXTS = [
    "How we scaled our Kubernetes cluster to 10,000 nodes",
    "Understanding how compilers optimize WebAssembly bytecode",
    "A new programming language for systems development",
    "Scientists discover new species of deep-sea jellyfish",
    "The history of the Roman Empire's economic collapse",
    "Local team wins championship after dramatic final match",
]


def embed_texts(texts):
    client = OpenAI()
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _average_vector(vectors):
    length = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(length)]


def get_ai_probabilities(titles):
    ai_proto = _average_vector(embed_texts(AI_SEED_TEXTS))
    non_ai_proto = _average_vector(embed_texts(NON_AI_SEED_TEXTS))

    title_vectors = embed_texts(titles)

    probabilities = []
    for vector in title_vectors:
        sim_ai = _cosine_similarity(vector, ai_proto) * SIMILARITY_TEMPERATURE
        sim_non_ai = _cosine_similarity(vector, non_ai_proto) * SIMILARITY_TEMPERATURE
        probabilities.append(1 / (1 + math.exp(sim_non_ai - sim_ai)))  # sigmoid of the sim gap
    return probabilities


def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last month.")

    popular_posts = [post for post in all_posts if post.get('points', 0) >= 10]

    titles = [post.get('title', '') for post in popular_posts]
    probabilities = get_ai_probabilities(titles)

    ai_posts = [
        (post, probability) for post, probability in zip(popular_posts, probabilities)
        if probability >= AI_THRESHOLD
    ]

    for post, probability in ai_posts:
        print(f"- [{post.get('points')}pts] [{probability:.2f} ai] {post.get('title')} by {post.get('author')}")

    print(f"\nTotal posts: {len(ai_posts)}")


if __name__ == "__main__":
    main()
