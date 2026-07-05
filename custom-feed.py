import math
import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 1000  # max allowed, minimizes number of requests

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

def _fetch_page(session, one_week_ago, page):
    params = {
        'tags': 'story',
        'numericFilters': f'created_at_i>{one_week_ago}',
        'page': page,
        'hitsPerPage': HITS_PER_PAGE
    }
    response = session.get(URL, params=params)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    return response.json()

def get_past_week_hn_posts():
    # Calculate Unix timestamp for exactly 7 days ago
    seconds_in_a_week = 7 * 24 * 60 * 60
    one_week_ago = int(time.time()) - seconds_in_a_week

    with requests.Session() as session:
        first = _fetch_page(session, one_week_ago, 0)
        if not first:
            return []

        posts = first.get('hits', [])
        nb_pages = first.get('nbPages', 1)

        if nb_pages <= 1:
            return posts

        # Remaining pages don't depend on each other, so fetch them concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(
                lambda p: _fetch_page(session, one_week_ago, p),
                range(1, nb_pages)
            )
            for data in results:
                if data:
                    posts.extend(data.get('hits', []))

    return posts

# Execute and view results
all_posts = get_past_week_hn_posts()
print(f"Retrieved {len(all_posts)} posts from the last week.")

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