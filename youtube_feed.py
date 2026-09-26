import json
import math
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ["YOUTUBE_API_KEY"]
PLAYLIST_FILE = "youtube-playlist.txt"
TOPICS_FILE = "youtube-topics.txt"
TOPIC_SEEDS_FILE = "youtube-topic-seeds.json"
EMBED_MODEL = "text-embedding-3-small"

with open(PLAYLIST_FILE) as f:
    PLAYLIST_IDS = [line.strip() for line in f if line.strip()]

def get_playlist_videos(playlist_id, api_key):
    videos = []
    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
        "part": "snippet,contentDetails",
        "playlistId": playlist_id,
        "maxResults": 50,
        "key": api_key,
    }

    while True:
        resp = requests.get(url, params=params).json()
        for item in resp.get("items", []):
            snippet = item["snippet"]
            videos.append({
                "title": snippet["title"],
                "video_id": item["contentDetails"]["videoId"],
                "channel": snippet.get("videoOwnerChannelTitle", snippet["channelTitle"]),
                "published_at": snippet["publishedAt"],
                "description": snippet["description"],
            })

        next_token = resp.get("nextPageToken")
        if not next_token:
            break
        params["pageToken"] = next_token

    return videos


def get_all_playlist_videos(playlist_ids, api_key):
    videos = {}
    for playlist_id in playlist_ids:
        for video in get_playlist_videos(playlist_id, api_key):
            videos[video["video_id"]] = video  # de-dupe videos shared across playlists
    return list(videos.values())


def get_embeddable_status(video_ids, api_key):
    """Look up whether each video allows embedding via the Data API's
    videos.list status field, since that can't be reliably detected
    client-side (the IFrame API's error events depend on postMessage,
    which misbehaves when the page is opened over file://)."""
    embeddable = {}
    url = "https://www.googleapis.com/youtube/v3/videos"
    for i in range(0, len(video_ids), 50):  # videos.list caps at 50 ids per request
        batch = video_ids[i:i + 50]
        params = {"part": "status", "id": ",".join(batch), "key": api_key}
        resp = requests.get(url, params=params).json()
        for item in resp.get("items", []):
            embeddable[item["id"]] = item["status"]["embeddable"]
    # Videos missing from the response (deleted/private) can't be embedded.
    return {vid: embeddable.get(vid, False) for vid in video_ids}


def load_topics(path):
    with open(path) as f:
        return json.load(f)


def load_topic_seeds(path):
    with open(path) as f:
        return json.load(f)


def embed_texts(texts):
    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _average_vector(vectors):
    length = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(length)]


def classify_titles(texts, topics, topic_seeds):
    """Classifies each text by nearest topic prototype. A topic's prototype
    is the average embedding of its hand-written example titles (falling
    back to the bare topic name if it has no seeds yet) rather than the
    topic name's own embedding, since a single short label is too weak/noisy
    a target — ambiguous titles would land on whichever label happened to be
    nearest for reasons unrelated to actual topical meaning."""
    seed_lists = [topic_seeds.get(topic, [topic]) for topic in topics]
    flat_seeds = [seed for seeds in seed_lists for seed in seeds]
    seed_vectors = embed_texts(flat_seeds)

    prototypes = []
    i = 0
    for seeds in seed_lists:
        prototypes.append(_average_vector(seed_vectors[i:i + len(seeds)]))
        i += len(seeds)
    topic_vectors = list(zip(topics, prototypes))

    text_vectors = embed_texts(texts)

    classifications = []
    for vector in text_vectors:
        best_topic = max(
            topic_vectors,
            key=lambda tv: _cosine_similarity(vector, tv[1]),
        )[0]
        classifications.append(best_topic)
    return classifications


if __name__ == "__main__":
    videos = get_all_playlist_videos(PLAYLIST_IDS, API_KEY)
    topics = load_topics(TOPICS_FILE)
    topic_seeds = load_topic_seeds(TOPIC_SEEDS_FILE)
    classifications = classify_titles([v["title"] for v in videos], topics, topic_seeds)

    for v, topic in zip(videos, classifications):
        print(f"[{topic}] {v['title']} — {v['channel']} ({v['video_id']})")