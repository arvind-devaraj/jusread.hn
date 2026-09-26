import json
import os

from youtube_feed import (
    API_KEY,
    PLAYLIST_IDS,
    TOPIC_SEEDS_FILE,
    TOPICS_FILE,
    classify_titles,
    get_all_playlist_videos,
    get_embeddable_status,
    load_topic_seeds,
    load_topics,
)

TEMPLATE_FILE = "youtube_site_template.html"
OUTPUT_FILE = "youtube_index.html"
CACHE_FILE = "youtube-classification-cache.json"
EMBEDDABLE_CACHE_FILE = "youtube-embeddable-cache.json"
DESCRIPTION_CHARS = 300  # titles alone are often too vague/generic to classify well


def load_json_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main():
    videos = get_all_playlist_videos(PLAYLIST_IDS, API_KEY)
    print(f"Retrieved {len(videos)} videos from {len(PLAYLIST_IDS)} playlist(s).")

    topics = load_topics(TOPICS_FILE)
    topic_seeds = load_topic_seeds(TOPIC_SEEDS_FILE)

    # Classification depends on title + description, so results are cached
    # by video id and reused across runs instead of re-embedding videos that
    # were already classified in a prior run.
    cache = load_json_cache(CACHE_FILE)
    uncached_videos = [v for v in videos if v["video_id"] not in cache]
    if uncached_videos:
        # Titles alone are often too generic to classify well (e.g. "Distributed
        # ML Talk @ UC Berkeley" gives no hint it's about GPU parallelism), so
        # the description is included as extra context.
        texts = [
            f"{v['title']} {v['description'][:DESCRIPTION_CHARS]}".strip()
            for v in uncached_videos
        ]
        classifications = classify_titles(texts, topics, topic_seeds)
        for video, topic in zip(uncached_videos, classifications):
            cache[video["video_id"]] = {"title": video["title"], "topic": topic}
        save_json_cache(CACHE_FILE, cache)
    print(f"Classified {len(uncached_videos)} new videos ({len(videos) - len(uncached_videos)} from cache).")

    # Whether a video allows embedding can't be reliably detected client-side
    # (see youtube_feed.get_embeddable_status), so it's looked up via the
    # Data API here and cached by video id alongside classification.
    embeddable_cache = load_json_cache(EMBEDDABLE_CACHE_FILE)
    uncached_ids = [v["video_id"] for v in videos if v["video_id"] not in embeddable_cache]
    if uncached_ids:
        embeddable_cache.update(get_embeddable_status(uncached_ids, API_KEY))
        save_json_cache(EMBEDDABLE_CACHE_FILE, embeddable_cache)
    print(f"Checked embeddability for {len(uncached_ids)} new videos ({len(videos) - len(uncached_ids)} from cache).")

    data = []
    for v in videos:
        data.append({
            "title": v["title"],
            "video_id": v["video_id"],
            "channel": v["channel"],
            "published_at": v["published_at"],
            "topic": cache[v["video_id"]]["topic"],
            "embeddable": embeddable_cache[v["video_id"]],
        })

    unique_topics = {d["topic"] for d in data}
    unique_channels = {d["channel"] for d in data}

    with open(TEMPLATE_FILE, "r") as f:
        template = f.read()

    out = template
    out = out.replace("const DATA = __DATA__;", f"const DATA = {json.dumps(data, ensure_ascii=False)};")
    out = out.replace("__STAT_COUNT__", str(len(data)))
    out = out.replace("__STAT_TOPICS__", str(len(unique_topics)))
    out = out.replace("__STAT_CHANNELS__", str(len(unique_channels)))

    with open(OUTPUT_FILE, "w") as f:
        f.write(out)

    print(f"Wrote {OUTPUT_FILE} with {len(data)} videos across {len(unique_topics)} topics.")


if __name__ == "__main__":
    main()
