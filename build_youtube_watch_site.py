import json

from build_youtube_site import (
    CACHE_FILE,
    DESCRIPTION_CHARS,
    EMBEDDABLE_CACHE_FILE,
    load_json_cache,
    save_json_cache,
)
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

TEMPLATE_FILE = "youtube_watch_template.html"
OUTPUT_FILE = "youtube_watch.html"
DESC_DISPLAY_CHARS = 600  # separate from DESCRIPTION_CHARS, which is just the classifier's input


def main():
    videos = get_all_playlist_videos(PLAYLIST_IDS, API_KEY)
    print(f"Retrieved {len(videos)} videos from {len(PLAYLIST_IDS)} playlist(s).")

    topics = load_topics(TOPICS_FILE)
    topic_seeds = load_topic_seeds(TOPIC_SEEDS_FILE)

    # Reuses the same caches build_youtube_site.py populates, keyed by video
    # id, so running this script doesn't re-classify or re-check videos
    # already processed for the grid feed.
    cache = load_json_cache(CACHE_FILE)
    uncached_videos = [v for v in videos if v["video_id"] not in cache]
    if uncached_videos:
        texts = [
            f"{v['title']} {v['description'][:DESCRIPTION_CHARS]}".strip()
            for v in uncached_videos
        ]
        classifications = classify_titles(texts, topics, topic_seeds)
        for video, topic in zip(uncached_videos, classifications):
            cache[video["video_id"]] = {"title": video["title"], "topic": topic}
        save_json_cache(CACHE_FILE, cache)
    print(f"Classified {len(uncached_videos)} new videos ({len(videos) - len(uncached_videos)} from cache).")

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
            "description": v["description"][:DESC_DISPLAY_CHARS],
        })

    with open(TEMPLATE_FILE, "r") as f:
        template = f.read()

    out = template.replace("const DATA = __DATA__;", f"const DATA = {json.dumps(data, ensure_ascii=False)};")

    with open(OUTPUT_FILE, "w") as f:
        f.write(out)

    print(f"Wrote {OUTPUT_FILE} with {len(data)} videos.")


if __name__ == "__main__":
    main()
