import json
import os
import requests

from hn_fetch import LOOKBACK_DAYS, get_past_month_hn_posts
from new_feed import get_ai_probabilities, AI_THRESHOLD
from summarize import save_summary

TEMPLATE_FILE = "ai_site_template.html"
OUTPUT_FILE = "ai_index.html"
SUMMARY_DIR = "summary-hn"
CACHE_FILE = "classification-cache.json"
MIN_COMMENTS_FOR_SUMMARY = 5


def load_summary(item_id):
    path = os.path.join(SUMMARY_DIR, f"{item_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        contents = f.read()
    # saved files are "<header>\n<hn_link>\n\n<summary body>"
    _, _, body = contents.partition("\n\n")
    return body.strip() or None


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE) as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last {LOOKBACK_DAYS} days.")

    popular_posts = [post for post in all_posts if post.get('points', 0) >= 10]

    # Classification only depends on the title, so results are cached
    # by HN item id and reused across runs instead of re-embedding
    # titles that were already classified in a prior run.
    cache = load_cache()
    uncached_posts = [p for p in popular_posts if p.get("objectID") not in cache]
    if uncached_posts:
        titles = [post.get('title', '') for post in uncached_posts]
        probabilities = get_ai_probabilities(titles)
        for post, probability in zip(uncached_posts, probabilities):
            cache[post["objectID"]] = {
                "title": post.get("title") or "",
                "probability": round(probability, 3),
            }
        save_cache(cache)
    print(f"Classified {len(uncached_posts)} new posts ({len(popular_posts) - len(uncached_posts)} from cache).")

    ai_posts = [
        post for post in popular_posts
        if cache[post["objectID"]]["probability"] >= AI_THRESHOLD
    ]

    missing = [
        post for post in ai_posts
        if post.get("num_comments", 0) >= MIN_COMMENTS_FOR_SUMMARY
        and load_summary(post["objectID"]) is None
    ]
    print(f"{len(missing)} posts missing summaries.")
    for post in missing:
        item_id = post["objectID"]
        try:
            save_summary(item_id)
            print(f"Saved {item_id}: {post['title'][:60]}")
        except Exception as e:
            print(f"FAILED {item_id}: {e}")

    data = []
    for post in ai_posts:
        item_id = post.get("objectID")
        data.append({
            "title": post.get("title") or "",
            "points": post.get("points", 0),
            "author": post.get("author") or "",
            "num_comments": post.get("num_comments", 0),
            "probability": cache[item_id]["probability"],
            "url": post.get("url"),
            "hn_link": f"https://news.ycombinator.com/item?id={item_id}",
            "created_at": post.get("created_at"),
            "summary": load_summary(item_id),
        })

    total_points = sum(d["points"] for d in data)
    avg_prob = sum(d["probability"] for d in data) / len(data) if data else 0

    with open(TEMPLATE_FILE, "r") as f:
        template = f.read()

    out = template
    out = out.replace('const DATA = __DATA__;', f'const DATA = {json.dumps(data, ensure_ascii=False)};')
    out = out.replace('__STAT_COUNT__', str(len(data)))
    out = out.replace('__STAT_POINTS__', f"{total_points:,}")
    out = out.replace('__STAT_AVG_AI__', f"{round(avg_prob * 100)}%")
    out = out.replace('__STAT_THRESHOLD__', f"{round(AI_THRESHOLD * 100)}%")
    out = out.replace('__STAT_WINDOW_DAYS__', str(LOOKBACK_DAYS))

    with open(OUTPUT_FILE, "w") as f:
        f.write(out)

    print(f"Wrote {OUTPUT_FILE} with {len(data)} posts, {total_points} combined points, {round(avg_prob*100)}% avg AI confidence.")


if __name__ == "__main__":
    main()
