import json
import os
import requests

from hn_fetch import LOOKBACK_DAYS, get_past_month_hn_posts
from new_feed import get_ai_probabilities, AI_THRESHOLD

TEMPLATE_FILE = "ai_site_template.html"
OUTPUT_FILE = "ai_index.html"
SUMMARY_DIR = "summary-hn"


def load_summary(item_id):
    path = os.path.join(SUMMARY_DIR, f"{item_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        contents = f.read()
    # saved files are "<header>\n<hn_link>\n\n<summary body>"
    _, _, body = contents.partition("\n\n")
    return body.strip() or None


def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last {LOOKBACK_DAYS} days.")

    popular_posts = [post for post in all_posts if post.get('points', 0) >= 10]

    titles = [post.get('title', '') for post in popular_posts]
    probabilities = get_ai_probabilities(titles)

    ai_posts = [
        (post, probability) for post, probability in zip(popular_posts, probabilities)
        if probability >= AI_THRESHOLD
    ]

    data = []
    for post, probability in ai_posts:
        item_id = post.get("objectID")
        data.append({
            "title": post.get("title") or "",
            "points": post.get("points", 0),
            "author": post.get("author") or "",
            "num_comments": post.get("num_comments", 0),
            "probability": round(probability, 3),
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
