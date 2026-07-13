import json
import requests

from hn_fetch import get_past_month_hn_posts
from custom_feed import get_tech_probabilities
from classify import classify_title

TEMPLATE_FILE = "site_template.html"
OUTPUT_FILE = "index.html"


def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last month.")

    popular_posts = [post for post in all_posts if post.get('points', 0) > 5]

    with requests.Session() as session:
        titles = [post.get('title', '') for post in popular_posts]
        probabilities = get_tech_probabilities(session, titles)

    tech_posts = [
        (post, probability) for post, probability in zip(popular_posts, probabilities)
        if probability >= 0.6
    ]

    data = []
    for post, probability in tech_posts:
        category = classify_title(post.get("title", ""))
        data.append({
            "title": post.get("title") or "",
            "points": post.get("points", 0),
            "author": post.get("author") or "",
            "num_comments": post.get("num_comments", 0),
            "probability": round(probability, 3),
            "category": category,
            "url": post.get("url"),
            "hn_link": f"https://news.ycombinator.com/item?id={post.get('objectID')}",
            "created_at": post.get("created_at"),
        })

    total_points = sum(d["points"] for d in data)
    avg_prob = sum(d["probability"] for d in data) / len(data) if data else 0

    with open(TEMPLATE_FILE, "r") as f:
        template = f.read()

    out = template
    out = out.replace('const DATA = __DATA__;', f'const DATA = {json.dumps(data, ensure_ascii=False)};')
    out = out.replace('__STAT_COUNT__', str(len(data)))
    out = out.replace('__STAT_POINTS__', f"{total_points:,}")
    out = out.replace('__STAT_AVG_TECH__', f"{round(avg_prob * 100)}%")

    with open(OUTPUT_FILE, "w") as f:
        f.write(out)

    print(f"Wrote {OUTPUT_FILE} with {len(data)} posts, {total_points} combined points, {round(avg_prob*100)}% avg tech confidence.")


if __name__ == "__main__":
    main()
