import json
import os
import re

from summarize import OUTPUT_DIR, save_summary

DATA_FILE = "ai_index.html"


def load_posts():
    with open(DATA_FILE) as f:
        html = f.read()
    match = re.search(r"const DATA = (\[.*?\]);", html, re.S)
    return json.loads(match.group(1))


def main():
    posts = load_posts()
    missing = [
        p for p in posts
        if not p.get("summary") and p.get("num_comments", 0) >= 5
    ]
    print(f"{len(missing)} posts missing summaries.")

    for post in missing:
        item_id = post["hn_link"].rsplit("id=", 1)[-1]
        out_path = os.path.join(OUTPUT_DIR, f"{item_id}.txt")
        if os.path.exists(out_path):
            continue
        try:
            save_summary(item_id)
            print(f"Saved {item_id}: {post['title'][:60]}")
        except Exception as e:
            print(f"FAILED {item_id}: {e}")


if __name__ == "__main__":
    main()
