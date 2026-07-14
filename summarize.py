import html
import os
import re
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"
SUMMARIZE_MODEL = "gpt-4o-mini"
OUTPUT_DIR = "summary-hn"

MAX_COMMENTS = 30
MAX_COMMENT_CHARS = 400

TAG_RE = re.compile(r"<[^>]+>")


def extract_item_id(link_or_id):
    match = re.search(r"id=(\d+)", link_or_id)
    if match:
        return match.group(1)
    if link_or_id.isdigit():
        return link_or_id
    raise ValueError(f"Could not extract an HN item id from {link_or_id!r}")


def clean_text(raw_html):
    if not raw_html:
        return ""
    text = raw_html.replace("<p>", "\n")
    text = TAG_RE.sub("", text)
    return html.unescape(text).strip()


def fetch_item(item_id):
    response = requests.get(ITEM_URL.format(item_id=item_id))
    response.raise_for_status()
    return response.json()


def flatten_comments(item, out):
    for child in item.get("children") or []:
        if child.get("type") == "comment" and child.get("text"):
            out.append({
                "author": child.get("author") or "unknown",
                "text": clean_text(child["text"]),
            })
        flatten_comments(child, out)


def build_summary_prompt(item, comments):
    title = item.get("title") or "(no title)"
    story_text = clean_text(item.get("text") or "")
    story_url = item.get("url") or ""

    comment_block = "\n\n".join(
        f"[{c['author']}]: {c['text'][:MAX_COMMENT_CHARS]}"
        for c in comments[:MAX_COMMENTS]
    )

    return f"""Summarize this Hacker News discussion for someone who hasn't read it.

Title: {title}
URL: {story_url}
{f"Post text: {story_text}" if story_text else ""}

Top comments ({min(len(comments), MAX_COMMENTS)} of {len(comments)}):
{comment_block}

Write a concise summary covering: (1) what the post is about, (2) the main
points of agreement or debate in the comments, (3) any notable opinions or
insights. Keep it to a few short paragraphs."""


def summarize(link_or_id):
    item_id = extract_item_id(link_or_id)
    item = fetch_item(item_id)

    comments = []
    flatten_comments(item, comments)

    prompt = build_summary_prompt(item, comments)
    client = OpenAI()
    response = client.chat.completions.create(
        model=SUMMARIZE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return item, comments, response.choices[0].message.content.strip()


def save_summary(link_or_id):
    item, comments, summary = summarize(link_or_id)

    header = f"{item.get('title')}  ({item.get('points')} pts, {len(comments)} comments)"
    hn_link = f"https://news.ycombinator.com/item?id={item.get('id')}"
    output = f"{header}\n{hn_link}\n\n{summary}\n"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{item.get('id')}.txt")
    with open(out_path, "w") as f:
        f.write(output)
    return item, out_path, output


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize.py <hn_url_or_item_id>")
        sys.exit(1)

    _, out_path, output = save_summary(sys.argv[1])
    print(output)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
