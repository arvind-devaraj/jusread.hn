import json
import ollama
import requests

from hn_fetch import get_past_month_hn_posts

CLASSIFY_MODEL = "gemma3:4b"

CATEGORIES = [
    "Programming Languages, Runtimes & Compilers",
    "Databases & Storage Engines",
    "Systems Architecture & Distributed Systems",
    "Web, Browser & Frontend Engineering",
    "DevOps, SRE & Cloud Infrastructure",
    "Core AI & Machine Learning",
    "Data Engineering & Pipelines",
    "Cybersecurity & Vulnerability Research",
    "Applied Cryptography & Privacy Tech",
    "Networking & Telecom Protocols",
    "Computer Architecture & Semiconductors",
    "Embedded Systems & Physical Hardware",
    "Graphics, Audio & Signal Processing",
    "Open-Source Ecosystems & Tooling",
    "Operating Systems & Kernel Development",
    "Command Line Tools & Terminal Utilities",
    "Mobile Application Development",
    "Developer Productivity & Editor Workflows",
    "Retrocomputing & Emulation",
    "Cloud Native & Serverless Computing",
    "Testing, QA & CI/CD Automation",
    "API Design & Documentation Standards",
    "Hardware Hacking & DIY Electronics",
    "Virtualization, Hypervisors & MicroVMs",
    "High-Performance Computing & Parallel Programming",
]

CLASSIFY_PROMPT = f"""You are categorizing a Hacker News post title into a topic.

Categories (choose only from this list):
{json.dumps(CATEGORIES, indent=2)}

Title: {{title}}

Pick the single category that best matches the title.
Respond with ONLY the category name copied exactly from the list above. No other text."""


def classify_title(title):
    response = ollama.generate(
        model=CLASSIFY_MODEL,
        prompt=CLASSIFY_PROMPT.format(title=title),
        options={"temperature": 0.0},
    )
    pick = response["response"].strip().strip('"')
    return pick if pick in CATEGORIES else None


def main():
    with requests.Session() as session:
        all_posts = get_past_month_hn_posts(session)
    print(f"Retrieved {len(all_posts)} posts from the last month.")

    popular_posts = [post for post in all_posts if post.get('points', 0) > 5]

    for post in popular_posts:
        category = classify_title(post.get('title', ''))
        print(f"- [{post.get('points')}pts] {post.get('title')} -> {category}")

    print(f"\nTotal posts classified: {len(popular_posts)}")


if __name__ == "__main__":
    main()
