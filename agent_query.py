"""
Minimal agent demo, contrasted with rag_query.py's fixed pipeline.

rag_query.py always does the same sequence: embed -> retrieve top-k -> one
generate call. This script instead gives the model a small toolset and lets
it decide, step by step, what to do: how many searches to run, whether a
cached summary is detailed enough or it needs to pull the full live thread,
and when it has enough information to answer. The loop, not the answer, is
the point — each step is printed so you can watch the model plan.

Usage:
    python3 agent_query.py "your question"
"""
import json
import sys

from dotenv import load_dotenv
from openai import OpenAI

from rag_query import load_corpus, get_doc_embeddings, retrieve
from summarize import clean_text, fetch_item, flatten_comments

load_dotenv()

AGENT_MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 6
MAX_COMMENTS = 40
MAX_COMMENT_CHARS = 500

SYSTEM_PROMPT = """You are a research agent answering questions about Hacker \
News discussions. You have two tools:

- search: semantic search over cached discussion summaries. Use this first \
to find relevant threads.
- get_full_thread: fetches the full live story text and comments for one \
HN item. Use this when a summary doesn't have enough detail to answer \
confidently, or when you need direct quotes/opinions rather than a \
paraphrase.

Call tools as many times as needed (searching again with a refined query, \
or drilling into more than one thread), but stop once you can answer \
confidently. Give your final answer in plain text, citing threads by title \
and HN link. If nothing relevant turns up, say so instead of guessing."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Semantically search cached HN discussion summaries for threads relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Number of results to return, default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_thread",
            "description": "Fetch the full live story text and comments for a specific HN item, beyond what's in the cached summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                },
                "required": ["item_id"],
            },
        },
    },
]


def _do_search(docs, embeddings, client, args):
    query = args["query"]
    k = args.get("k", 5)
    retrieved = retrieve(client, query, docs, embeddings, k=k)
    return json.dumps([
        {
            "item_id": doc["item_id"],
            "title": doc["title"],
            "hn_link": doc["hn_link"],
            "similarity": round(score, 3),
            "summary": doc["summary"],
        }
        for doc, score in retrieved
    ])


def _do_get_full_thread(args):
    item = fetch_item(args["item_id"])
    comments = []
    flatten_comments(item, comments)
    comment_block = "\n\n".join(
        f"[{c['author']}]: {c['text'][:MAX_COMMENT_CHARS]}"
        for c in comments[:MAX_COMMENTS]
    )
    return json.dumps({
        "title": item.get("title"),
        "points": item.get("points"),
        "num_comments": len(comments),
        "story_text": clean_text(item.get("text") or ""),
        "comments": comment_block,
    })


def execute_tool(tool_call, docs, embeddings, client):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    if name == "search":
        return _do_search(docs, embeddings, client, args)
    if name == "get_full_thread":
        return _do_get_full_thread(args)
    return json.dumps({"error": f"unknown tool {name}"})


def run_agent(question, max_iterations=MAX_ITERATIONS):
    client = OpenAI()
    docs = load_corpus()
    embeddings = get_doc_embeddings(client, docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_unset=True))

        if not message.tool_calls:
            return message.content.strip()

        for tool_call in message.tool_calls:
            args = tool_call.function.arguments
            print(f"  step {step}: {tool_call.function.name}({args})")
            result = execute_tool(tool_call, docs, embeddings, client)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Hit the step limit without reaching a confident answer."


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 agent_query.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")
    answer = run_agent(question)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
