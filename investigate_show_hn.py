"""
Agent use case #1: investigative ranking.

"Which Show HN post got genuinely positive *technical* feedback, not just
upvotes?" isn't answerable by sorting on a metric — points and comment count
don't distinguish substantive engagement from a crowd of "cool!" replies.
This agent lists Show HN candidates, then has to actually read full comment
threads on the promising ones and judge, the way a human would skim a few
threads before recommending one.

Notably this agent doesn't use embedding search at all — list_show_hn is a
plain deterministic filter over the cached summaries, get_full_thread is a
live fetch. Agents don't require RAG under the hood; they just need tools
scoped to the task.

Usage:
    python3 investigate_show_hn.py
    python3 investigate_show_hn.py "your question about Show HN posts"
"""
import ast
import json
import operator
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

from summarize import clean_text, fetch_item, flatten_comments

load_dotenv()

SUMMARY_DIR = "summary-hn"
AGENT_MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 8
MAX_COMMENTS = 40
MAX_COMMENT_CHARS = 500

HEADER_RE = re.compile(r"^(.*?)\s+\((\d+) pts, (\d+) comments\)$")

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sum": sum, "min": min, "max": max, "len": len,
    "sorted": sorted, "abs": abs, "round": round,
}


def _safe_eval(node, names):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name: {node.id!r} (available: {sorted(names)})")
    if isinstance(node, ast.List):
        return [_safe_eval(el, names) for el in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left, names), _safe_eval(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_eval(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_safe_eval(a, names) for a in node.args]
        kwargs = {kw.arg: _safe_eval(kw.value, names) for kw in node.keywords}
        return _FUNCS[node.func.id](*args, **kwargs)
    if isinstance(node, ast.Subscript):
        value = _safe_eval(node.value, names)
        sl = node.slice
        if isinstance(sl, ast.Slice):
            lower = _safe_eval(sl.lower, names) if sl.lower else None
            upper = _safe_eval(sl.upper, names) if sl.upper else None
            step = _safe_eval(sl.step, names) if sl.step else None
            return value[lower:upper:step]
        return value[_safe_eval(sl, names)]
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def calculate(expression, names=None):
    return _safe_eval(ast.parse(expression, mode="eval"), names or {})

DEFAULT_QUESTION = (
    "Which Show HN post got genuinely positive, substantive technical "
    "feedback in the comments -- not just upvotes or generic praise? "
    "Pick one winner and justify it with specific comments."
)

SYSTEM_PROMPT = """You are an investigative research agent judging Hacker \
News "Show HN" launches. You have three tools:

- list_show_hn: lists cached Show HN threads (title, points, comment count, \
a short summary), most recent first, default 20. Use this to browse \
candidates for qualitative judging, not to gather numbers for statistics -- \
see calculate below for that. Points and comment count are weak, gameable \
signals -- a post can rack up points from a flashy demo while the comments \
are lukewarm or critical, or the reverse.
- get_full_thread: fetches the full live story text and every comment for \
one item. This is the only way to actually judge the quality of feedback.
- calculate: evaluates a numeric Python expression exactly (arithmetic, \
sum/min/max/len/sorted/abs/round, list literals and slicing). It has two \
variables already bound to the full cached dataset: `points` and \
`comments` (one entry per cached Show HN post, same order). ALWAYS use \
calculate for any aggregate statistic -- averages, percentages, sums, \
comparisons, deviations -- across more than a couple of posts, and ALWAYS \
reference the `points`/`comments` variables rather than retyping numbers \
you saw in a list_show_hn result. Retyping a long list by hand is exactly \
the kind of transcription error this tool exists to avoid -- do not do it. \
Do not do the arithmetic yourself either; you will get it wrong.

For qualitative "which post is best/worst" questions: do not rank by \
points or comment count alone. Use list_show_hn to gather candidates, then \
use get_full_thread on several promising ones to actually read what \
commenters said -- distinguish substantive technical engagement (specific \
critiques, comparisons, "I tried this and...") from generic enthusiasm or \
pile-on negativity. Justify your pick by quoting or closely paraphrasing \
specific comments, and give the HN link.

For quantitative/statistical questions: gather the numbers with \
list_show_hn, compute with calculate, and report the exact result -- show \
the calculation, don't just assert a number."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_show_hn",
            "description": "List cached Show HN threads, most recent first, with points/comment counts and a short summary. Default limit is 20; raise it only if you actually need to read more summaries. Do NOT use this to gather numbers for statistics -- the calculate tool already has the full points/comments dataset bound, no need to list everything here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max number of threads to return, default 20."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_thread",
            "description": "Fetch the full live story text and comments for a specific HN item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric Python expression exactly: arithmetic operators, sum/min/max/len/sorted/abs/round, list literals and slicing. Two variables are pre-bound to the FULL cached dataset (do not retype numbers by hand, reference these instead): `points` (list of every cached Show HN post's point count) and `comments` (list of every cached Show HN post's comment count), same order, one entry per post. E.g. 'sum(sorted(points, reverse=True)[:5]) / sum(points) * 100' for a top-5 share of total points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]


def load_show_hn_corpus():
    docs = []
    for filename in sorted(os.listdir(SUMMARY_DIR)):
        if not filename.endswith(".txt"):
            continue
        item_id = filename[:-4]
        with open(os.path.join(SUMMARY_DIR, filename)) as f:
            header, hn_link, _, summary = f.read().split("\n", 3)
        match = HEADER_RE.match(header)
        if not match:
            continue
        title, points, num_comments = match.group(1), int(match.group(2)), int(match.group(3))
        if not title.strip().lower().startswith("show hn"):
            continue
        docs.append({
            "item_id": item_id,
            "title": title,
            "points": points,
            "num_comments": num_comments,
            "hn_link": hn_link,
            "summary": summary.strip(),
        })
    docs.sort(key=lambda d: int(d["item_id"]), reverse=True)
    return docs


def _do_list_show_hn(corpus, args):
    limit = args.get("limit", 20)
    return json.dumps(corpus[:limit])


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


def execute_tool(tool_call, corpus):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    if name == "list_show_hn":
        return _do_list_show_hn(corpus, args)
    if name == "get_full_thread":
        return _do_get_full_thread(args)
    if name == "calculate":
        names = {
            "points": [d["points"] for d in corpus],
            "comments": [d["num_comments"] for d in corpus],
        }
        try:
            return json.dumps({"result": calculate(args["expression"], names)})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"unknown tool {name}"})


def run_agent(question, max_iterations=MAX_ITERATIONS):
    client = OpenAI()
    corpus = load_show_hn_corpus()

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
            print(f"  step {step}: {tool_call.function.name}({tool_call.function.arguments})")
            result = execute_tool(tool_call, corpus)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Hit the step limit without reaching a confident answer."


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"Question: {question}\n")
    answer = run_agent(question)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
