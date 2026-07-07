# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-script tool (`custom-feed.py`) that builds a "tech-only" Hacker News feed. It pulls the last 7 days of HN stories via the Algolia HN Search API, filters to posts with >5 points, then uses local embeddings to classify each title as tech-related or not, printing the tech-classified posts sorted by the order they were fetched.

## Running

```bash
python3 custom-feed.py
```

Requirements:
- `requests` must be installed (`pip install requests`) — there is no requirements.txt/pyproject.toml in this repo yet.
- [Ollama](https://ollama.com) must be running locally on `http://localhost:11434` with the `embeddinggemma` model pulled (`ollama pull embeddinggemma`), since classification calls Ollama's `/api/embed` endpoint.

There is no build step, lint config, or test suite in this repo currently.

## Architecture

The script is a linear pipeline with two independent phases:

1. **Fetch (`get_past_week_hn_posts`)** — queries the Algolia HN Search API (`search_by_date`, tag `story`) for everything created in the last 7 days. Page 0 is fetched first to learn `nbPages`, then remaining pages are fetched concurrently via a `ThreadPoolExecutor` (max 8 workers) since pages are independent.

2. **Classify (`get_tech_probabilities`)** — instead of a trained classifier, this uses zero-shot embedding similarity:
   - `TECH_SEED_TEXTS` / `NON_TECH_SEED_TEXTS` are hand-written example headlines representing each class.
   - Each seed set is embedded (via Ollama `embeddinggemma`) and averaged into a single "prototype" vector per class.
   - Each candidate title is embedded, and its cosine similarity to both prototypes is computed.
   - `SIMILARITY_TEMPERATURE` scales the similarity gap before a sigmoid turns it into a pseudo-probability of "tech-ness". A threshold of `0.6` (set at the bottom of the script) decides inclusion in the final feed.

Tuning the feed's behavior (broader/narrower topic definition, stricter/looser inclusion) is done by editing the seed text lists, `SIMILARITY_TEMPERATURE`, or the `0.6` threshold — not by adding ML training code.

The script currently has no `__main__` guard: fetching, classification, and printing all run at module import time (lines 101-120).
