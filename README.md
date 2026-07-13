# jusread.hn

A tool that builds a "tech-only" Hacker News feed: it pulls the last 30 days of HN stories, filters to posts with more than 5 points, then uses local embeddings to classify each title as tech-related or not.

## Running

```bash
python3 custom-feed.py
```

Requirements:
- `requests` must be installed (`pip install requests`).
- [Ollama](https://ollama.com) must be running locally on `http://localhost:11434` with the `embeddinggemma` model pulled (`ollama pull embeddinggemma`), since classification calls Ollama's `/api/embed` endpoint.

## How it works

1. **Fetch** — queries the Algolia HN Search API (`search_by_date`, tag `story`) for everything created in the last 30 days. Since the Algolia endpoint caps total retrievable results at ~1000 per query (page 1+ comes back empty even when far more posts match), the 30-day window is split into per-day queries fetched concurrently (8 workers) and merged, rather than one query with page-based pagination.

2. **Classify** — instead of a trained classifier, this uses zero-shot embedding similarity:
   - Two hardcoded lists of example headlines (`TECH_SEED_TEXTS` / `NON_TECH_SEED_TEXTS`) represent each class.
   - Each seed set is embedded via Ollama (`embeddinggemma`) and averaged into a single "prototype" vector per class.
   - Each candidate title is embedded, and its cosine similarity to both prototypes is computed.
   - `SIMILARITY_TEMPERATURE` scales the similarity gap before a sigmoid turns it into a pseudo-probability of "tech-ness". A threshold of `0.6` decides inclusion in the final feed.

Tuning the feed's behavior (broader/narrower topic definition, stricter/looser inclusion) is done by editing the seed text lists, `SIMILARITY_TEMPERATURE`, or the `0.6` threshold — not by adding ML training code.

## Recreating this from scratch

Prompt used to one-shot the original implementation:

> Write a single Python script that builds a "tech-only" Hacker News feed.
>
> **Fetching:**
> - Use the Algolia HN Search API (`https://hn.algolia.com/api/v1/search_by_date`) to fetch all `story`-tagged posts from the last 7 days (`created_at_i` numeric filter, computed from the current Unix timestamp).
> - Use `hitsPerPage=1000` (the max) to minimize requests.
> - Fetch page 0 first to learn `nbPages` from the response, then fetch the remaining pages concurrently with a `ThreadPoolExecutor` (8 workers), reusing a single `requests.Session`.
> - Handle non-200 responses gracefully (print an error, skip that page).
>
> **Filtering:**
> - Keep only posts with more than 5 points.
>
> **Classification (no training, zero-shot via local embeddings):**
> - Use a local Ollama server (`http://localhost:11434/api/embed`) with the `embeddinggemma` model to embed text. Prefix every input with `"task: classification | query: "` before embedding.
> - Define two small hardcoded lists of example headlines: ~6 clearly tech-related titles (e.g. Show HN posts, programming languages, GPUs, Kubernetes, compilers, AI APIs) and ~6 clearly non-tech titles (science, history, politics, health, sports, memoir).
> - Embed each list and average the vectors into a single "tech prototype" and "non-tech prototype".
> - Embed all candidate post titles in one batch call.
> - For each title vector, compute cosine similarity to both prototypes, scale the gap between them by a `SIMILARITY_TEMPERATURE` constant (start at 10.0), and pass through a sigmoid to get a pseudo-probability of "tech-ness".
> - Keep posts with probability >= 0.6.
>
> **Output:**
> - Print total posts retrieved, then each tech-classified post as `- [{points}pts] [{probability:.2f} tech] {title} by {author}`, then a final count.
>
> Keep it to one flat script (no CLI framework, no classes, no tests) — top-level constants for the tunable knobs (temperature, threshold, seed texts), small helper functions for embedding/cosine similarity/averaging/page-fetching, and execute everything at the bottom of the file.

Note: this produces a script with no `if __name__ == "__main__":` guard, so importing the module runs the whole pipeline.
