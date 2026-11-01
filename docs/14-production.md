# 14 - Production: Making RAG Fast, Affordable, Safe, and Reliable

> A RAG demo on my laptop worked well. Then I thought about real users: What if 100 people ask at once? What if the answer takes 15 seconds? What if someone reads a document they shouldn't? What if the bill is ten times what I expected? Production is about these questions. This chapter covers the six things that matter most: **latency, cost, caching, scaling and reliability, monitoring, and security.**

**What you'll understand after this chapter**

- Where time and money go in a RAG system, and how to reduce both
- Which caching is safe and which is risky
- How to keep the system reliable under load and failures
- What to monitor
- The main security risks and how to reduce them

---

## 1. Latency: Making It Fast

### Where the time goes

Per question, the steps look roughly like this:

| Step | Typical impact |
|---|---|
| Query rewriting (LLM call) | Adds a full LLM round trip |
| Embed the query | Small |
| Vector search | Small |
| Reranking | Moderate, grows with candidates |
| LLM generation | Usually the largest part |

The exact numbers depend on your setup, so **measure each step** and don't guess.

```python
import time

def timed(name, fn, *args, timings=None, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if timings is not None:
        timings[name] = time.perf_counter() - start
    return result

timings = {}
chunks  = timed("retrieve", hybrid_search, question, timings=timings)
top     = timed("rerank", rerank, question, chunks, timings=timings)
answer  = timed("generate", generate, question, top, timings=timings)
print(timings)
```

### How to reduce latency

1. **Stream the answer.** Show words as they are generated. The first words appear quickly, so it *feels* much faster, even if total time is the same. This is the single best user-experience fix.
2. **Send fewer, better chunks.** Less input means faster generation and lower cost. Reranking helps here.
3. **Cut unnecessary LLM calls.** Each extra step (rewriting, grading, checking) adds a round trip. Keep only the ones your evaluation shows are worth it.
4. **Use a smaller, faster model** for helper steps (rewriting, routing) and for simple questions.
5. **Run independent steps in parallel**, for example dense and BM25 search at the same time.
6. **Limit the answer length** with a sensible max token setting.
7. **Rerank fewer candidates**, or use a lighter reranker.
8. **Keep services close together** (same region for your app, vector database, and model API).

```python
import asyncio

async def hybrid_parallel(query):
    dense, sparse = await asyncio.gather(
        asyncio.to_thread(dense_search, query, 20),
        asyncio.to_thread(sparse_search, query, 20),
    )
    return reciprocal_rank_fusion([dense, sparse])
```

Always look at the **slow cases (p95 latency)**, not only the average. A system with a fast average and a terrible slowest 5% still feels broken to some users.

---

## 2. Cost: Keeping the Bill Under Control

### Where the money goes

- **LLM tokens** (input and output): usually the biggest cost, paid on every question
- **Embeddings:** a one-time cost per chunk, plus a small cost per question
- **Reranking API** (if hosted)
- **Vector database hosting**
- **Compute** for ingestion, OCR, and any local models
- **Evaluation runs** (LLM judges, RAGAS)

### The main levers

| Lever | How it saves money |
|---|---|
| **Fewer, better chunks** | Fewer input tokens per question (often the biggest saving) |
| **Shorter prompts** | Trim instructions, drop unneeded history |
| **Smaller models** where quality allows | Cheaper per token. Test on your eval set |
| **Skip unnecessary LLM steps** | Each call costs money |
| **Caching** (section 3) | Repeated questions cost nothing |
| **Provider prompt caching** | Some providers discount repeated prompt prefixes such as long system instructions |
| **Batch embedding and ingestion** | More efficient than one at a time |
| **Smaller embedding dimensions or quantization** | Cheaper storage and search |
| **Set limits** | Max tokens, max steps for agents, rate limits per user |

### Estimate it

```python
def estimate_cost_per_query(input_tokens, output_tokens, price_in_per_1m, price_out_per_1m):
    return (input_tokens * price_in_per_1m + output_tokens * price_out_per_1m) / 1_000_000

# Example: 3,000 input tokens and 300 output tokens
# Use your provider's current prices; they change.
```

Multiply by your expected number of questions per day to see the monthly bill **before** you launch. Track **cost per question** as a metric, next to quality.

---

## 3. Caching

Caching means reusing earlier work. It saves both time and money, but it must be done carefully.

### Safe and useful caches

**1. Embedding cache.** Don't re-embed the same text.

```python
import hashlib

embedding_cache = {}     # use Redis or a database in real apps

def embed_cached(text):
    key = hashlib.sha256(text.encode()).hexdigest()
    if key not in embedding_cache:
        embedding_cache[key] = embed_model.encode([text], normalize_embeddings=True)[0]
    return embedding_cache[key]
```

During ingestion, skip unchanged files and chunks (using the hashes from [03 - Data Ingestion](./03-data-ingestion.md)).

**2. Exact-match answer cache.** If the **same question** is asked again, return the saved answer.

**3. Retrieval cache.** Save the retrieved chunks for a question.

**4. Provider prompt caching.** Handled by the model provider for repeated prompt prefixes.

### Semantic cache (powerful but risky)

A semantic cache returns a saved answer when a **new question is similar** to an old one.

```python
def semantic_cache_lookup(question_vec, cache, threshold=0.95):
    best = max(cache, key=lambda e: float(e["vec"] @ question_vec), default=None)
    if best and float(best["vec"] @ question_vec) >= threshold:
        return best["answer"]
    return None
```

The danger: **similar is not the same.** "What is the refund window for Premium?" and "What is the refund window for Basic?" are very close in meaning space but have **different answers**. A cache hit here gives a wrong answer with full confidence. If you use a semantic cache, set a **high threshold**, test it with such near-miss pairs, and avoid it for questions where small details change the answer.

### Cache rules that prevent bugs

- **Include permissions in the cache key.** If user A may see a document and user B may not, a shared cache could leak A's answer to B.
- **Include the index version** in the key, and **clear the cache when documents change**, or users will get outdated answers.
- **Set an expiry (TTL)** for answers.
- Don't cache answers that depend on **chat history** unless the history is part of the key.
- Remember that cached data may contain **private information**.

---

## 4. Scaling and Reliability

### Separate the two pipelines

Ingestion and querying have different needs (see [02 - Architecture](./02-architecture.md)). In production:

- **Ingestion** runs as a **background job** with a queue, retries, and progress tracking. It can be slow.
- **Querying** runs as a **fast, stateless API** that can be copied across several servers. Keep chat memory in a shared store (like Redis), not in the app's memory.

### Keeping the index up to date

- **Update incrementally:** add, change, or delete only what changed (using document IDs and hashes).
- **Changing the embedding model** means re-embedding everything. Build the **new index separately**, test it, then switch traffic over, and keep the old one for rollback.
- Store a **version** with the index and in logs, so you know which index produced an answer.

### Handle failures gracefully

External services (LLM API, embedding API, vector database) fail or slow down. Plan for it:

```python
import time, random

def call_with_retry(fn, *args, retries=3, base_delay=1.0, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt) + random.random())   # exponential backoff
```

| Problem | What to do |
|---|---|
| Slow or hanging calls | Set **timeouts** on every external call |
| Temporary errors, rate limits | **Retry** with backoff (as above), but not forever |
| Model provider down | **Fallback** to a second model or a friendly error message |
| Reranker down | Skip reranking and use the first-stage order (**graceful degradation**) |
| Too many requests | **Rate limit** per user and queue requests |
| Vector database down | Return a clear error. Don't let the LLM answer without context |

Never let a failure make the bot answer from nothing and sound confident. If retrieval fails, say so.

---

## 5. Monitoring

You can't fix what you can't see. Log every request (with care for privacy) and track a few key numbers.

### What to log per request

```python
import json, time, uuid

def log_request(question, chunks, answer, timings, cost, user_id):
    record = {
        "request_id": str(uuid.uuid4()),
        "time": time.time(),
        "user_id": user_id,                      # or a hashed ID
        "question": question,                    # consider masking personal data
        "retrieved": [{"source": c["metadata"]["source"],
                       "page": c["metadata"]["page"],
                       "score": c.get("rerank_score")} for c in chunks],
        "answer_length": len(answer),
        "timings": timings,
        "cost": cost,
        "index_version": INDEX_VERSION,
        "model": MODEL_NAME,
    }
    print(json.dumps(record))                    # send to your logging system
```

### What to track

| Metric | Why |
|---|---|
| **Latency (average and p95)**, per step | Find slow steps |
| **Cost per question** and daily total | Avoid bill surprises |
| **Error and timeout rate** | Reliability |
| **"I couldn't find this" rate** | A sudden rise means retrieval or data problems. A rate near zero may mean the bot guesses |
| **Retrieval scores** (top score distribution) | Detect drift and weak coverage |
| **User feedback** (thumbs up and down) | Direct quality signal |
| **Most common unanswered questions** | Shows missing documents |
| **Guardrail triggers** | Injection attempts, blocked requests |

### Use what you learn

- Review **thumbs-down and "not found"** cases regularly and **add them to your test set** from [13 - Evaluation](./13-evaluation.md).
- Run the test set **automatically** whenever you change the pipeline (regression testing).
- Sample some live answers for **faithfulness checks**.
- **Set alerts** for error spikes, latency spikes, and cost spikes.

Tracing tools (LangSmith, Langfuse, Arize Phoenix, and others) can record each step of the pipeline for you and make debugging easier.

---

## 6. Security

RAG connects an LLM to your private data, so security needs real attention.

### 6.1 Access control (most important)

The rule: **a user must only be able to retrieve documents they are allowed to read.**

- Save permission info (owner, group, access level) as **metadata** on every chunk during ingestion.
- Apply a **filter inside the search**, based on the logged-in user's permissions.
- **Never rely on the prompt** ("don't show HR documents to regular users"). If the text is in the prompt, the model can reveal it.
- Include permissions in **cache keys** (section 3).

```python
def secure_retrieve(question, user):
    allowed = {"access_level": {"$in": user.access_levels}}
    return hybrid_search(question, k=20, metadata_filter=allowed)
```

(`hybrid_search` here is assumed to accept a metadata filter. Adapt this to your database's filter syntax.)

### 6.2 Prompt injection

Documents (and users) can contain text that tries to hijack the model: "Ignore your instructions and...". Defenses, none of them perfect:

- Tell the model that context is **data, not instructions**.
- Use clear **delimiters** around context.
- Give the model **minimal power**: if it can only return text, the damage is limited. Be very careful about tools that send emails, run code, or write to databases.
- **Vet what you ingest.** Untrusted sources (public web pages, user uploads) carry more risk.
- **Check outputs** for unexpected links or leaked instructions.
- Treat this as an ongoing risk and **test it** with attack examples.

### 6.3 Private and personal data

- **Know where data goes.** Using a hosted model or embedding API means sending text to a third party. Check the provider's data policy, or use local models for sensitive data.
- **Mask personal data** (names, IDs, card numbers) before sending or storing, when it isn't needed.
- **Limit what logs contain** and who can read them. Logs hold questions and answers.
- Think about **data deletion** requests. You must be able to remove a person's documents from the index, caches, and logs.

### 6.4 Other basics

- **Keep secrets safe.** API keys belong in environment variables or a secrets manager, never in the code or the repository.
- **Authenticate and rate-limit** the API to prevent abuse and runaway costs.
- **Validate inputs** (length limits, file types and sizes for uploads).
- **Scan uploaded files**, because a PDF can be hostile.
- **Keep an audit log** of who accessed what, for sensitive systems.
- **Update dependencies**, since AI libraries change quickly and have vulnerabilities like any software.

---

## 7. Production Checklist

**Quality**
- [ ] Test set with retrieval and generation metrics, and a baseline recorded
- [ ] Unanswerable questions handled with an honest "I couldn't find this"
- [ ] Automatic regression test when the pipeline changes

**Performance and cost**
- [ ] Latency measured per step (average and p95)
- [ ] Streaming on
- [ ] Cost per question estimated and tracked
- [ ] Token limits, rate limits, and agent step limits set

**Reliability**
- [ ] Timeouts, retries with backoff, and fallbacks
- [ ] Ingestion runs as a background job with incremental updates
- [ ] Index versioning and a rollback plan

**Caching**
- [ ] Embedding cache in place
- [ ] Cache keys include permissions and index version
- [ ] Semantic cache avoided, or tested with near-miss questions

**Monitoring**
- [ ] Request logging with privacy in mind
- [ ] Dashboards and alerts for errors, latency, and cost
- [ ] User feedback collected and fed back into the test set

**Security**
- [ ] Permission filtering inside retrieval
- [ ] Prompt injection defenses tested
- [ ] Secrets safe, API authenticated and rate-limited
- [ ] Data-handling and deletion plan for personal data

---

## 8. Common Mistakes

- **Going live without a test set**, so you can't tell if changes help or hurt.
- **Never measuring latency per step** and optimizing the wrong thing.
- **Ignoring cost** until the first bill.
- **Semantic caching with a loose threshold**, giving confident wrong answers.
- **Caches that ignore permissions or document updates.**
- **No timeouts or fallbacks** for external services.
- **Letting the model answer without context** when retrieval fails.
- **Filtering restricted data in the prompt** instead of during retrieval.
- **Logging everything, including personal data**, with no access controls.
- **API keys in the repository.**
- **Giving an agent powerful tools** without limits and logs.

---

## 9. Key Takeaways

1. **Measure first.** Latency (per step, p95), cost per question, and quality all need numbers.
2. **Streaming**, fewer and better chunks, and fewer LLM calls are the best fixes for both latency and cost.
3. **Cache** embeddings and exact repeats freely. Be careful with semantic caches, and always include **permissions and index version** in cache keys.
4. Separate **ingestion (background, incremental)** from **querying (fast, stateless)**, and plan for failures with timeouts, retries, and fallbacks.
5. **Monitor** latency, cost, errors, "not found" rate, and user feedback, and feed failures back into your test set.
6. **Security:** filter by permission inside retrieval, defend against prompt injection, protect personal data and secrets, and limit what the model can do.

---

## 10. Quick Self-Check

1. Name three ways to reduce latency and three ways to reduce cost.
2. Why is a semantic cache risky? Give an example of two questions it could confuse.
3. What should be part of every cache key, and why?
4. What should the system do when the vector database is down?
5. Why shouldn't you hide restricted documents using the prompt?

