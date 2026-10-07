````python
import os
import json
import random
import time
import requests
import feedparser
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

REQUEST_TIMEOUT = 180

COHERE_QUERY_MODEL = "command-a-plus-05-2026"

PRIMARY_MODELS = [
    "z-ai/glm-5-3-flash",
    "z-ai/glm-5-3",
]

FALLBACK_MODELS = [
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "google/gemma-4-31b-it",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
]

# The weakest fallback is deliberately last.
DUMBEST_LAST_MODEL = "openai/gpt-oss-20b"


# ============================================================
# SECRET HELPERS
# ============================================================

def get_required_secret(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"{name} is missing from the GitHub Actions environment. "
            "Add it under GitHub Settings -> Secrets and variables -> Actions."
        )

    return value


NVIDIA_NIM_API_KEY = get_required_secret("NVIDIA_NIM_API_KEY")
COHERE_API_KEY = get_required_secret("COHERE_API_KEY")


# ============================================================
# COHERE CLIENT
# ============================================================

# Official Cohere OpenAI-compatible endpoint.
cohere_client = OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key=COHERE_API_KEY,
)


# ============================================================
# RSS SOURCES
# ============================================================

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return " ".join(str(value).split()).strip()


def truncate_text(value, max_chars):
    value = clean_text(value)

    if len(value) <= max_chars:
        return value

    return value[:max_chars].rstrip() + "..."


def extract_response_text(data):
    """
    Extract text from an NVIDIA OpenAI-compatible response.
    """

    try:
        choices = data.get("choices", [])

        if not choices:
            return ""

        message = choices[0].get("message", {})

        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            pieces = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        pieces.append(str(text))

            return "\n".join(pieces).strip()

    except Exception:
        pass

    return ""


# ============================================================
# STEP 1 — RSS COLLECTION
# ============================================================

def collect_rss_articles():
    print("[1/5] Collecting RSS financial and technology news...")

    articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:10]:
                title = clean_text(entry.get("title"))
                summary = clean_text(
                    entry.get("summary")
                    or entry.get("description")
                    or ""
                )
                link = clean_text(entry.get("link"))

                if not title:
                    continue

                articles.append(
                    {
                        "title": title,
                        "summary": truncate_text(summary, 1200),
                        "link": link,
                        "source": feed_url,
                    }
                )

        except Exception as exc:
            print(f"  RSS error for {feed_url}: {exc}")

    # Remove duplicate titles.
    deduplicated = []
    seen_titles = set()

    for article in articles:
        key = article["title"].lower()

        if key in seen_titles:
            continue

        seen_titles.add(key)
        deduplicated.append(article)

    print(f"  RSS articles collected: {len(deduplicated)}")

    return deduplicated


# ============================================================
# STEP 2 — COHERE AUTONOMOUS RESEARCH PLANNER
# ============================================================

def autonomous_agent_search(articles):
    print("[2/5] Asking Cohere to generate autonomous research queries...")

    article_context = "\n".join(
        f"- {article['title']}: {article['summary']}"
        for article in articles[:20]
    )

    agent_prompt = f"""
You are the autonomous research planner for a financial intelligence system.

Based on the news headlines below, generate exactly THREE useful web-search
queries.

The three queries MUST cover:

1. Global financial markets
2. Technology, AI, and semiconductor companies
3. Global macroeconomics, inflation, interest rates, central banks, trade, or GDP

Requirements:

- Return ONLY valid JSON.
- Return an object with exactly one key: "queries".
- "queries" must contain exactly three strings.
- Each string must be an actual search query.
- Do not include numbering.
- Do not include explanations.
- Do not repeat the instructions.
- Do not write things like "We need to..." or "The queries should..."
- Keep each query concise and useful for a search engine.

Example format:

{{
  "queries": [
    "latest global stock bond commodity markets central banks",
    "latest AI semiconductor technology company earnings market news",
    "latest inflation interest rates GDP central bank global economy"
  ]
}}

Recent news:

{article_context}
"""

    try:
        response = cohere_client.chat.completions.create(
            model=COHERE_QUERY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON-producing research planner. "
                        "Never echo the user's instructions."
                    ),
                },
                {
                    "role": "user",
                    "content": agent_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )

        raw = response.choices[0].message.content.strip()

        print("  Cohere raw planner output:")
        print(f"    {raw}")

        # Handle accidental markdown fences.
        if raw.startswith("```"):
            raw = raw.replace("```json", "", 1)
            raw = raw.replace("```", "")
            raw = raw.strip()

        parsed = json.loads(raw)

        queries = parsed.get("queries")

        if not isinstance(queries, list):
            raise ValueError("Cohere response did not contain a queries list.")

        queries = [
            clean_text(query)
            for query in queries
            if isinstance(query, str) and clean_text(query)
        ]

        if len(queries) != 3:
            raise ValueError(
                f"Cohere returned {len(queries)} valid queries instead of 3."
            )

        print("  Research queries:")

        for query in queries:
            print(f"    - {query}")

        return queries

    except Exception as exc:
        print(f"  Cohere planner failed: {exc}")
        print("  Using safe research-query fallback.")

        fallback_queries = [
            "latest global financial markets stocks bonds commodities central banks",
            "latest AI technology semiconductor companies earnings market news",
            "latest global economy inflation interest rates trade GDP central banks",
        ]

        for query in fallback_queries:
            print(f"    - {query}")

        return fallback_queries


# ============================================================
# STEP 3 — SERPAPI SEARCH
# ============================================================

def search_serpapi(query):
    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        return []

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": 8,
                "hl": "en",
                "gl": "us",
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for result in data.get("organic_results", []):
            title = clean_text(result.get("title"))
            snippet = clean_text(result.get("snippet"))
            link = clean_text(result.get("link"))

            if not title:
                continue

            results.append(
                {
                    "title": title,
                    "summary": snippet,
                    "link": link,
                    "source": "SerpAPI",
                }
            )

        return results

    except Exception as exc:
        print(f"  SerpAPI error for '{query}': {exc}")
        return []


def collect_serpapi_results(queries):
    print("[3/5] Running autonomous web research...")

    all_results = []

    for query in queries:
        print(f"  Searching: {query}")

        results = search_serpapi(query)

        print(f"    Results: {len(results)}")

        all_results.extend(results)

        time.sleep(1)

    # Deduplicate.
    deduplicated = []
    seen = set()

    for result in all_results:
        key = (
            result["link"]
            or result["title"]
        ).lower()

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(result)

    print(f"  Total unique web results: {len(deduplicated)}")

    return deduplicated


# ============================================================
# STEP 4 — ALPHA VANTAGE
# ============================================================

def collect_alpha_vantage_news():
    print("[4/5] Collecting Alpha Vantage market news...")

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        print("  ALPHA_VANTAGE_API_KEY not configured.")
        return []

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets,economy_fiscal,economy_monetary,technology",
                "limit": 20,
                "apikey": api_key,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        feed = data.get("feed", [])

        results = []

        for item in feed:
            title = clean_text(item.get("title"))
            summary = clean_text(item.get("summary"))
            url = clean_text(item.get("url"))

            if not title:
                continue

            results.append(
                {
                    "title": title,
                    "summary": truncate_text(summary, 1200),
                    "link": url,
                    "source": "Alpha Vantage",
                }
            )

        print(f"  Alpha Vantage articles: {len(results)}")

        return results

    except Exception as exc:
        print(f"  Alpha Vantage error: {exc}")
        return []


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_model_order():
    """
    Model order:

        1. GLM-5.3-Flash
        2. GLM-5.3
        3-7. Randomized fallback models
        LAST. GPT-OSS-20B

    This means the system gets randomness without ever accidentally
    putting the weakest fallback ahead of the stronger models.
    """

    randomized = [
        model
        for model in FALLBACK_MODELS
        if model != DUMBEST_LAST_MODEL
    ]

    random.shuffle(randomized)

    return (
        PRIMARY_MODELS
        + randomized
        + [DUMBEST_LAST_MODEL]
    )


def model_payload(model, prompt):
    """
    Build an NVIDIA-compatible request.

    Some models have model-specific reasoning controls, so only add
    parameters where they are known to be appropriate.
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial research analyst. "
                    "Produce accurate, concise, evidence-grounded analysis. "
                    "Do not invent facts, prices, events, or sources."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2500,
        "stream": False,
    }

    # GLM supports explicit reasoning effort.
    if model in {
        "z-ai/glm-5-3-flash",
        "z-ai/glm-5-3",
    }:
        payload["reasoning_effort"] = "max"

    # Nemotron 3 uses the NVIDIA chat-template thinking switch.
    elif model in {
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
    }:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True
        }

    return payload


# ============================================================
# STEP 5 — NVIDIA MODEL CALL
# ============================================================

def call_nvidia_model(model, prompt):
    print(f"\n  >>> Trying model: {model}")

    payload = model_payload(model, prompt)

    try:
        response = requests.post(
            NVIDIA_CHAT_URL,
            headers={
                "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"      HTTP status: {response.status_code}")

        if not response.ok:
            error_body = response.text[:2000]

            print("      NVIDIA API error:")
            print(f"      {error_body}")

            raise RuntimeError(
                f"NVIDIA returned HTTP {response.status_code}: "
                f"{error_body}"
            )

        data = response.json()

        text = extract_response_text(data)

        if not text:
            raise RuntimeError(
                "NVIDIA returned a successful response but no text content."
            )

        print(
            f"      ✓ {model} succeeded "
            f"({len(text)} characters)"
        )

        return text

    except requests.Timeout:
        raise RuntimeError(
            f"{model} timed out after {REQUEST_TIMEOUT} seconds."
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"{model} network error: {exc}"
        )

    except ValueError as exc:
        raise RuntimeError(
            f"{model} returned invalid JSON: {exc}"
        )


def generate_daily_summary(rss_articles, web_results, market_news):
    print("\n[5/5] Generating financial intelligence brief...")

    combined_sources = []

    for article in rss_articles:
        combined_sources.append(
            {
                "title": article["title"],
                "summary": article["summary"],
                "link": article["link"],
                "source": "RSS",
            }
        )

    for result in web_results:
        combined_sources.append(result)

    for result in market_news:
        combined_sources.append(result)

    # Prevent the prompt from becoming unnecessarily enormous.
    combined_sources = combined_sources[:80]

    source_text_parts = []

    for index, item in enumerate(combined_sources, start=1):
        source_text_parts.append(
            f"""
SOURCE {index}
Title: {item.get('title', '')}
Source: {item.get('source', '')}
Summary: {truncate_text(item.get('summary', ''), 1000)}
URL: {item.get('link', '')}
""".strip()
        )

    source_text = "\n\n".join(source_text_parts)

    prompt = f"""
Create today's autonomous financial intelligence brief.

Use ONLY the supplied research material as factual grounding.

Analyze:

1. Global financial markets
2. Stocks and major market movements
3. AI and technology companies
4. Semiconductor industry
5. Macroeconomics
6. Inflation and interest rates
7. Central-bank developments
8. Important risks or opportunities
9. What deserves attention next

Rules:

- Clearly distinguish facts from interpretation.
- Do not invent numbers.
- Do not claim something happened unless the supplied sources support it.
- If sources disagree, say so.
- Prefer recent developments.
- Avoid generic filler.
- Keep it readable for a human investor/researcher.
- Use concise headings and bullet points.
- Include a "Key Takeaways" section.
- Include a "What To Watch Next" section.
- Do not provide personalized financial advice.
- Do not tell the reader to buy or sell a specific security.

Research material:

{source_text}
"""

    model_order = build_model_order()

    print("\n  MODEL FALLBACK ORDER:")

    for position, model in enumerate(model_order, start=1):
        print(f"    {position}. {model}")

    failures = []

    for model in model_order:
        try:
            summary = call_nvidia_model(model, prompt)

            if summary and len(summary.strip()) >= 100:
                print(
                    f"\n  ✓ Daily brief generated successfully "
                    f"using {model}"
                )

                return summary.strip(), model

            failure = f"{model}: response was too short"
            print(f"      ✗ {failure}")
            failures.append(failure)

        except Exception as exc:
            failure = f"{model}: {exc}"

            print(f"      ✗ {failure}")

            failures.append(failure)

            # Small delay before moving to the next provider/model.
            time.sleep(1)

    failure_text = "\n".join(
        f"  - {failure}"
        for failure in failures
    )

    raise RuntimeError(
        "ALL NVIDIA MODELS FAILED.\n"
        "The financial brief was not generated.\n\n"
        f"{failure_text}"
    )


# ============================================================
# DISCORD
# ============================================================

def send_to_discord(summary, model_used):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("DISCORD_WEBHOOK_URL is not configured.")
        return False

    content = (
        f"**🤖 Autonomous Financial AI Brief**\n"
        f"*Generated by `{model_used}`*\n\n"
        f"{summary}"
    )

    # Discord has a 2000-character message limit.
    chunks = [
        content[i:i + 1900]
        for i in range(0, len(content), 1900)
    ]

    try:
        for chunk in chunks:
            response = requests.post(
                webhook_url,
                json={
                    "content": chunk
                },
                timeout=30,
            )

            response.raise_for_status()

            time.sleep(0.5)

        print("✓ Discord delivery successful.")

        return True

    except Exception as exc:
        print(f"Discord delivery failed: {exc}")
        return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("AUTONOMOUS FINANCIAL AI DISPATCH")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # 1. RSS
    # --------------------------------------------------------

    rss_articles = collect_rss_articles()

    # --------------------------------------------------------
    # 2. Cohere autonomous research planning
    # --------------------------------------------------------

    queries = autonomous_agent_search(rss_articles)

    # --------------------------------------------------------
    # 3. Search
    # --------------------------------------------------------

    web_results = collect_serpapi_results(queries)

    # --------------------------------------------------------
    # 4. Market data/news
    # --------------------------------------------------------

    market_news = collect_alpha_vantage_news()

    # --------------------------------------------------------
    # 5. AI synthesis + fallback ladder
    # --------------------------------------------------------

    summary, model_used = generate_daily_summary(
        rss_articles=rss_articles,
        web_results=web_results,
        market_news=market_news,
    )

    # --------------------------------------------------------
    # Discord
    # --------------------------------------------------------

    if not send_to_discord(summary, model_used):
        raise RuntimeError(
            "The financial brief was generated successfully, "
            "but Discord delivery failed."
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Model used: {model_used}")
    print(f"Runtime: {elapsed:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
        raise

    except Exception as exc:
        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(str(exc))
        print("=" * 70)

        # Important:
        # Exit non-zero so GitHub Actions correctly marks the run FAILED.
        raise
````
