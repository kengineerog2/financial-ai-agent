import os
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# 30 seconds is the TOTAL request timeout.
# This is intentionally aggressive because this pipeline
# does not need deep reasoning.
REQUEST_TIMEOUT = 30

MAX_WORKERS = 24

COHERE_QUERY_MODEL = "command-a-plus-05-2026"


# ============================================================
# NVIDIA MODEL FALLBACK LADDER
# ============================================================

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

DUMBEST_LAST_MODEL = "openai/gpt-oss-20b"


# ============================================================
# RSS FEEDS
# ============================================================

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=reuters",
    "http://rss.cnn.com/rss/edition.rss",
    "https://www.theguardian.com/world/rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://apnews.com/hub/ap-top-news/feed",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.ft.com/?format=rss",
    "https://www.economist.com/rss",
    "https://feeds.bloomberg.com/technology/news.rss",
    "https://www.forbes.com/business/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://techcrunch.com/feed/",
    "https://hnrss.org/frontpage",
    "https://www.technologyreview.com/feed/",
    "https://www.wired.com/feed/rss",
]


# ============================================================
# SECRETS
# ============================================================

def get_required_secret(name):
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"{name} is missing from the GitHub Actions environment."
        )

    return value


NVIDIA_NIM_API_KEY = get_required_secret(
    "NVIDIA_NIM_API_KEY"
)

COHERE_API_KEY = get_required_secret(
    "COHERE_API_KEY"
)


# ============================================================
# COHERE OPENAI-COMPATIBLE CLIENT
# ============================================================

cohere_client = OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key=COHERE_API_KEY,
)


# ============================================================
# HTTP SESSION FACTORY
# ============================================================

def make_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; FinancialAIResearchBot/1.0)"
            ),
            "Accept": "*/*",
        }
    )

    return session


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


def truncate_text(value, max_chars):
    value = clean_text(value)

    if len(value) <= max_chars:
        return value

    return (
        value[:max_chars]
        .rstrip()
        + "..."
    )


# ============================================================
# GENERIC RESPONSE EXTRACTION
# ============================================================

def extract_response_text(data):
    """
    Handles several OpenAI-compatible response formats.

    Some reasoning models may expose:
        message.content

Others may return content as blocks.

We deliberately do NOT use reasoning_content as the final
answer because internal reasoning is not the financial brief.
"""

    if not isinstance(data, dict):
        return ""

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        return ""

    choice = choices[0]

    if not isinstance(choice, dict):
        return ""

    message = choice.get("message")

    if isinstance(message, dict):

        content = message.get("content")

        # Normal OpenAI-compatible response.
        if isinstance(content, str):
            return content.strip()

        # Some APIs return content blocks.
        if isinstance(content, list):

            pieces = []

            for item in content:

                if isinstance(item, str):
                    pieces.append(item)

                elif isinstance(item, dict):

                    text = item.get("text")

                    if isinstance(text, str):
                        pieces.append(text)

            result = "\n".join(
                piece for piece in pieces
                if piece
            ).strip()

            if result:
                return result

        # Some providers may put the generated text here.
        for key in (
            "output_text",
            "text",
        ):
            value = message.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    # Last-resort top-level fields.
    for key in (
        "output_text",
        "text",
    ):
        value = choice.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


# ============================================================
# RSS
# ============================================================

def fetch_rss_feed(feed_url):

    print(
        f"  RSS -> {feed_url}",
        flush=True,
    )

    try:

        session = make_session()

        response = session.get(
            feed_url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        feed = feedparser.parse(
            response.content
        )

        articles = []

        for entry in feed.entries[:10]:

            title = clean_text(
                entry.get("title")
            )

            summary = clean_text(
                entry.get("summary")
                or entry.get("description")
                or ""
            )

            link = clean_text(
                entry.get("link")
            )

            if not title:
                continue

            articles.append(
                {
                    "title": title,
                    "summary": truncate_text(
                        summary,
                        1200,
                    ),
                    "link": link,
                    "source": feed_url,
                }
            )

        print(
            f"  RSS ✓ {feed_url} -> "
            f"{len(articles)} articles",
            flush=True,
        )

        return articles

    except requests.Timeout:

        print(
            f"  RSS ⏱ TIMEOUT after "
            f"{REQUEST_TIMEOUT}s -> {feed_url}",
            flush=True,
        )

        return []

    except requests.RequestException as exc:

        print(
            f"  RSS ✗ HTTP/network error -> "
            f"{feed_url}: {exc}",
            flush=True,
        )

        return []

    except Exception as exc:

        print(
            f"  RSS ✗ parser error -> "
            f"{feed_url}: {exc}",
            flush=True,
        )

        return []


def collect_rss_articles():

    print(
        f"[1/5] Collecting "
        f"{len(RSS_FEEDS)} RSS feeds concurrently...",
        flush=True,
    )

    all_articles = []

    with ThreadPoolExecutor(
        max_workers=min(
            MAX_WORKERS,
            len(RSS_FEEDS),
        )
    ) as executor:

        futures = {
            executor.submit(
                fetch_rss_feed,
                feed_url,
            ): feed_url
            for feed_url in RSS_FEEDS
        }

        for future in as_completed(futures):

            feed_url = futures[future]

            try:

                articles = future.result()

                all_articles.extend(
                    articles
                )

            except Exception as exc:

                print(
                    f"  RSS worker crashed -> "
                    f"{feed_url}: {exc}",
                    flush=True,
                )

    # Deduplicate headlines.
    deduplicated = []

    seen_titles = set()

    for article in all_articles:

        key = article["title"].lower()

        if key in seen_titles:
            continue

        seen_titles.add(key)

        deduplicated.append(
            article
        )

    print(
        f"  RSS collection complete: "
        f"{len(deduplicated)} unique articles",
        flush=True,
    )

    return deduplicated


# ============================================================
# COHERE QUERY PLANNER
# ============================================================

def extract_cohere_content(response):
    """
    Extract Cohere's actual answer.

    The previous run demonstrated that the compatibility layer
    may return a response object whose visible content needs
    to be handled carefully.
    """

    try:

        message = response.choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        if isinstance(
            content,
            list,
        ):

            pieces = []

            for item in content:

                if isinstance(
                    item,
                    str,
                ):
                    pieces.append(item)

                elif isinstance(
                    item,
                    dict,
                ):

                    text = item.get(
                        "text"
                    )

                    if text:
                        pieces.append(
                            str(text)
                        )

            return "\n".join(
                pieces
            ).strip()

    except Exception:
        pass

    return ""


def autonomous_agent_search(
    articles
):

    print(
        "[2/5] Asking Cohere to generate "
        "autonomous research queries...",
        flush=True,
    )

    article_context = "\n".join(
        f"- {article['title']}: "
        f"{article['summary']}"
        for article in articles[:30]
    )

    agent_prompt = f"""
Generate exactly three search-engine queries for a financial
intelligence system.

Cover:

1. Global financial markets
2. AI, technology, and semiconductor companies
3. Global macroeconomics, inflation, interest rates,
   central banks, trade, or GDP

Return ONLY this JSON object:

{{
  "queries": [
    "query one",
    "query two",
    "query three"
  ]
}}

Do not explain anything.
Do not include markdown.
Do not include reasoning.
Do not repeat the instructions.

Recent headlines:

{article_context}
"""

    try:

        response = cohere_client.chat.completions.create(
            model=COHERE_QUERY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only the requested JSON. "
                        "Do not explain your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": agent_prompt,
                },
            ],
            temperature=0,
            max_tokens=300,
        )

        raw = extract_cohere_content(
            response
        )

        print(
            "  Cohere planner output:",
            flush=True,
        )

        print(
            raw,
            flush=True,
        )

        # Try to locate JSON even if the model
        # unfortunately added surrounding text.
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1:

            raw_json = raw[
                start:end + 1
            ]

            parsed = json.loads(
                raw_json
            )

            queries = parsed.get(
                "queries"
            )

            if (
                isinstance(
                    queries,
                    list,
                )
                and len(queries) == 3
            ):

                cleaned = [
                    clean_text(query)
                    for query in queries
                    if isinstance(
                        query,
                        str,
                    )
                    and clean_text(query)
                ]

                if len(cleaned) == 3:

                    print(
                        "  ✓ Cohere generated "
                        "3 queries.",
                        flush=True,
                    )

                    return cleaned

        raise ValueError(
            "Could not extract exactly "
            "three search queries."
        )

    except Exception as exc:

        print(
            f"  Cohere planner failed: {exc}",
            flush=True,
        )

        print(
            "  Using deterministic research "
            "queries.",
            flush=True,
        )

    fallback_queries = [
        (
            "latest global financial markets "
            "stocks bonds commodities central banks"
        ),
        (
            "latest AI technology semiconductor "
            "companies earnings market news"
        ),
        (
            "latest global economy inflation "
            "interest rates trade GDP central banks"
        ),
    ]

    for query in fallback_queries:

        print(
            f"    - {query}",
            flush=True,
        )

    return fallback_queries


# ============================================================
# SERPAPI
# ============================================================

def search_serpapi(query):

    api_key = os.getenv(
        "SERPAPI_API_KEY"
    )

    if not api_key:

        print(
            "  SerpAPI skipped: "
            "SERPAPI_API_KEY missing.",
            flush=True,
        )

        return []

    try:

        session = make_session()

        response = session.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": 8,
                "hl": "en",
                "gl": "us",
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for result in data.get(
            "organic_results",
            [],
        ):

            title = clean_text(
                result.get("title")
            )

            snippet = clean_text(
                result.get("snippet")
            )

            link = clean_text(
                result.get("link")
            )

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

        print(
            f"  SerpAPI ✓ {query} -> "
            f"{len(results)} results",
            flush=True,
        )

        return results

    except requests.Timeout:

        print(
            f"  SerpAPI ⏱ TIMEOUT after "
            f"{REQUEST_TIMEOUT}s -> {query}",
            flush=True,
        )

        return []

    except Exception as exc:

        print(
            f"  SerpAPI ✗ {query}: {exc}",
            flush=True,
        )

        return []


def collect_serpapi_results(
    queries
):

    print(
        "[3/5] Running autonomous web "
        "research concurrently...",
        flush=True,
    )

    all_results = []

    with ThreadPoolExecutor(
        max_workers=len(queries)
    ) as executor:

        futures = {
            executor.submit(
                search_serpapi,
                query,
            ): query
            for query in queries
        }

        for future in as_completed(
            futures
        ):

            query = futures[future]

            try:

                results = future.result()

                all_results.extend(
                    results
                )

            except Exception as exc:

                print(
                    f"  SerpAPI worker crashed -> "
                    f"{query}: {exc}",
                    flush=True,
                )

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

        deduplicated.append(
            result
        )

    print(
        f"  Total unique web results: "
        f"{len(deduplicated)}",
        flush=True,
    )

    return deduplicated


# ============================================================
# ALPHA VANTAGE
# ============================================================

def collect_alpha_vantage_news():

    print(
        "[4/5] Collecting Alpha Vantage "
        "market news...",
        flush=True,
    )

    api_key = os.getenv(
        "ALPHA_VANTAGE_API_KEY"
    )

    if not api_key:

        print(
            "  ALPHA_VANTAGE_API_KEY "
            "not configured.",
            flush=True,
        )

        return []

    try:

        session = make_session()

        response = session.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": (
                    "financial_markets,"
                    "economy_fiscal,"
                    "economy_monetary,"
                    "technology"
                ),
                "limit": 20,
                "apikey": api_key,
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        feed = data.get(
            "feed",
            []
        )

        results = []

        for item in feed:

            title = clean_text(
                item.get("title")
            )

            summary = clean_text(
                item.get("summary")
            )

            url = clean_text(
                item.get("url")
            )

            if not title:
                continue

            results.append(
                {
                    "title": title,
                    "summary": truncate_text(
                        summary,
                        1200,
                    ),
                    "link": url,
                    "source": "Alpha Vantage",
                }
            )

        print(
            f"  Alpha Vantage ✓ "
            f"{len(results)} articles",
            flush=True,
        )

        return results

    except requests.Timeout:

        print(
            "  Alpha Vantage ⏱ TIMEOUT "
            f"after {REQUEST_TIMEOUT}s",
            flush=True,
        )

        return []

    except Exception as exc:

        print(
            f"  Alpha Vantage ✗ {exc}",
            flush=True,
        )

        return []


# ============================================================
# MODEL ORDER
# ============================================================

def build_model_order():

    randomized = [
        model
        for model in FALLBACK_MODELS
        if model != DUMBEST_LAST_MODEL
    ]

    random.shuffle(
        randomized
    )

    return (
        PRIMARY_MODELS
        + randomized
        + [
            DUMBEST_LAST_MODEL
        ]
    )


# ============================================================
# MODEL PAYLOAD
# ============================================================

def model_payload(
    model,
    prompt,
):

    payload = {
        "model": model,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial "
                    "research analyst. "
                    "Produce accurate, concise, "
                    "evidence-grounded analysis. "
                    "Do not invent facts, prices, "
                    "events, or sources."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": 0.2,
        "top_p": 0.95,

        # We do not need a huge answer.
        "max_tokens": 1800,

        "stream": False,
    }


    # --------------------------------------------------------
    # DISABLE / MINIMIZE THINKING
    # --------------------------------------------------------

    if model in {
        "z-ai/glm-5-3-flash",
        "z-ai/glm-5-3",
    }:

        # GLM's minimum reasoning level.
        payload[
            "reasoning_effort"
        ] = "low"


    elif model == (
        "deepseek-ai/"
        "deepseek-v4.1-flash"
    ):

        payload[
            "reasoning_effort"
        ] = "none"


    elif model in {
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
    }:

        payload[
            "chat_template_kwargs"
        ] = {
            "enable_thinking": False,
            "force_nonempty_content": True,
        }


    # --------------------------------------------------------
    # GPT-OSS / OTHER NON-THINKING MODELS
    # --------------------------------------------------------
    #
    # We deliberately do not add reasoning parameters to
    # models that do not need them.

    return payload


# ============================================================
# NVIDIA CALL
# ============================================================

def call_nvidia_model(
    model,
    prompt,
):

    print(
        f"\n  >>> Trying model: {model}",
        flush=True,
    )

    payload = model_payload(
        model,
        prompt,
    )

    started = time.time()

    try:

        response = requests.post(
            NVIDIA_CHAT_URL,

            headers={
                "Authorization": (
                    f"Bearer "
                    f"{NVIDIA_NIM_API_KEY}"
                ),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },

            json=payload,

            timeout=REQUEST_TIMEOUT,
        )

        elapsed = (
            time.time()
            - started
        )

        print(
            f"      HTTP status: "
            f"{response.status_code} "
            f"({elapsed:.2f}s)",
            flush=True,
        )

        if not response.ok:

            error_body = (
                response.text[:2000]
            )

            print(
                "      NVIDIA API error:",
                flush=True,
            )

            print(
                error_body,
                flush=True,
            )

            raise RuntimeError(
                "NVIDIA returned HTTP "
                f"{response.status_code}: "
                f"{error_body}"
            )

        try:

            data = response.json()

        except ValueError as exc:

            raise RuntimeError(
                "NVIDIA returned invalid JSON: "
                f"{exc}"
            )

        text = extract_response_text(
            data
        )

        if not text:

            # THIS IS IMPORTANT.
            # Instead of silently moving on,
            # show the response structure.
            print(
                "      ⚠ HTTP 200 but no "
                "extractable text.",
                flush=True,
            )

            print(
                "      Response keys: "
                f"{list(data.keys())}",
                flush=True,
            )

            if isinstance(
                data.get("choices"),
                list,
            ) and data["choices"]:

                choice = data[
                    "choices"
                ][0]

                print(
                    "      Choice keys: "
                    f"{list(choice.keys())}",
                    flush=True,
                )

                message = choice.get(
                    "message"
                )

                if isinstance(
                    message,
                    dict,
                ):

                    print(
                        "      Message keys: "
                        f"{list(message.keys())}",
                        flush=True,
                    )

            raise RuntimeError(
                "NVIDIA returned a successful "
                "response but no text content."
            )

        print(
            f"      ✓ {model} succeeded "
            f"({len(text)} chars, "
            f"{elapsed:.2f}s)",
            flush=True,
        )

        return text


    except requests.Timeout:

        raise RuntimeError(
            f"{model} timed out after "
            f"{REQUEST_TIMEOUT} seconds."
        )


    except requests.RequestException as exc:

        raise RuntimeError(
            f"{model} network error: "
            f"{exc}"
        )


# ============================================================
# FINANCIAL SYNTHESIS
# ============================================================

def generate_daily_summary(
    rss_articles,
    web_results,
    market_news,
):

    print(
        "\n[5/5] Generating financial "
        "intelligence brief...",
        flush=True,
    )

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

        combined_sources.append(
            result
        )

    for result in market_news:

        combined_sources.append(
            result
        )


    # Keep prompt manageable.
    combined_sources = (
        combined_sources[:100]
    )


    source_text_parts = []

    for index, item in enumerate(
        combined_sources,
        start=1,
    ):

        source_text_parts.append(
            f"""
SOURCE {index}
Title: {item.get('title', '')}
Source: {item.get('source', '')}
Summary: {truncate_text(
    item.get('summary', ''),
    900,
)}
URL: {item.get('link', '')}
""".strip()
        )


    source_text = "\n\n".join(
        source_text_parts
    )


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
- Do not invent events.
- Do not invent sources.
- If sources disagree, say so.
- Prefer recent developments.
- Avoid generic filler.
- Be concise.
- Use readable headings and bullet points.
- Include "Key Takeaways".
- Include "What To Watch Next".
- Do not provide personalized financial advice.
- Do not tell the reader to buy or sell a specific security.

Return ONLY the finished financial brief.

Research material:

{source_text}
"""


    model_order = build_model_order()

    print(
        "\n  MODEL FALLBACK ORDER:",
        flush=True,
    )

    for position, model in enumerate(
        model_order,
        start=1,
    ):

        print(
            f"    {position}. {model}",
            flush=True,
        )


    failures = []


    for model in model_order:

        try:

            summary = call_nvidia_model(
                model,
                prompt,
            )

            if (
                summary
                and len(
                    summary.strip()
                ) >= 100
            ):

                print(
                    "\n  ✓ Daily brief generated "
                    f"successfully using {model}",
                    flush=True,
                )

                return (
                    summary.strip(),
                    model,
                )


            failure = (
                f"{model}: response "
                "was too short"
            )

            print(
                f"      ✗ {failure}",
                flush=True,
            )

            failures.append(
                failure
            )


        except Exception as exc:

            failure = (
                f"{model}: {exc}"
            )

            print(
                f"      ✗ {failure}",
                flush=True,
            )

            failures.append(
                failure
            )


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

def post_discord_chunk(
    webhook_url,
    chunk,
):

    response = requests.post(
        webhook_url,
        json={
            "content": chunk
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()


def send_to_discord(
    summary,
    model_used,
):

    webhook_url = os.getenv(
        "DISCORD_WEBHOOK_URL"
    )

    if not webhook_url:

        print(
            "DISCORD_WEBHOOK_URL is not configured.",
            flush=True,
        )

        return False


    content = (
        "**🤖 Autonomous Financial AI Brief**\n"
        f"*Generated by `{model_used}`*\n\n"
        f"{summary}"
    )


    chunks = [
        content[i:i + 1900]
        for i in range(
            0,
            len(content),
            1900,
        )
    ]


    try:

        # Keep Discord chunks ordered.
        for chunk in chunks:

            post_discord_chunk(
                webhook_url,
                chunk,
            )

            time.sleep(
                0.5
            )


        print(
            "✓ Discord delivery successful.",
            flush=True,
        )

        return True


    except Exception as exc:

        print(
            f"Discord delivery failed: "
            f"{exc}",
            flush=True,
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "AUTONOMOUS FINANCIAL AI DISPATCH",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    start_time = time.time()


    # --------------------------------------------------------
    # 1. RSS
    # --------------------------------------------------------

    rss_articles = (
        collect_rss_articles()
    )


    # --------------------------------------------------------
    # 2. COHERE
    # --------------------------------------------------------

    queries = (
        autonomous_agent_search(
            rss_articles
        )
    )


    # --------------------------------------------------------
    # 3. SERPAPI
    # --------------------------------------------------------

    web_results = (
        collect_serpapi_results(
            queries
        )
    )


    # --------------------------------------------------------
    # 4. ALPHA VANTAGE
    # --------------------------------------------------------

    market_news = (
        collect_alpha_vantage_news()
    )


    # --------------------------------------------------------
    # 5. NVIDIA
    # --------------------------------------------------------

    summary, model_used = (
        generate_daily_summary(
            rss_articles=rss_articles,
            web_results=web_results,
            market_news=market_news,
        )
    )


    # --------------------------------------------------------
    # 6. DISCORD
    # --------------------------------------------------------

    if not send_to_discord(
        summary,
        model_used,
    ):

        raise RuntimeError(
            "The financial brief was generated "
            "successfully, but Discord delivery failed."
        )


    elapsed = (
        time.time()
        - start_time
    )


    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "PIPELINE COMPLETE",
        flush=True,
    )

    print(
        f"Model used: {model_used}",
        flush=True,
    )

    print(
        f"Runtime: {elapsed:.1f} seconds",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nPipeline interrupted.",
            flush=True,
        )

        raise

    except Exception as exc:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "PIPELINE FAILED",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            str(exc),
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        raise
