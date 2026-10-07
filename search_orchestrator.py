```python
import os
import time
import requests
import feedparser
from openai import OpenAI


# ============================================================
# API CLIENTS
# ============================================================

# NVIDIA NIM
# Used for the final financial briefing with GLM-5.3-Flash.
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_NIM_API_KEY")
)

# Cohere OpenAI-compatible API
# Used for autonomous research/query generation.
cohere_client = OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key=os.getenv("COHERE_API_KEY")
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
# RSS COLLECTION
# ============================================================

def collect_rss_news():
    print("Collecting RSS news...")

    articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:10]:
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                summary = getattr(entry, "summary", "").strip()

                if not title:
                    continue

                articles.append({
                    "title": title,
                    "link": link,
                    "summary": summary
                })

        except Exception as e:
            print(f"RSS error for {feed_url}: {e}")

    return articles


# ============================================================
# COHERE AUTONOMOUS RESEARCH
# ============================================================

def autonomous_agent_search():
    """
    Ask Cohere Command A+ to determine which financial/technology
    topics should be searched next.

    Cohere is accessed through its OpenAI-compatible API, so no
    separate Cohere SDK is required.
    """

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        print("WARNING: COHERE_API_KEY is not configured.")
        print("Using fallback search queries.")

        return [
            "latest financial market trends",
            "latest technology and AI stock market news",
            "latest global economic developments"
        ]

    agent_prompt = """
You are the autonomous research planner for a daily financial news brief.

Determine the three most useful web search queries for finding
important developments in:

1. Global financial markets
2. Technology and AI companies
3. Macroeconomics and major economic events

Return EXACTLY three search queries.

Return ONLY the queries, one per line.
Do not number them.
Do not explain them.
Do not use quotation marks.
"""

    for attempt in range(3):
        try:
            print(f"  Asking Cohere for research queries (attempt {attempt + 1}/3)...")

            response = cohere_client.chat.completions.create(
                model="command-a-plus-05-2026",
                messages=[
                    {
                        "role": "user",
                        "content": agent_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Cohere returned empty content.")

            queries = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

            # Remove accidental numbering/bullets.
            cleaned_queries = []

            for query in queries:
                query = query.lstrip("-•* ")
                query = query.lstrip("0123456789")
                query = query.lstrip(". ")
                query = query.strip()

                if query:
                    cleaned_queries.append(query)

            if len(cleaned_queries) >= 3:
                return cleaned_queries[:3]

            raise ValueError(
                f"Cohere returned fewer than 3 usable queries: {cleaned_queries}"
            )

        except Exception as e:
            print(f"  Cohere error: {e}")

            if attempt < 2:
                time.sleep(2)

    print("Cohere failed after 3 attempts. Using fallback queries.")

    return [
        "latest financial market trends",
        "latest technology and AI stock market news",
        "latest global economic developments"
    ]


# ============================================================
# SERPAPI SEARCH
# ============================================================

def search_web(query):
    """
    Search the web through SerpAPI.
    """

    serp_key = os.getenv("SERPAPI_API_KEY")

    if not serp_key:
        print("WARNING: SERPAPI_API_KEY is not configured.")
        return []

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "api_key": serp_key,
                "engine": "google"
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("organic_results", [])[:8]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })

        return results

    except Exception as e:
        print(f"SerpAPI error for '{query}': {e}")
        return []


# ============================================================
# ALPHA VANTAGE
# ============================================================

def collect_market_news():
    """
    Collect financial/technology news from Alpha Vantage.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        print("WARNING: ALPHA_VANTAGE_API_KEY is not configured.")
        return []

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "technology,financial_markets",
                "apikey": api_key
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return data.get("feed", [])[:20]

    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return []


# ============================================================
# KIMI / GLM REPLACEMENT:
# GLM-5.3-FLASH FINAL SYNTHESIS
# ============================================================

def generate_daily_summary(rss_articles, search_results, market_news):
    """
    Synthesize all collected information into the final daily brief.

    GLM-5.3-Flash is used with MAX reasoning effort.

    NVIDIA documents max as the largest reasoning budget and the
    default for GLM-5.3-Flash.
    """

    if not os.getenv("NVIDIA_NIM_API_KEY"):
        return "Error: NVIDIA_NIM_API_KEY is not configured."

    # --------------------------------------------------------
    # Build source material
    # --------------------------------------------------------

    source_sections = []

    # RSS
    if rss_articles:
        rss_text = "\n".join(
            f"- {article['title']}\n"
            f"  {article['summary'][:500]}\n"
            f"  URL: {article['link']}"
            for article in rss_articles[:30]
        )

        source_sections.append(
            "RSS NEWS:\n" + rss_text
        )

    # Web search
    if search_results:
        web_text = "\n".join(
            f"- {result['title']}\n"
            f"  {result['snippet'][:500]}\n"
            f"  URL: {result['link']}"
            for result in search_results[:30]
        )

        source_sections.append(
            "WEB SEARCH RESULTS:\n" + web_text
        )

    # Alpha Vantage
    if market_news:
        market_text_parts = []

        for item in market_news[:20]:
            title = item.get("title", "")
            summary = item.get("summary", "")
            url = item.get("url", "")

            if title:
                market_text_parts.append(
                    f"- {title}\n"
                    f"  {summary[:500]}\n"
                    f"  URL: {url}"
                )

        if market_text_parts:
            source_sections.append(
                "ALPHA VANTAGE MARKET NEWS:\n"
                + "\n".join(market_text_parts)
            )

    source_material = "\n\n".join(source_sections)

    if not source_material:
        return "Error: No financial news data was collected."

    # Prevent accidentally enormous requests.
    source_material = source_material[:50000]

    # --------------------------------------------------------
    # Final synthesis prompt
    # --------------------------------------------------------

    prompt = f"""
You are the final analyst for an autonomous daily financial intelligence
system.

Analyze the collected information below and produce a concise,
high-quality financial briefing.

IMPORTANT:
- Do not invent facts.
- Do not make up prices, percentages, earnings, dates, or events.
- Distinguish confirmed information from speculation.
- Prefer information supported by multiple sources.
- Identify contradictory reports when relevant.
- Focus on developments that actually matter.
- Explain why each major development matters.
- Do not give personalized investment advice.
- Do not tell the reader to buy or sell a specific security.

Structure the briefing as:

# Daily Financial Intelligence Brief

## 🔥 Top Developments
3-5 most important developments.

## 📈 Markets
Major market-moving developments.

## 🤖 Technology & AI
Important technology, AI, semiconductor, and major-company developments.

## 🌎 Macro & Economy
Important economic, geopolitical, central-bank, inflation, employment,
trade, or other macroeconomic developments.

## 👀 What To Watch
Important developments to monitor next.

Keep the final answer readable and reasonably concise.

COLLECTED INFORMATION:

{source_material}
"""

    try:
        print("Synthesizing brief with GLM-5.3-Flash...")

        completion = nim_client.chat.completions.create(
            model="z-ai/glm-5-3-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=2500,
            reasoning_effort="max"
        )

        result = completion.choices[0].message.content

        if not result:
            return "Error: GLM-5.3-Flash returned an empty response."

        return result.strip()

    except Exception as e:
        print(f"GLM-5.3-Flash error: {e}")
        return "Error: Unable to generate daily summary from GLM-5.3-Flash at this time."


# ============================================================
# DISCORD
# ============================================================

def send_to_discord(message):
    """
    Send the final briefing to Discord.

    Discord webhooks have a message length limit, so long reports
    are split into multiple messages.
    """

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("WARNING: DISCORD_WEBHOOK_URL is not configured.")
        return False

    chunks = [
        message[i:i + 1900]
        for i in range(0, len(message), 1900)
    ]

    try:
        for chunk in chunks:
            response = requests.post(
                webhook_url,
                json={
                    "content": chunk
                },
                timeout=15
            )

            response.raise_for_status()

        return True

    except Exception as e:
        print(f"Discord error: {e}")
        return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("AUTONOMOUS FINANCIAL AI DISPATCH")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. RSS
    # --------------------------------------------------------

    print("\n[1/5] Collecting RSS news...")
    rss_articles = collect_rss_news()
    print(f"Collected {len(rss_articles)} RSS articles.")

    # --------------------------------------------------------
    # 2. Cohere research planning
    # --------------------------------------------------------

    print("\n[2/5] Generating autonomous research queries with Cohere...")
    queries = autonomous_agent_search()

    print("Research queries:")
    for query in queries:
        print(f"  - {query}")

    # --------------------------------------------------------
    # 3. Web research
    # --------------------------------------------------------

    print("\n[3/5] Searching the web...")

    search_results = []

    for query in queries:
        print(f"  Searching: {query}")

        results = search_web(query)
        search_results.extend(results)

        # Small delay to avoid unnecessarily hammering APIs.
        time.sleep(0.5)

    print(f"Collected {len(search_results)} web results.")

    # --------------------------------------------------------
    # 4. Market data
    # --------------------------------------------------------

    print("\n[4/5] Collecting Alpha Vantage market news...")

    market_news = collect_market_news()

    print(f"Collected {len(market_news)} market-news items.")

    # --------------------------------------------------------
    # 5. Final synthesis
    # --------------------------------------------------------

    print("\n[5/5] Generating final briefing with GLM-5.3-Flash...")
    print("Reasoning effort: MAX")

    summary = generate_daily_summary(
        rss_articles=rss_articles,
        search_results=search_results,
        market_news=market_news
    )

    print("\n" + "=" * 60)
    print("FINAL BRIEF")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    # --------------------------------------------------------
    # Discord
    # --------------------------------------------------------

    print("\nSending briefing to Discord...")

    if send_to_discord(summary):
        print("Discord delivery successful.")
    else:
        print("Discord delivery failed.")

    print("\nPipeline complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
```
