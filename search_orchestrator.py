import os
import time
import requests
import feedparser
import cohere
from openai import OpenAI


# ============================================================
# 1. Initialize AI Clients
# ============================================================

# NVIDIA NIM — used for final financial synthesis with Kimi
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_NIM_API_KEY")
)

# Cohere — used for autonomous research/query generation
cohere_client = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)


# ============================================================
# 2. Embedded RSS Feeds
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
    "https://www.wired.com/feed/rss"
]


# ============================================================
# 3. RSS Collection
# ============================================================

def fetch_rss_data():
    collected_rss = []

    for url in RSS_FEEDS:
        try:
            res = requests.get(url, timeout=5)

            if res.status_code == 200:
                parsed = feedparser.parse(res.content)
                entries = getattr(parsed, "entries", []) or []

                for entry in entries[:3]:
                    collected_rss.append({
                        "source": url,
                        "title": entry.get("title", "No Title"),
                        "link": entry.get("link", "")
                    })

        except Exception as e:
            print(f"Warning: RSS fetch failed for {url}: {e}")

    return collected_rss


# ============================================================
# 4. Autonomous Research Agent
#    Cohere generates targeted SerpAPI queries
# ============================================================

def autonomous_agent_search(
    topic="global financial and tech market trends"
):
    agent_prompt = f"""
You are an autonomous financial research agent.

Based on the topic:

"{topic}"

Generate exactly 3 highly targeted search queries that will find
the latest and most important financial, economic, technology,
stock-market, AI, and business intelligence.

The queries will be sent directly to a web search engine.

Requirements:
- Make each query specific and information-dense.
- Prioritize very recent developments.
- Avoid generic queries.
- Cover different aspects of the topic.
- Return ONLY the 3 queries separated by commas.
- Do not number them.
- Do not explain them.
"""

    raw_content = None

    for attempt in range(3):
        try:
            response = cohere_client.chat(
                model="command-a-plus-05-2026",
                messages=[
                    {
                        "role": "user",
                        "content": agent_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )

            if response and getattr(response, "message", None):
                content_blocks = (
                    getattr(response.message, "content", None) or []
                )

                for block in content_blocks:
                    if (
                        getattr(block, "type", None) == "text"
                        and getattr(block, "text", None)
                    ):
                        raw_content = block.text
                        break

                if raw_content:
                    break

        except Exception as e:
            print(
                f"Cohere attempt {attempt + 1} failed: {e}"
            )

            if attempt < 2:
                time.sleep(2)

    # --------------------------------------------------------
    # Fallback if Cohere is unavailable
    # --------------------------------------------------------

    if not raw_content:
        print(
            "Warning: Cohere returned empty content after retries. "
            "Using fallback queries."
        )

        queries = [
            "latest financial market trends",
            "latest technology and AI stock market news",
            "latest global economic developments"
        ]

    else:
        queries = [
            q.strip()
            for q in raw_content.split(",")
            if q.strip()
        ]

        # Safety fallback if the model returned something weird
        if len(queries) < 3:
            queries = [
                "latest financial market trends",
                "latest technology and AI stock market news",
                "latest global economic developments"
            ]

        # Only use the first three
        queries = queries[:3]

    # --------------------------------------------------------
    # SerpAPI
    # --------------------------------------------------------

    serpapi_results = []
    serp_key = os.getenv("SERPAPI_API_KEY")

    if serp_key:
        for q in queries:
            try:
                res = requests.get(
                    "https://serpapi.com/search.json",
                    params={
                        "q": q,
                        "api_key": serp_key
                    },
                    timeout=5
                )

                if res.status_code == 200:
                    data = res.json()

                    organic = (
                        data.get("organic_results")
                        if isinstance(data, dict)
                        else None
                    ) or []

                    for item in organic[:3]:
                        serpapi_results.append({
                            "query": q,
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", "")
                        })

                else:
                    print(
                        f"SerpAPI query '{q}' returned "
                        f"HTTP {res.status_code}"
                    )

            except Exception as e:
                print(
                    f"SerpAPI query '{q}' failed: {e}"
                )

    else:
        print(
            "Warning: SERPAPI_API_KEY is not configured. "
            "Skipping web search."
        )

    return serpapi_results


# ============================================================
# 5. Alpha Vantage Financial Data
# ============================================================

def fetch_alpha_vantage():
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not av_key:
        return []

    try:
        res = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "technology,financial_markets",
                "apikey": av_key
            },
            timeout=5
        )

        if res.status_code == 200:
            data = res.json()

            feed = (
                data.get("feed")
                if isinstance(data, dict)
                else None
            ) or []

            return feed[:5]

    except Exception as e:
        print(
            f"Alpha Vantage fetch failed: {e}"
        )

    return []


# ============================================================
# 6. Kimi Financial Synthesis
# ============================================================

def synthesize_with_Kimi(all_data):
    prompt = f"""
You are an elite financial strategist and research analyst.

Analyze, cross-verify, and synthesize the multi-source data below.

Your task is to construct a comprehensive, polished daily financial
and technology intelligence briefing.

You should:

- Identify the most important developments.
- Eliminate duplicate stories.
- Resolve conflicting information where possible.
- Distinguish facts from speculation.
- Highlight important financial-market implications.
- Highlight major technology and AI developments.
- Identify notable companies, sectors, and economic trends.
- Prioritize recent and consequential information.
- Avoid inventing facts that are not present in the supplied data.
- Produce a polished approximately 5-minute daily read.

AGGREGATED SOURCE DATA:

{all_data}
"""

    for attempt in range(3):
        try:
            completion = nim_client.chat.completions.create(
                model="moonshotai/kimi-k3",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )

            if completion and getattr(
                completion,
                "choices",
                None
            ):
                choice = (
                    completion.choices[0]
                    if len(completion.choices) > 0
                    else None
                )

                if (
                    choice
                    and getattr(choice, "message", None)
                    and choice.message.content
                ):
                    return choice.message.content

        except Exception as e:
            print(
                f"Kimi attempt {attempt + 1} failed: {e}"
            )

            if attempt < 2:
                time.sleep(2)

    return (
        "Error: Unable to generate daily summary from "
        "Kimi at this time."
    )


# ============================================================
# 7. Discord Output
# ============================================================

def post_to_discord(brief):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url or not brief:
        return

    # Discord has a 2000-character message limit.
    # Keep a little room below the hard limit.
    chunks = [
        brief[i:i + 1900]
        for i in range(0, len(brief), 1900)
    ]

    for chunk in chunks:
        try:
            response = requests.post(
                webhook_url,
                json={
                    "content": chunk
                },
                timeout=5
            )

            if response.status_code >= 400:
                print(
                    "Discord webhook returned "
                    f"HTTP {response.status_code}"
                )

            time.sleep(1)

        except Exception as e:
            print(
                f"Failed to post chunk to Discord: {e}"
            )


# ============================================================
# 8. Main Pipeline
# ============================================================

if __name__ == "__main__":
    print(
        "=================================================="
    )
    print("AUTONOMOUS FINANCIAL AI DISPATCH")
    print(
        "=================================================="
    )

    print(
        "\n[1/5] Gathering data from RSS feeds..."
    )
    rss_data = fetch_rss_data()
    print(
        f"Collected {len(rss_data)} RSS articles."
    )

    print(
        "\n[2/5] Running autonomous research "
        "queries via Cohere..."
    )
    search_data = autonomous_agent_search()
    print(
        f"Collected {len(search_data)} web search results."
    )

    print(
        "\n[3/5] Fetching Alpha Vantage financial data..."
    )
    av_data = fetch_alpha_vantage()
    print(
        f"Collected {len(av_data)} financial API results."
    )

    master_payload = {
        "rss_streams": rss_data,
        "autonomous_web_search": search_data,
        "financial_apis": av_data
    }

    print(
        "\n[4/5] Synthesizing brief with Kimi..."
    )
    final_brief = synthesize_with_Kimi(
        master_payload
    )

    print(
        "\n[5/5] Pushing brief to Discord..."
    )
    post_to_discord(final_brief)

    print(
        "\n=================================================="
    )
    print("Pipeline completed successfully!")
    print(
        "=================================================="
    )
