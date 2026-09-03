import os
import requests
import feedparser
from openai import OpenAI

# 1. Initialize Clients
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_NIM_API_KEY")
)

opencode_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_API_KEY")
)

# 2. Embedded RSS Feeds (extracted from OPML)
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

def fetch_rss_data():
    collected_rss = []
    for url in RSS_FEEDS:
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                parsed = feedparser.parse(res.content)
                for entry in parsed.entries[:3]:
                    collected_rss.append({"source": url, "title": entry.get("title", ""), "link": entry.get("link", "")})
        except Exception as e:
            print(f"Error fetching RSS {url}: {e}")
    return collected_rss

def autonomous_agent_search(topic="global financial and tech market trends"):
    # Use Nemotron to generate precise custom search requests
    agent_prompt = f"""
    You are an autonomous research agent. Based on the topic '{topic}', generate 3 targeted search queries 
    optimized for Serpapi to find the absolute latest intelligence. Return only the queries separated by commas.
    """
    
    response = opencode_client.chat.completions.create(
        model="nemotron-3-ultra-free",
        messages=[{"role": "user", "content": agent_prompt}],
        temperature=0.3
    )
    queries = [q.strip() for q in response.choices[0].message.content.split(",")]
    
    serpapi_results = []
    serp_key = os.getenv("SERPAPI_API_KEY")
    
    for q in queries:
        try:
            res = requests.get(f"https://serpapi.com/search.json?q={q}&api_key={serp_key}", timeout=5)
            if res.status_code == 200:
                organic = res.json().get("organic_results", [])[:3]
                for item in organic:
                    serpapi_results.append({
                        "query": q,
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "link": item.get("link")
                    })
        except Exception as e:
            print(f"Serpapi search failed for query '{q}': {e}")
            
    return serpapi_results

def fetch_alpha_vantage():
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_key:
        return []
    try:
        res = requests.get(f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=technology,financial_markets&apikey={av_key}", timeout=5)
        if res.status_code == 200:
            return res.json().get("feed", [])[:5]
    except Exception as e:
        print(f"Alpha Vantage fetch failed: {e}")
    return []

def synthesize_with_deepseek(all_data):
    prompt = f"""
    You are an elite financial strategist. Analyze, cross-verify, and synthesize the massive multi-source data below 
    (combining live search intelligence, structured financial feeds, and global RSS streams). 
    Eliminate duplicate articles, resolve conflicting details, and construct a comprehensive, polished 5-minute daily read.

    AGGREGATED SOURCE DATA:
    {all_data}
    """
    
    completion = nim_client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-pro-0813",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    return completion.choices[0].message.content

def post_to_discord(brief):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        requests.post(webhook_url, json={"content": brief})

if __name__ == "__main__":
    print("Gathering data from RSS feeds...")
    rss_data = fetch_rss_data()
    
    print("Running autonomous agent queries via Nemotron and Serpapi...")
    search_data = autonomous_agent_search()
    
    print("Fetching Alpha Vantage financial data...")
    av_data = fetch_alpha_vantage()
    
    master_payload = {
        "rss_streams": rss_data,
        "autonomous_web_search": search_data,
        "financial_apis": av_data
    }
    
    print("Synthesizing brief with DeepSeek-V4-Pro-0813...")
    final_brief = synthesize_with_deepseek(master_payload)
    
    print("Pushing brief to Discord...")
    post_to_discord(final_brief)
