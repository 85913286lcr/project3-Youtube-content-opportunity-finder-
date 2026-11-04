"""Project settings shared by the scripts, notebooks and the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_LABELS = ROOT / "data" / "labels"
FIGURES = ROOT / "reports" / "figures"

for p in (DATA_RAW, DATA_PROCESSED, DATA_LABELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

# Search queries used to build each niche. A video belongs to the niche of the
# query that found it (first match wins if it shows up under several queries).
NICHES = {
    "AI Career": [
        "AI product manager",
        "AI career change",
        "how to get a job in AI",
        "AI skills to learn",
        "machine learning engineer roadmap",
    ],
    "AI Tools": [
        "best AI tools",
        "free AI tools",
        "ChatGPT tips and tricks",
        "AI tools for productivity",
        "ChatGPT vs Claude vs Gemini",
    ],
    "AI Agents": [
        "AI agents tutorial",
        "build an AI agent",
        "n8n AI automation",
        "AI agents explained",
        "MCP server tutorial",
    ],
}

# How far back to search for videos. 120 days gives a 90 day research window
# plus room for the time-based backtest.
LOOKBACK_DAYS = 120
SEARCH_PAGES_PER_QUERY = 2      # 50 results per page
COMMENT_PAGES_PER_VIDEO = 3     # 100 top-level comments per page

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
