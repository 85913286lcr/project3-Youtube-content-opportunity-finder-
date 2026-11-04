"""Content Opportunity Finder · public showcase by Congrui Liu.

    streamlit run app_public.py

Reads only the small, anonymised files in ./data (topic names, sizes, scores,
example questions with no author names). No raw comments or user ids.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
DATA = HERE / "data"
RED, INK, MUTED = "#E5402A", "#13151B", "#80868F"
LINES = ["AI Career", "AI Tools", "AI Agents"]
LINE_EMOJI = {"AI Career": "🎓", "AI Tools": "🛠️", "AI Agents": "🤖"}

st.set_page_config(page_title="Content Opportunity Finder · Congrui Liu", page_icon="▶️", layout="wide")


@st.cache_data
def load():
    t = pd.read_csv(DATA / "topics_public.csv")
    p = pd.read_csv(DATA / "picks_public.csv")
    m = json.loads((DATA / "meta.json").read_text())
    return t, p, m


topics, picks, meta = load()

st.markdown(f"""
<style>
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 1.2rem; max-width: 1200px;}}
.hero {{background:#0F0F0F; color:white; border-radius:18px; padding:28px 32px 24px;}}
.hero h1 {{color:white; font-size:2.4rem; margin:0; padding:0; display:flex; align-items:center; gap:14px; line-height:1.2;}}
.yt {{background:{RED}; border-radius:12px; width:58px; height:40px; display:inline-flex; align-items:center; justify-content:center; font-size:20px;}}
.hero .sub {{color:#bbb; font-size:1.1rem; margin:6px 0 0 0;}}
.hero .by {{color:#ddd; font-size:.95rem; margin:4px 0 0 0;}}
.chips {{display:flex; flex-wrap:wrap; gap:10px; margin-top:20px; align-items:center;}}
.chip {{background:#272727; border-radius:12px; padding:10px 14px; color:white;}}
.chip b {{font-size:1.25rem;}} .chip span {{color:#aaa; font-size:.85rem;}}
.arrow {{color:{RED}; font-weight:700;}}
.result {{background:#1d1d1d; border-left:4px solid {RED}; border-radius:10px; padding:10px 14px; margin-top:16px; color:#ddd;}}
.result b {{color:#FF6B57; font-size:1.15rem;}}
.card {{background:white; border:1px solid #E2E5EA; border-radius:16px; padding:18px; height:100%; box-shadow:0 2px 10px rgba(20,22,29,.06);}}
.badge {{display:inline-block; background:#FDECE9; color:{RED}; border-radius:999px; padding:3px 10px; font-weight:700; font-size:.8rem;}}
.card h3 {{font-size:1.15rem; margin:10px 0 6px; line-height:1.3;}}
.from {{color:{MUTED}; font-size:.85rem;}}
.score {{font-size:2rem; font-weight:800; color:{RED};}}
.bar {{background:#EEF0F3; border-radius:6px; height:8px; margin:2px 0 8px;}}
.bar div {{background:#454B57; height:8px; border-radius:6px;}}
.quote {{background:#F6F7F9; border-radius:10px; padding:10px 12px; font-style:italic; color:#454B57; font-size:.9rem; margin-top:8px;}}
.sig {{background:white; border:1px solid #E2E5EA; border-radius:14px; padding:14px; text-align:center;}}
.sig .e {{font-size:2rem;}} .sig b {{color:{RED};}}
.op {{background:#FDECE9; border-radius:14px; padding:14px; text-align:center;}}
.step {{background:#EEF0F3; border-radius:14px; padding:14px; height:100%;}}
.step .n {{font-size:1.6rem; font-weight:800;}}
.les {{color:#2E59E6; font-weight:700; font-size:.8rem;}}
.foot {{color:{MUTED}; text-align:center; margin-top:30px; font-size:.9rem;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <h1><span class="yt">▶</span>Content Opportunity Finder</h1>
  <div class="sub">Turns YouTube comments into 3 video ideas a week for AI creators</div>
  <div class="by">by <b>Congrui Liu</b> · NTU Module 3 project · data up to {pd.Timestamp(meta['data_end']):%d %b %Y}</div>
  <div class="chips">
    <div class="chip">📥 <b>{meta['raw']:,}</b> <span>comments</span></div><span class="arrow">→</span>
    <div class="chip">🧹 <b>{meta['clean']:,}</b> <span>clean</span></div><span class="arrow">→</span>
    <div class="chip">🔍 <b>{meta['questions']:,}</b> <span>questions</span></div><span class="arrow">→</span>
    <div class="chip">🧩 <b>{meta['topics']}</b> <span>topics</span></div><span class="arrow">→</span>
    <div class="chip">🏆 <b>3</b> <span>videos / week</span></div>
  </div>
  <div class="result">🎉 Tested on 7 past weeks: <b>7 in 10</b> of the weekly ideas were still hot a month later (random: under 5 in 10)</div>
</div>
""", unsafe_allow_html=True)

st.write("")
tab_ideas, tab_topics, tab_how = st.tabs(["🎬  This week's 3 ideas", "🧩  All 50 topics", "💡  How it works"])

# ---------------------------------------------------------------- this week's ideas
def bar(label, v):
    v = 0 if pd.isna(v) else float(v)
    return f'<div class="from">{label} · rank {v:.0f}</div><div class="bar"><div style="width:{v}%"></div></div>'


with tab_ideas:
    st.subheader("This week's 3 video ideas")
    st.caption("One idea per content line: the best angle inside the highest-scoring topic.")
    cols = st.columns(3)
    for col, line in zip(cols, LINES):
        p = picks[picks["niche"] == line].iloc[0]
        t = topics[topics["topic"] == p["topic"]].iloc[0]
        with col:
            st.markdown(f"""
<div class="card">
  <span class="badge">{LINE_EMOJI[line]} {line}</span>
  <h3>{p['angle_name']}</h3>
  <div class="from">from topic: {t['topic_name']}</div>
  <div style="display:flex;align-items:baseline;gap:8px;margin:10px 0 4px"><span class="score">{t['opportunity']:.0f}</span><span class="from">score · #1 in {line}</span></div>
  {bar('👥 Demand', t['D'])}{bar('📹 Gap', t['G'])}{bar('🔥 Fresh', t['F'])}
  <div class="from">{int(t['n_users'])} people asked in the last 30 days · {t['uncovered_share']:.0%} have no video yet</div>
  <div class="quote">“{p['example']}”</div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- all 50 topics
with tab_topics:
    st.subheader("All 50 topics, found by K-Means")
    c1, c2 = st.columns([2, 1])
    line = c1.segmented_control("Content line", ["All"] + LINES, default="All") or "All"
    q = c2.text_input("Search a topic", placeholder="e.g. Claude, roadmap, MCP")
    d = topics if line == "All" else topics[topics["niche"] == line]
    if q:
        d = d[d["topic_name"].str.contains(q, case=False) | d["keywords"].str.contains(q, case=False)]
    k1, k2, k3 = st.columns(3)
    k1.metric("Topics", len(d))
    k2.metric("Questions in these topics", f"{int(d['n_train'].sum()):,}", help="Of the 2,574 Jul–Sep questions, about 90% fit a topic.")
    k3.metric("Filmable, scored this week", int(d["opportunity"].notna().sum()))

    s = d[d["opportunity"].notna()].copy()
    if len(s):
        s["Top 3"] = s["rank"].notna().map({True: "Top 3 this week", False: "other topics"})
        fig = px.scatter(s, x="G", y="D", size="n_train", color="Top 3", hover_name="topic_name",
                         hover_data={"niche": True, "opportunity": ":.0f", "n_train": True, "G": False, "D": False, "Top 3": False},
                         color_discrete_map={"Top 3 this week": RED, "other topics": "#9AA0A8"}, size_max=38,
                         labels={"G": "Gap rank (fewer videos →)", "D": "Demand rank (more people ↑)", "n_train": "questions"})
        fig.add_shape(type="rect", x0=50, y0=50, x1=102, y1=102, line=dict(color=RED, dash="dash"), fillcolor="rgba(229,64,42,0.05)")
        fig.add_annotation(x=76, y=114, text="🎯 sweet spot: wanted + not yet filmed", showarrow=False, font=dict(color=RED))
        fig.update_layout(height=430, plot_bgcolor="white", legend_title_text="", margin=dict(t=30, l=10, r=10, b=10))
        fig.update_xaxes(range=[-5, 108], gridcolor="#EEF0F3"); fig.update_yaxes(range=[-5, 120], gridcolor="#EEF0F3")
        st.markdown("**🗺️ Opportunity map** · one bubble = one topic · bigger = more questions")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**📋 Topic list** · sorted by this week's score")
    d = d.sort_values(["opportunity", "n_train"], ascending=[False, False], na_position="last")
    for _, r in d.iterrows():
        star = "🏆 " if pd.notna(r["rank"]) else ""
        score = f"score **{r['opportunity']:.0f}**" if pd.notna(r["opportunity"]) else "_not filmable as one video_"
        with st.expander(f"{star}{LINE_EMOJI[r['niche']]} {r['topic_name']}  ·  {int(r['n_train'])} questions"):
            a, b = st.columns([1, 2])
            a.markdown(f"{score}  \nformat: {r['video_format']}  \n" + (f"👥 {r['D']:.0f} · 📹 {r['G']:.0f} · 🔥 {r['F']:.0f}" if pd.notna(r["D"]) else ""))
            b.markdown("**Viewers ask:**  \n" + "  \n".join(f"> {e}" for e in str(r["examples"]).split(" || ") if e and e != "nan"))

# ---------------------------------------------------------------- how it works
with tab_how:
    st.subheader("💡 My idea: a video is worth making when the comments show 3 signals")
    a, p1, b, p2, c, eq, o = st.columns([3, .4, 3, .4, 3, .4, 3])
    a.markdown('<div class="sig"><div class="e">👥</div><b>DEMAND</b><br>Many people ask it<br><span class="from">45%</span></div>', unsafe_allow_html=True)
    p1.markdown("<h2 style='text-align:center'>+</h2>", unsafe_allow_html=True)
    b.markdown('<div class="sig"><div class="e">📹</div><b>GAP</b><br>Few videos on it<br><span class="from">35%</span></div>', unsafe_allow_html=True)
    p2.markdown("<h2 style='text-align:center'>+</h2>", unsafe_allow_html=True)
    c.markdown('<div class="sig"><div class="e">🔥</div><b>FRESHNESS</b><br>Rising this week<br><span class="from">20%</span></div>', unsafe_allow_html=True)
    eq.markdown("<h2 style='text-align:center'>=</h2>", unsafe_allow_html=True)
    o.markdown(f'<div class="op"><div class="e" style="font-size:2rem">🎯</div><b style="color:{RED}">OPPORTUNITY</b><br>Worth filming now</div>', unsafe_allow_html=True)
    st.caption("Gap uses semantic search: a comment and recent video titles become meaning numbers (MiniLM); cosine ≥ 0.60 means a video already answers it.")

    st.subheader("🔬 5 steps from raw comments to 3 videos")
    STEPS = [("📥", "51,447", "comments", "YouTube API · 120 days", ""),
             ("🧹", "25,226", "clean", "English, no spam", "split by time · L06"),
             ("🔍", "3,678", "questions", "Supervised: question or not?", "L03 · L04 · L09"),
             ("🧩", "50", "topics", "Unsupervised: K-Means", "L05 · L09"),
             ("🏆", "3", "videos / week", "Score + back-test", "L06 · L09")]
    for col, (e, n, u, how, les) in zip(st.columns(5), STEPS):
        col.markdown(f'<div class="step">{e} <span class="n">{n}</span> <span class="from">{u}</span><br><b>{how}</b><br><span class="les">{les}</span></div>', unsafe_allow_html=True)

    st.subheader("📊 Does it work?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Our picks still hot a month later", f"{meta['hit_ours']:.0%}", "+23 pts vs random")
    m2.metric("Random picks", f"{meta['hit_random']:.0%}")
    m3.metric("Most-asked picks", f"{meta['hit_most']:.0%}", "but ~half already filmed", delta_color="off")
    st.caption("Back-test on 7 past Mondays (3 Aug – 14 Sep 2026): pick the Top 3 with older data only, then check the next 4 weeks.")

st.markdown('<div class="foot">Built with ❤️ by <b>Congrui Liu</b> · NTU Module 3 · Data: public YouTube comments, anonymised</div>', unsafe_allow_html=True)
