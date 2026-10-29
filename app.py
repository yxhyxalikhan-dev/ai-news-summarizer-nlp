"""
AI-Based News Summarizer Using Natural Language Processing
Author: Khan Yahya Shahid Ali
UIN: 221A048 | Roll No: 28
Department of Artificial Intelligence & Data Science Engineering
Rizvi College of Engineering | University of Mumbai (2026-2027)
"""

import re
import math
from collections import Counter

# Try importing streamlit and pandas; if not present, provide CLI fallback
try:
    import streamlit as st
    import pandas as pd
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# ==========================================
# 1. NLP STOP-WORDS & PRESET ARTICLES
# ==========================================
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now', 'also', 'said',
    'one', 'two', 'new', 'time', 'first', 'even', 'many', 'much', 'like'
}

SAMPLE_ARTICLES = {
    "AI & Technology": (
        "Artificial Intelligence has reached an unprecedented inflection point in recent years, "
        "evolving from passive generative chatbots into proactive autonomous agents. "
        "Modern AI systems can now orchestrate multi-step reasoning, plan software architectures, "
        "and execute complex system commands across digital environments. "
        "Tech leaders emphasize that agentic workflows represent the next frontier of enterprise productivity. "
        "However, researchers caution that safety guardrails, alignment protocols, and hallucination reduction "
        "must remain top engineering priorities. "
        "As organizations integrate autonomous agents into critical business operations, "
        "the need for transparent verification and human-in-the-loop oversight becomes essential. "
        "Ultimately, the fusion of reasoning models and tool-calling execution is transforming how humanity solves intricate global challenges."
    ),
    "Space Exploration": (
        "The James Webb Space Telescope has captured breathtaking new deep-field observations, "
        "revealing ancient galaxies that formed merely a few hundred million years after the Big Bang. "
        "Astronomers were surprised to discover that early galaxies were significantly more massive and brighter "
        "than prevailing theoretical cosmological models previously predicted. "
        "These cosmic structures contain mature stellar populations and complex chemical elements. "
        "Spectral analysis indicates rapid star formation rates driven by supermassive black hole seed mergers. "
        "International astrophysics teams are now recalibrating simulations of cosmic reionization. "
        "This landmark discovery challenges standard galaxy evolutionary timelines and offers profound insights "
        "into the primordial dawn of our universe."
    ),
    "Renewable Energy": (
        "Global investments in clean energy technologies surged past traditional fossil fuels over the past year. "
        "Next-generation solar photovoltaic cells and offshore wind turbine farms achieved record-breaking efficiencies. "
        "Advancements in solid-state grid storage batteries are successfully stabilizing fluctuating power grids. "
        "Developing nations are rapidly adopting decentralized micro-grid systems to electrify remote rural populations. "
        "Economists report that plummeting levelized costs of renewable power are attracting substantial venture capital. "
        "Environmental agencies stress that accelerating clean infrastructure deployment is crucial to meeting Paris Agreement targets. "
        "With strategic international policy cooperation, the worldwide transition towards zero-carbon economies is gaining unstoppable momentum."
    )
}

# ==========================================
# 2. NLP EXTRACTIVE SUMMARIZATION ENGINE
# ==========================================
def split_into_sentences(text: str):
    """Accurately splits raw text into individual sentence strings."""
    # Split on sentence terminals followed by space or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    return sentences

def summarize_text(text: str, summary_ratio: float = 0.4, min_sentences: int = 2):
    """
    Performs Frequency-Weighted Extractive Text Summarization:
    1. Normalization & Tokenization
    2. Word Frequency Weighting (excluding stop-words)
    3. Sentence Salience Scoring (normalized by sentence length)
    4. Top-ranked sentence extraction maintaining chronological document order
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    sentences = split_into_sentences(cleaned_text)
    total_sentences = len(sentences)
    if total_sentences == 0:
        return None

    # Word Tokenization
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())
    total_words = len(words)

    # Filter stop-words and short words
    meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    if not meaningful_words:
        return None

    # Frequency Table
    word_freq = Counter(meaningful_words)
    max_freq = max(word_freq.values())

    # Weighted word frequencies (0.0 to 1.0)
    word_weights = {w: count / max_freq for w, count in word_freq.items()}

    # Score each sentence based on word weights
    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        if not sent_words:
            sentence_scores.append((idx, 0.0, sentence))
            continue

        # Score is sum of word weights divided by squareroot of length (prevents length bias)
        score = sum(word_weights.get(w, 0.0) for w in sent_words)
        score = score / (math.sqrt(len(sent_words)) + 1e-5)
        sentence_scores.append((idx, score, sentence))

    # Determine number of sentences to extract
    target_count = max(min_sentences, int(round(total_sentences * summary_ratio)))
    target_count = min(target_count, total_sentences)

    # Sort by score descending to get top sentences
    ranked = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    selected = ranked[:target_count]

    # Re-sort selected sentences by original index to preserve narrative flow
    selected_in_order = sorted(selected, key=lambda x: x[0])
    summary_sentences = [item[2] for item in selected_in_order]
    summary_text = " ".join(summary_sentences)

    # Metrics
    summary_words = len(re.findall(r'\b[a-zA-Z]+\b', summary_text))
    reduction_pct = int(round((1.0 - (summary_words / max(total_words, 1))) * 100))
    reduction_pct = max(0, min(99, reduction_pct))

    reading_time_orig = max(1, math.ceil(total_words / 200))
    reading_time_summ = max(1, math.ceil(summary_words / 200))
    time_saved = max(0, reading_time_orig - reading_time_summ)

    # Top keywords
    top_keywords = [f"{w} ({c})" for w, c in word_freq.most_common(8)]

    return {
        'original_text': cleaned_text,
        'summary_text': summary_text,
        'summary_sentences': summary_sentences,
        'total_words': total_words,
        'summary_words': summary_words,
        'total_sentences': total_sentences,
        'summary_sentences_count': len(summary_sentences),
        'reduction_pct': reduction_pct,
        'time_saved': time_saved,
        'top_keywords': top_keywords,
        'word_freq': word_freq.most_common(8)
    }

# ==========================================
# 3. STREAMLIT WEB APPLICATION
# ==========================================
def run_streamlit_app():
    st.set_page_config(
        page_title="AI News Summarizer - NLP",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin-bottom: 0px;
        }
        .sub-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #10B981, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: -10px;
            margin-bottom: 8px;
        }
        .tagline {
            color: #94A3B8;
            font-size: 1.05rem;
            margin-bottom: 25px;
        }
        .metric-box {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .metric-num {
            font-size: 2.2rem;
            font-weight: 700;
            color: #10B981;
        }
        .metric-sub {
            font-size: 0.85rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .summary-box {
            background-color: #1E293B;
            border-left: 5px solid #10B981;
            border-radius: 8px;
            padding: 18px;
            color: #F8FAFC;
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .bullet-item {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
            color: #E2E8F0;
            font-size: 0.95rem;
        }
        .disclaimer-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 18px;
            color: #CBD5E1;
            font-size: 0.85rem;
            margin-top: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title & Subtitle
    st.markdown('<div class="main-title">NewsBrief AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Based News & Text Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Extract concise executive summaries, key bullet insights, and document metrics using Natural Language Processing.</div>', unsafe_allow_html=True)

    # Session State
    if 'news_input' not in st.session_state:
        st.session_state.news_input = SAMPLE_ARTICLES["AI & Technology"]

    # Two column workspace
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 01 / Original Article")
        
        # Preset buttons
        st.write("**Load Preset Articles:**")
        bcol1, bcol2, bcol3 = st.columns(3)
        if bcol1.button("🤖 AI & Tech"):
            st.session_state.news_input = SAMPLE_ARTICLES["AI & Technology"]
        if bcol2.button("🚀 Space Discovery"):
            st.session_state.news_input = SAMPLE_ARTICLES["Space Exploration"]
        if bcol3.button("⚡ Clean Energy"):
            st.session_state.news_input = SAMPLE_ARTICLES["Renewable Energy"]

        user_article = st.text_area(
            "Paste news text here",
            value=st.session_state.news_input,
            height=260,
            label_visibility="collapsed"
        )

        # Controls: Compression slider
        compression = st.slider("Summary Ratio (Length)", min_value=20, max_value=80, value=40, step=10, format="%d%%")
        summary_ratio = compression / 100.0

        summarize_btn = st.button("✦ Generate Summary", type="primary", use_container_width=True)
        st.caption(f"{len(user_article)} characters • Processed locally via Frequency-Weighted Extractive NLP.")

    with col2:
        st.markdown("### 02 / AI Summary & Key Takeaways")

        res = summarize_text(user_article, summary_ratio=summary_ratio)
        if res:
            # 4 Metric Cards
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(f'<div class="metric-box"><div class="metric-sub">Original</div><div class="metric-num">{res["total_words"]}</div><div class="metric-sub">Words</div></div>', unsafe_allow_html=True)
            with mcol2:
                st.markdown(f'<div class="metric-box"><div class="metric-sub">Summary</div><div class="metric-num">{res["summary_words"]}</div><div class="metric-sub">Words</div></div>', unsafe_allow_html=True)
            with mcol3:
                st.markdown(f'<div class="metric-box"><div class="metric-sub">Reduced</div><div class="metric-num">{res["reduction_pct"]}%</div><div class="metric-sub">Compressed</div></div>', unsafe_allow_html=True)
            with mcol4:
                st.markdown(f'<div class="metric-box"><div class="metric-sub">Saved</div><div class="metric-num">~{res["time_saved"]}m</div><div class="metric-sub">Read Time</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Display View Tabs: Executive Paragraph vs Key Points
            tab_para, tab_bullets, tab_keywords = st.tabs(["📄 Executive Summary", "📌 Key Point Bullets", "🏷️ Keywords"])
            
            with tab_para:
                st.markdown(f'<div class="summary-box">{res["summary_text"]}</div>', unsafe_allow_html=True)

            with tab_bullets:
                for s in res['summary_sentences']:
                    st.markdown(f'<div class="bullet-item">• {s}</div>', unsafe_allow_html=True)

            with tab_keywords:
                st.write("**Extracted Top Key Entities:**")
                st.write(" • ".join(res['top_keywords']))
                if res['word_freq']:
                    chart_df = pd.DataFrame(res['word_freq'], columns=['Keyword', 'Importance']).sort_values(by='Importance', ascending=True)
                    st.bar_chart(chart_df.set_index('Keyword'), horizontal=True, color="#10B981")

        else:
            st.info("Please enter a news article or select a preset and click 'Generate Summary'.")

    # Bottom Expander: Architecture Pipeline
    with st.expander("› How does the NLP Summarization Algorithm work?"):
        st.markdown("""
        **Summarization NLP Workflow:**
        1. **Sentence Boundary Disambiguation:** Input text is parsed into grammatical sentence entities using regex lookbehind rules.
        2. **Word Normalization & Stop-word Filtering:** Non-alphanumeric noise and frequent syntactic stop-words are eliminated.
        3. **Term Frequency Normalization:** Word salience is computed: $Weight(w) = Frequency(w) / MaxFrequency$.
        4. **Sentence Scoring Function:** Each sentence is evaluated:
           $$Score(S_i) = \\frac{\\sum_{w \\in S_i} Weight(w)}{\\sqrt{Length(S_i)}}$$
        5. **Ranked Extraction:** Sentences with highest information density are extracted while preserving their original chronological order.
        """)

    # Project Limitation Note
    st.markdown("""
        <div class="disclaimer-card">
            <strong>Project Note:</strong> Developed for the <strong>Natural Language Processing Lab Mini-Project</strong>, Department of Artificial Intelligence & Data Science Engineering, Rizvi College of Engineering. This system uses frequency-weighted extractive NLP algorithms and operates 100% locally with zero external API dependencies.
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    f1, f2 = st.columns([1, 1])
    with f1:
        st.caption("AI-Based News Summarizer • Python + Streamlit • University of Mumbai (2026-2027)")
    with f2:
        st.caption("Developed by **Khan Yahya Shahid Ali** (UIN: 221A048 | Roll No: 28)")


# ==========================================
# 4. HEADLESS TERMINAL CLI DEMO
# ==========================================
def run_cli_demo():
    print("=" * 70)
    print("  AI-Based News Summarizer Using Natural Language Processing")
    print("  Author: Khan Yahya Shahid Ali | UIN: 221A048 | Roll: 28")
    print("  Department of AI & DS Engineering | Rizvi College of Engineering")
    print("=" * 70)

    for topic, text in SAMPLE_ARTICLES.items():
        print(f"\n[Testing Topic]: {topic}")
        res = summarize_text(text, summary_ratio=0.4)
        print(f"  • Original:     {res['total_words']} words ({res['total_sentences']} sentences)")
        print(f"  • Summary:      {res['summary_words']} words ({res['summary_sentences_count']} sentences)")
        print(f"  • Compression:  Reduced by {res['reduction_pct']}% (~{res['time_saved']} min saved)")
        print(f"  • Top Keywords: {', '.join(res['top_keywords'][:4])}")
        print("  • Summary Output:")
        for s in res['summary_sentences']:
            print(f"    - {s}")

    print("\n" + "=" * 70)
    print("To launch the Streamlit Web Application:")
    print("  1. pip install -r requirements.txt")
    print("  2. streamlit run app.py")
    print("=" * 70)


if __name__ == '__main__':
    if HAS_STREAMLIT:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx():
                run_streamlit_app()
            else:
                run_cli_demo()
        except Exception:
            run_cli_demo()
    else:
        run_cli_demo()
