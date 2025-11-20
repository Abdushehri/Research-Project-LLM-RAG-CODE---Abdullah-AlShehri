from __future__ import annotations
# ---------------------------------------------------------------------
#LLM Adaptive Learning Assistant — Streamlit
#GPT-5 + RAG 

#Run:
#  pip install --upgrade openai streamlit tiktoken edge-tts requests pandas numpy textstat
#  streamlit run filename.py

#Notes:
#- Uses your OpenAI key for lesson generation.
#- Supports RAG and TTS output.
#- Includes updated readability + token allocation logic.
# ------------------------------------------------------------------

# SECTION 1: Imports, textstat, readability, token logic

import os, re, io, asyncio, tempfile, json
from dataclasses import dataclass
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ------------------ OpenAI client ------------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ------------------ Edge TTS ------------------
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

# ------------------ textstat  ------------------
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

# ------------------ API key & embedding model ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "INSERT YOUR API KEY HERE")
EMBED_MODEL = "text-embedding-3-small"


# ------------------ Readability metrics ------------------
def _split_sentences(text: str) -> list[str]:
    """
    split text into sentences using punctuation
    """
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

def _word_tokens(text: str) -> list[str]:
    """
    Tokenize text into words
    """
    return re.findall(r"[A-Za-z']+", text)

VOWELS = set("aeiouyAEIOUY")

def _syllable_count(word: str) -> int:
    """
    syllable counter used for Flesch-Kincaid and Gunning Fog when textstat is unavailable.
    """
    w = word.lower()
    if len(w) <= 3:
        return 1
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# Below are manually implemented formulas of the metrics used for evaluation 
def flesch_kincaid_grade(text: str) -> float:
    sentences = _split_sentences(text)
    words = _word_tokens(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllable_count(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59

def gunning_fog_index(text: str) -> float:
    sentences = _split_sentences(text)
    words = _word_tokens(text)
    if not sentences or not words:
        return 0.0
    complex_words = [w for w in words if _syllable_count(w) >= 3]
    return 0.4 * ((len(words) / len(sentences)) + 100 * (len(complex_words) / len(words)))

def avg_sentence_length(text: str) -> float:
    sentences = _split_sentences(text)
    words = _word_tokens(text)
    return len(words) / len(sentences) if sentences else 0.0

def type_token_ratio(text: str) -> float:
    words = [w.lower() for w in _word_tokens(text)]
    return len(set(words)) / len(words) if words else 0.0

# ---- Wrappers fot textstat ----
def fk_grade(text: str) -> float:
    if TEXTSTAT_AVAILABLE:
        try:
            return float(textstat.flesch_kincaid_grade(text))
        except Exception:
            pass
    return flesch_kincaid_grade(text)

def gf_grade(text: str) -> float:
    if TEXTSTAT_AVAILABLE:
        try:
            return float(textstat.gunning_fog(text))
        except Exception:
            pass
    return gunning_fog_index(text)

# ------------------ TOKEN LOGIC ------------------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _tokens_from_engagement(education: str, engagement) -> int:
    """
    Determine token count based on education level, then apply engagement modifier.
    
    Base tokens:
        Post-grad     = 2500
        Undergraduate = 2000
        High School   = 1500
    
    Engagement modifier:
        High   = x1.0
        Medium = x0.9
        Low    = x0.8
    """

    # --- Base tokens based on education level ---
    edu_norm = _normalize(education)
    if "post" in edu_norm or "grad" in edu_norm or "master" in edu_norm or "phd" in edu_norm:
        base = 2500
    elif "under" in edu_norm or "bachelor" in edu_norm:
        base = 2000
    else:
        base = 1500  

    # --- Engagement modifier ---
    if engagement is None:
        modifier = 1.0
    else:
        if isinstance(engagement, str):
            e = _normalize(engagement)
            if "high" in e:
                modifier = 1.0
            elif "medium" in e:
                modifier = 0.9
            elif "low" in e:
                modifier = 0.8
            else:
                modifier = 1.0
        else:
            modifier = 1.0

    return int(base * modifier)


# Section 2: Style templates, Conformance validator, Fallback generator, OpenAI call wrapper

# ------------------ Style templates ------------------
STYLE_VISUAL = (
    "You are a teacher building visual learning materials. Audience: {age}, {education}\n"
    "Topic: {topic}\nSource text:\n\"\"\"\n{source}\n\"\"\"\n"
    "Produce: 1) Key Idea (<=2 sentences) 2) Diagram (mermaid code block) "
    "3) Captioned Walkthrough (<=7 bullets) 4) Quick Check."
)

STYLE_AUDITORY = (
    "You are a tutor writing an audio script for {age}, {education}.\n"
    "Topic: {topic}\nSource:\n\"\"\"\n{source}\n\"\"\"\n"
    "Write SSML with Hook, Explain, Example, Recap, Self-Check. Use <speak>...</speak>."
)

STYLE_RW = (
    "Audience: {age}, {education}\nTask: Convert to study notes.\n"
    "Sections: Outline, Key Terms table, Worked Example, 3 Practice Prompts + Answers.\n"
    "Source:\n\"\"\"\n{source}\n\"\"\""
)

STYLE_KINESTHETIC = (
    "Design a mini-activity for {age}, {education}.\n"
    "Include: Materials, Setup, Activity Steps, Checkpoints, Reflection, Safety.\n"
    "Source:\n\"\"\"\n{source}\n\"\"\""
)

STYLE_TEMPLATES: Dict[str, str] = {
    "Visual": STYLE_VISUAL,
    "Auditory": STYLE_AUDITORY,
    "Reading/Writing": STYLE_RW,
    "Kinesthetic": STYLE_KINESTHETIC,
}

# ------------------ Conformance Checking ------------------
@dataclass
class ConformanceResult:
    ok: bool
    messages: list[str]

def check_conformance(style: str, text: str) -> ConformanceResult:
    """
    Check whether the generated text includes key structures
    """
    msgs: list[str] = []

    if style == "Visual" and not re.search(r"```mermaid[\s\S]*?```", text, re.IGNORECASE):
        msgs.append("Missing mermaid diagram block.")

    if style == "Auditory" and not re.search(r"^\s*<speak\b", text, re.IGNORECASE | re.DOTALL):
        msgs.append("SSML must be wrapped in <speak>...</speak>.")

    if style == "Reading/Writing" and not re.search(r"\|[^|]*term[^|]*\|[^|]*defin", text, re.IGNORECASE):
        msgs.append("Missing Key Terms markdown table.")

    if style == "Kinesthetic" and not re.search(r"\bMaterials\b", text, re.IGNORECASE):
        msgs.append("Missing Materials section.")

    return ConformanceResult(ok=(len(msgs) == 0), messages=msgs)

# ------------------ Fallback generator ------------------
def fallback_generate(prompt: str) -> str:
    """
    Provide a simple hard-coded output when the API call fails or no key is set.
    NO API = NO LLM. This is just to avoid errors
    """
    if "mermaid" in prompt:
        return (
            "Key Idea: This is a simplified visual explanation.\n\n"
            "```mermaid\nflowchart TD\n  A[Start] --> B[Concept 1] --> C[Concept 2] --> D[End]\n```\n\n"
            "- Step 1: Brief note\n- Step 2: Brief note\n- Step 3: Brief note\n\n"
            "Quick Check: What links Concept 1 to Concept 2?"
        )

    if "<speak>" in prompt:
        return (
            "<speak>\n <p><emphasis>Hook:</emphasis> Imagine a bright sunny day. <break time=\"500ms\"/></p>\n"
            "<p><emphasis>Explain:</emphasis> Let's explore how plants use sunlight to make food.</p>\n"
            "<p><emphasis>Example:</emphasis> Think of a leaf as a small solar panel!</p>\n"
            "<p><emphasis>Recap:</emphasis> Plants make glucose and oxygen through photosynthesis.</p>\n"
            "<p><emphasis>Self-Check:</emphasis> Can you name the inputs for photosynthesis?</p>\n</speak>"
        )

    if "Key Terms" in prompt:
        return (
            "## Outline\n\n| Term | Definition |\n|------|-------------|\n"
            "| Photosynthesis | Process where plants make sugar using sunlight |\n"
            "| Chlorophyll | Green pigment that absorbs sunlight |\n"
        )

    return "Draft lesson content."

# ------------------ OpenAI call wrapper ------------------
def call_llm(prompt: str, system: str | None = None,
             temperature: float = 0.4, max_tokens: int = 900) -> str:

    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages: list[dict] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        try:
            resp = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        except Exception as e:
            return f"Offline fallback due to API error: {type(e).__name__}: {e}\n\n" + fallback_generate(prompt)

    return "Offline fallback (no API key).\n\n" + fallback_generate(prompt)

# SECTION 3: RAG index builder, Embedding helper, context retriver, prompt builder

# ------------------ RAG ------------------
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {_normalize(c): c for c in df.columns}
    for name in candidates:
        if _normalize(name) in cols:
            return cols[_normalize(name)]
    return None


def _ensure_rag_index():
    """
    load a simple RAG index from st.session_state['rag_df']

      - st.session_state['rag_vectors']: np.ndarray (N, D)
      - st.session_state['rag_meta']:   list of dicts
    """
    df_rag = st.session_state.get("rag_df")
    if df_rag is None or df_rag.empty:
        return None, None

    if "rag_vectors" in st.session_state and "rag_meta" in st.session_state:
        return st.session_state["rag_vectors"], st.session_state["rag_meta"]

    if not (OPENAI_AVAILABLE and OPENAI_API_KEY):
        return None, None

    client = OpenAI(api_key=OPENAI_API_KEY)

    topic_col = _pick_col(df_rag, ["topic"])
    source_col = _pick_col(df_rag, ["source", "content", "text"])
    out_col = _pick_col(df_rag, ["ideal_output", "output", "lesson", "target"])
    style_col = _pick_col(df_rag, ["style", "learning_style"])
    age_col = _pick_col(df_rag, ["age"])
    edu_col = _pick_col(df_rag, ["education", "education_level", "level"])

    if not topic_col or not source_col or not out_col:
        return None, None

    vecs = []
    meta = []

    for _, row in df_rag.iterrows():
        topic = str(row.get(topic_col, "")).strip()
        source = str(row.get(source_col, "")).strip()
        ideal = str(row.get(out_col, "")).strip()

        if not topic or not source or not ideal:
            continue

        doc_text = f"Topic: {topic}\n\nSource:\n{source}\n\nLesson:\n{ideal}"

        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=[doc_text]
            )
            emb = resp.data[0].embedding
        except Exception:
            continue

        vecs.append(emb)
        meta.append({
            "topic": topic,
            "source": source,
            "ideal_output": ideal,
            "style": str(row.get(style_col, "")) if style_col else "",
            "age": int(row.get(age_col)) if age_col and pd.notna(row.get(age_col)) else None,
            "education": str(row.get(edu_col, "")) if edu_col else "",
        })

    if not vecs:
        return None, None

    vec_arr = np.array(vecs, dtype="float32")

    st.session_state["rag_vectors"] = vec_arr
    st.session_state["rag_meta"] = meta

    return vec_arr, meta


def _embed_query(text: str) -> np.ndarray | None:
    """
    Embed a query string using the same embedding model as the RAG index to retun a numpy victor
    """
    if not (OPENAI_AVAILABLE and OPENAI_API_KEY):
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        return np.array(resp.data[0].embedding, dtype="float32")
    except Exception:
        return None


def retrieve_context(style: str, topic: str, age: int, education: str,
                     source: str, k: int = 3) -> str:
    
    """
    Retrieve top-k similar examples from the RAG index
    based on the current topic, style, age, education and source text.
    """

    vecs, meta = _ensure_rag_index()
    if vecs is None or meta is None:
        return ""

    q_text = (
        f"Topic: {topic}\n"
        f"Source: {source}\n"
        f"Style: {style}\n"
        f"Age: {age}, Education: {education}"
    )

    q_vec = _embed_query(q_text)
    if q_vec is None:
        return ""

    # Cosine similarity
    dot = vecs @ q_vec
    q_norm = np.linalg.norm(q_vec)
    v_norm = np.linalg.norm(vecs, axis=1)
    sim = dot / (q_norm * v_norm + 1e-8)

    top_idx = np.argsort(-sim)[:k]

    chunks = []
    for i in top_idx:
        m = meta[int(i)]
        chunks.append(
            f"Topic: {m['topic']}\n"
            f"Source: {m['source']}\n"
            f"Sample Lesson:\n{m['ideal_output']}\n"
        )

    return "\n\n---\n\n".join(chunks)


# ------------------ Prompt Builder ------------------
def build_prompt(style: str, topic: str, source: str, age: int,
                 education: str) -> Tuple[str, str]:
    """
    Build the system and user prompts for the LLM:
    - fetches the student ID profile
    - based on learning style select template.
    - uses retrieved examples from the RAG index.
    """

    tpl = STYLE_TEMPLATES[style]

    # System prompt encourages clarity 
    system = (
        f"You adapt content for age {age}, {education}. "
        "Use clear explanations and structured formatting."
    )

    # retrieving matching examples from RAG
    try:
        rag_context = retrieve_context(style, topic, age, education, source, k=3)
    except Exception:
        rag_context = ""

    full_source = source
    if rag_context:
        full_source = (
            f"{source}\n\n"
            f"Additional helpful examples and notes:\n"
            f"---\n{rag_context}"
        )

    prompt = tpl.format(
        age=age,
        education=education,
        topic=topic,
        source=full_source
    )

    return system, prompt

# SECTION 4: TTS, Style mapping, temperature resolver, UI

# ------------------ TTS ------------------
def _ssml_to_plain_text(ssml_text: str) -> str:
    """
    Strip SSML tags and converting into plain text
    this helps for both text-to-speach and for readability metrics
    """
    text = ssml_text
    text = re.sub(r"<break[^>]*/>", ". ", text)
    text = re.sub(r"</p>\s*", "\n\n", text)
    text = re.sub(r"<p[^>]*>", "", text)
    text = re.sub(r"<emphasis[^>]*>(.*?)</emphasis>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\b(Hook|Explain|Example|Recap|Self-Check|Intro|Summary)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?speak[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

async def _synthesize_edge_tts_text(narration_text: str, voice: str, out_path: str):
    communicate = edge_tts.Communicate(text=narration_text, voice=voice)
    await communicate.save(out_path)

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)

def ssml_to_audio_bytes(ssml_text: str, voice: str = "en-US-JennyNeural") -> bytes:
    if not EDGE_TTS_AVAILABLE:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
    narration = _ssml_to_plain_text(ssml_text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name
    try:
        _run_async(_synthesize_edge_tts_text(narration, voice, tmp_path))
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ------------------ Style Mapping ------------------
_STYLE_ALIASES = {
    "visual": "Visual",
    "auditory": "Auditory",
    "readingwriting": "Reading/Writing",
    "reading/writing": "Reading/Writing",
    "rw": "Reading/Writing",
    "kinesthetic": "Kinesthetic",
}

def _map_style(val: str | None) -> str:
    if not val:
        return "Reading/Writing"
    key = _normalize(val)
    return _STYLE_ALIASES.get(key, "Reading/Writing")

def _temperature_from_row(row: pd.Series) -> float:
    for cand in ["temperature", "creativity", "randomness", "temp"]:
        if cand in row and pd.notna(row[cand]):
            try:
                t = float(row[cand])
                return max(0.0, min(t, 1.0))
            except Exception:
                pass
    return 0.4

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Adaptive Learning Assistant — GPT-5 + TTS", layout="wide")
st.title("LLM Adaptive learning Assistant")
st.caption("AI content — verify before classroom use.")

# --- Sidebar ---
with st.sidebar:
    st.header("Data Import & Learner Selection")

    # Learner dataset 
    ds_file = st.file_uploader("Import learner dataset (CSV)", type=["csv"])
    if ds_file is not None:
        try:
            df = pd.read_csv(ds_file)
            st.session_state["dataset_df"] = df
            st.success(f"Loaded {len(df)} learners")
            with st.expander("Preview learner dataset", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {type(e).__name__}: {e}")

    # Content knowledge base for RAG
    st.subheader("Content Knowledge Base (RAG)")
    rag_file = st.file_uploader(
        "Import lesson examples (CSV for RAG)", type=["csv"], key="rag_uploader"
    )
    if rag_file is not None:
        try:
            rag_df = pd.read_csv(rag_file)
            st.session_state["rag_df"] = rag_df
            # Clear any previous index so it's rebuilt with new data
            st.session_state.pop("rag_vectors", None)
            st.session_state.pop("rag_meta", None)
            st.success(f"Loaded {len(rag_df)} content rows for retrieval.")
            with st.expander("Preview RAG content", expanded=False):
                st.dataframe(rag_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read RAG CSV: {type(e).__name__}: {e}")

    df = st.session_state.get("dataset_df")
    learner_row = None
    style_resolved, age_resolved, education_resolved = "Reading/Writing", 16, "High School"
    temperature_resolved, max_tokens_resolved = 0.4, 500

    if df is not None and not df.empty:
        id_col = _pick_col(df, ["id", "learner_id", "student_id", "user_id"])
        if id_col is None:
            st.warning("No ID column found.")
        else:
            selected_id = st.selectbox("Select learner ID", df[id_col].astype(str))
            if selected_id:
                learner_row = df[df[id_col].astype(str) == selected_id].iloc[0]

    if learner_row is not None:
        style_col = _pick_col(df, ["learning_style", "style"])
        age_col = _pick_col(df, ["age", "learner_age"])
        edu_col = _pick_col(df, ["education_level", "education", "level"])
        engage_col = _pick_col(df, ["engagement_level", "engagement", "tokens"])

        style_resolved = _map_style(learner_row.get(style_col)) if style_col else "Reading/Writing"
        try:
            age_resolved = int(learner_row.get(age_col)) if age_col else 16
        except Exception:
            age_resolved = 16

        education_resolved = str(learner_row.get(edu_col)) if edu_col else "High School"

        # tokens depend on education + engagement
        engagement_val = learner_row.get(engage_col) if engage_col else None
        max_tokens_resolved = _tokens_from_engagement(
            education_resolved,
            engagement_val,
        )

        temperature_resolved = _temperature_from_row(learner_row)

        st.caption("Learner profile (from dataset)")
        st.write(
            {
                "style": style_resolved,
                "age": age_resolved,
                "education": education_resolved,
                "temperature": temperature_resolved,
                "max_tokens (education+engagement)": max_tokens_resolved,
            }
        )
    else:
        st.info("Import learner CSV and select a learner to populate profile.")

    st.subheader("Text-to-Speech (Auditory)")
    tts_enabled = st.checkbox("Enable Audio", value=True)
    voice = st.text_input("Voice", value="en-US-JennyNeural")
    if not EDGE_TTS_AVAILABLE and tts_enabled:
        st.info("Install with: pip install edge-tts")

    st.session_state.update(
        {
            "resolved_style": style_resolved,
            "resolved_age": age_resolved,
            "resolved_education": education_resolved,
            "resolved_temperature": float(temperature_resolved),
            "resolved_max_tokens": int(max_tokens_resolved),
        }
    )

# --- Main Layout ---
# The first column is used for the setup and file importing, coloumn 2 is for the output and evaluation
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Setup")
    st.write(
        "1) Import learner dataset and pick learner  "
        "\n2) Import content CSV for RAG  "
        "\n3) Enter an idea/topic  "
        "\n4) Generate."
    )
    idea = st.text_input("Idea / Topic", value="Photosynthesis")
    source = st.text_area(
        "Source text (optional)",
        height=260,
        value="Photosynthesis is how plants use sunlight to make sugar from carbon dioxide and water.",
    )

    if st.button("Generate Lesson", type="primary"):
        style = st.session_state["resolved_style"]
        age = int(st.session_state["resolved_age"])
        education = st.session_state["resolved_education"]
        temperature = float(st.session_state["resolved_temperature"])
        max_tokens = int(st.session_state["resolved_max_tokens"])

        if st.session_state.get("dataset_df") is None:
            st.warning("Please import learner dataset and select a learner.")
        else:
            system, user_prompt = build_prompt(style, idea, source, age, education)
            with st.spinner("Generating..."):
                out = call_llm(
                    user_prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            st.session_state["last_output"] = out
            st.session_state["last_style"] = style
            st.session_state["last_topic"] = idea

with col2:
    st.subheader("Adapted Output")
    output = st.session_state.get("last_output")
    style = st.session_state.get("last_style")

    if output:
        if style == "Auditory" and re.search(r"^\s*<speak\b", output, re.IGNORECASE):
            st.info("Audio script generated. Listen or download below.")
            if tts_enabled and EDGE_TTS_AVAILABLE and st.button("🔊 Generate & Play Audio"):
                try:
                    audio = ssml_to_audio_bytes(output, voice)
                    st.audio(audio, format="audio/mp3")
                    st.download_button(
                        "Download MP3",
                        data=audio,
                        file_name=f"{st.session_state.get('last_topic','lesson').replace(' ', '_')}_audio.mp3",
                    )
                except Exception as e:
                    st.error(f"TTS failed: {type(e).__name__}: {e}")
            with st.expander("Show SSML (advanced view)", expanded=False):
                st.code(output, language="xml")
        else:
            st.markdown(output)

        st.divider()
        st.subheader("Evaluator")

        # Run metrics on plain text if SSML
        metrics_text = _ssml_to_plain_text(output) if (style == "Auditory") else output
        fk = fk_grade(metrics_text)
        gf = gf_grade(metrics_text)
        asl = avg_sentence_length(metrics_text)
        ttr = type_token_ratio(metrics_text)
        conf = check_conformance(style, output)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flesch–Kincaid Grade", f"{fk:.2f}")
        m2.metric("Gunning Fog", f"{gf:.2f}")
        m3.metric("Avg Sent Length", f"{asl:.2f}")
        m4.metric("Type–Token Ratio", f"{ttr:.2f}")
        if conf.ok:
            st.success("Style conformance checks passed.")
        else:
            for msg in conf.messages:
                st.error(msg)

        st.divider()
        st.subheader("Export")
        if style == "Auditory":
            fname = f"{st.session_state.get('last_topic','lesson').replace(' ', '_').lower()}_audio.ssml"
            data = output
        else:
            fname = (
                f"{st.session_state.get('last_topic','lesson').replace(' ', '_').lower()}_"
                f"{style.replace('/', '_').lower()}.md"
            )
            data = output

        st.download_button("Download Output", data=data, file_name=fname)

st.caption("© 2025 — LLM modal Assistant. Built with Streamlit, OpenAI GPT-5, and RAG. TTS via edge-tts.")
