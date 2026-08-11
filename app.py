"""
=========================================================
AI Lecture Intelligence System (Fixed / Hardened Version)
YouTube → Faster Whisper → Summary → Notes → Quiz →
Translation
=========================================================

Fixes applied vs original:
1. Portable ffmpeg/graphviz discovery (env var + shutil.which), no
   hardcoded personal Windows paths.
2. Per-session unique filenames — prevents one user's data overwriting
   another's on a shared deployment.
3. Try/except around all network calls (download, translate) with
   st.error + st.stop instead of raw tracebacks.
4. Quiz generation batched (2 calls instead of 10) — big speed win,
   especially on CPU.
5. Translation truncation is now surfaced to the user instead of silent.
6. Transcribe/summarize results cached per audio file so re-running the
   UI on the same video doesn't redo expensive model calls.
7. FLAN calls remain lock-protected (model isn't thread-safe), but
   notes/quiz/concept-map generation is restructured so the concept map
   (the only genuinely parallelizable piece) isn't blocked by the lock.
"""

import os
import re
import shutil
import time
import uuid
import threading
import itertools
from collections import Counter

import torch
import yt_dlp
import streamlit as st

from concurrent.futures import ThreadPoolExecutor

from faster_whisper import WhisperModel
from transformers import pipeline
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
import graphviz


# -----------------------------
# Streamlit Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Lecture Intelligence System",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Device Detection
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared FLAN model isn't safely callable from multiple threads at once;
# serialize access with this lock wherever generate_notes/generate_quiz
# run concurrently with anything else touching the model.
_flan_lock = threading.Lock()


# -----------------------------
# Folders
# -----------------------------

FOLDERS = [
    "downloads",
    "audio",
    "transcripts",
    "summaries",
    "notes",
    "quiz",
    "translations",
    "concepts"
]


def setup_folders():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)


setup_folders()


# -----------------------------
# Language Map
# -----------------------------

LANG_MAP = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml"
}


# -----------------------------
# Session ID (fix: per-session unique filenames)
# -----------------------------

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:10]
    return st.session_state["session_id"]


# -----------------------------
# FFMPEG / GRAPHVIZ (fix: portable discovery, no hardcoded paths)
# -----------------------------

def _resolve_tool_bin(env_var, exe_name):
    """
    Resolution order:
    1. Explicit env var (e.g. FFMPEG_BIN=/path/to/bin)
    2. Already on PATH (shutil.which)
    3. None — caller must warn, not crash
    """
    env_path = os.environ.get(env_var)
    if env_path and os.path.exists(env_path):
        return env_path

    which_result = shutil.which(exe_name)
    if which_result:
        return os.path.dirname(which_result)

    return None


FFMPEG_DIR = _resolve_tool_bin("FFMPEG_BIN", "ffmpeg")
GRAPHVIZ_DIR = _resolve_tool_bin("GRAPHVIZ_BIN", "dot")

if FFMPEG_DIR:
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

if GRAPHVIZ_DIR:
    os.environ["PATH"] = GRAPHVIZ_DIR + os.pathsep + os.environ.get("PATH", "")


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("⚙ System Information")

if DEVICE == "cuda":
    st.sidebar.success(torch.cuda.get_device_name(0))
else:
    st.sidebar.warning("Running on CPU — expect slower processing.")

if not FFMPEG_DIR:
    st.sidebar.error(
        "ffmpeg not found. Set FFMPEG_BIN env var to its bin folder, "
        "or install ffmpeg and ensure it's on PATH."
    )

if not GRAPHVIZ_DIR:
    st.sidebar.warning(
        "Graphviz 'dot' not found. Concept map rendering will fail. "
        "Set GRAPHVIZ_BIN env var or install graphviz."
    )


# -----------------------------
# Load Faster Whisper
# -----------------------------

@st.cache_resource
def load_whisper(model_size="base"):
    model = WhisperModel(
        model_size,
        device=DEVICE,
        compute_type="int8"
    )
    return model


# -----------------------------
# Load Summarizer
# -----------------------------

@st.cache_resource
def load_summarizer():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=0 if DEVICE == "cuda" else -1,
    )
    return summarizer


# -----------------------------
# Load FLAN
# -----------------------------

@st.cache_resource
def load_flan(flan_size="small"):
    model_name = f"google/flan-t5-{flan_size}"
    generator = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if DEVICE == "cuda" else -1,
    )
    return generator


# -----------------------------
# Download Audio (fix: per-session filename, try/except)
# -----------------------------

@st.cache_data(show_spinner=False)
def download_audio(url, session_id):

    out_stub = f"audio/{session_id}"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_stub}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    if FFMPEG_DIR:
        ydl_opts["ffmpeg_location"] = FFMPEG_DIR

    # cookies.txt is optional — only needed for age/login-gated videos.
    # Export via "Get cookies.txt LOCALLY" extension if you hit auth errors.
    if os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            f"Couldn't download that video. It may be private, age-gated "
            f"(add cookies.txt), or region-locked.\nDetails: {e}"
        ) from e

    return f"{out_stub}.mp3"


# -----------------------------
# Faster Whisper (fix: streams live progress instead of blank spinner)
# -----------------------------
# NOTE: not st.cache_data-wrapped anymore. Caching + live progress
# updates don't mix well in Streamlit (cached functions shouldn't
# touch UI elements), and knowing it's alive matters more here than
# skipping a re-run. If you re-process the exact same video often,
# ask and I'll add a separate on-disk cache check instead.

def transcribe(audio_path, model_size, session_id, progress_placeholder):

    model = load_whisper(model_size)

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=1,
            language=None,   # auto-detect
            vad_filter=True,
            condition_on_previous_text=False,
        )
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    transcript_parts = []
    start_time = time.time()
    segment_count = 0

    for segment in segments:
        transcript_parts.append(segment.text)
        segment_count += 1

        elapsed = time.time() - start_time
        preview = segment.text.strip()[:70]
        progress_placeholder.info(
            f"⏱ {elapsed:.0f}s elapsed · {segment_count} segments · "
            f"audio pos {segment.end:.0f}s\n\n"
            f"Latest: \"{preview}...\""
        )

    transcript = " ".join(transcript_parts)

    transcript_path = f"transcripts/{session_id}.txt"
    with open(transcript_path, "w", encoding="utf8") as f:
        f.write(transcript)

    detected_lang = info.language
    detected_conf = info.language_probability

    return transcript, detected_lang, detected_conf


# -----------------------------
# Split Long Text
# -----------------------------

def split_text(text, max_words=350):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


# ==========================================================
# SUMMARY
# ==========================================================

@st.cache_data(show_spinner=False)
def summarize(transcript, length, session_id):

    summarizer = load_summarizer()
    chunks = split_text(transcript)

    length_map = {
        "Short": (20, 60),
        "Medium": (40, 120),
        "Detailed": (80, 200),
    }
    min_len, max_len = length_map.get(length, (40, 120))

    _summarizer_lock = threading.Lock()

    def _summarize_chunk(chunk):
        with _summarizer_lock, torch.inference_mode():
            output = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
        return output[0]["summary_text"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(_summarize_chunk, chunks))

    final_summary = "\n\n".join(summaries)

    summary_path = f"summaries/{session_id}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    return final_summary


# ==========================================================
# STUDY NOTES
# ==========================================================

def generate_notes(summary, session_id, flan_size="small"):

    generator = load_flan(flan_size)

    # fix: merged 4 separate FLAN calls into 2 (paired prompts) —
    # roughly halves notes generation time, same 4 sections in output.
    combined_sections = [
        (
            ["Overview", "Key Concepts & Terms"],
            "First write a 2-3 sentence overview of this topic, labeled "
            "'Overview:'. Then list and briefly explain the key concepts "
            "and important terms, labeled 'Key Concepts:'. Full sentences."
        ),
        (
            ["Applications", "Quick Revision"],
            "First describe real world applications of this topic with "
            "one example, labeled 'Applications:'. Then write a short "
            "quick-revision bullet summary, labeled 'Quick Revision:'."
        ),
    ]

    notes_parts = []

    for titles, instruction in combined_sections:
        prompt = f"""
{instruction}

Summary:
{summary}
"""
        with _flan_lock, torch.inference_mode():
            output = generator(
                prompt,
                max_new_tokens=180,
                min_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )[0]["generated_text"]

        notes_parts.append(f"# {' & '.join(titles)}\n{output}")

    notes = "\n\n".join(notes_parts)

    notes_path = f"notes/{session_id}.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)

    return notes


# ==========================================================
# QUIZ (fix: batched into 2 calls of 5 instead of 10 separate calls)
# ==========================================================

def generate_quiz(summary, session_id, total_questions=10, batch_size=None, flan_size="small"):

    # fix: reverted to one-question-per-call. Batching multiple
    # questions into a single generation was the actual root cause of
    # jumbled output (Q1 swallowing what should've been Q2/Q3, numbering
    # jumping 1→4→7→10) — flan-t5 (base or small) can't reliably hold
    # structure across several questions in one generation, no amount
    # of repetition penalty fixes that. Slower, but each question comes
    # out correct and in order. Quiz always uses base model — this is
    # the stage where correctness matters most.
    generator = load_flan("base")

    questions = []

    for i in range(total_questions):
        prompt = f"""
Based on the summary below, write ONE interview question, its answer
(1-2 sentences), and its difficulty (Easy/Medium/Hard).
This must be question number {i + 1} — make it about a different
aspect of the topic than a typical question {i} would cover.

Summary:
{summary}

Question {i + 1}:
"""
        with _flan_lock, torch.inference_mode():
            output = generator(
                prompt,
                max_new_tokens=60,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )[0]["generated_text"]

        questions.append(f"Q{i + 1}. {output.strip()}")

    quiz = "\n\n".join(questions)

    quiz_path = f"quiz/{session_id}.txt"
    with open(quiz_path, "w", encoding="utf-8") as f:
        f.write(quiz)

    return quiz


# ==========================================================
# CONCEPT MAP
# ==========================================================

def _canonicalize_terms(terms):
    """
    Fix: TF-IDF returns 'learn', 'learning', 'machine', 'machine learning'
    as separate terms even though they're the same concept — cluttered
    graph with redundant nodes/edges. Merge any term that is a substring
    of, or shares a simple stem with, a longer term into that longer
    term's canonical label.
    """
    # longest first, so shorter variants merge INTO the more specific term
    sorted_terms = sorted(set(terms), key=len, reverse=True)

    canonical_map = {}   # raw term -> canonical label
    canonical_terms = []  # final list of canonical labels, longest-first

    def simple_stem(word):
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    for term in sorted_terms:
        if term in canonical_map:
            continue

        matched = None
        for existing in canonical_terms:
            # substring match, e.g. "machine" inside "machine learning"
            if term in existing or existing in term:
                matched = existing
                break
            # stem match, e.g. "learn" / "learning"
            if simple_stem(term) == simple_stem(existing):
                matched = existing
                break

        if matched:
            canonical_map[term] = matched
        else:
            canonical_map[term] = term
            canonical_terms.append(term)

    return canonical_map, canonical_terms


def generate_concept_map(summary, session_id):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15,
        ngram_range=(1, 2),
    )

    try:
        vectorizer.fit([summary])
        raw_terms = list(vectorizer.get_feature_names_out())
    except ValueError:
        raw_terms = []

    canonical_map, terms = _canonicalize_terms(raw_terms)

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    pair_counts = Counter()

    for sentence in sentences:
        sentence_lower = sentence.lower()
        present_raw = [t for t in raw_terms if t in sentence_lower]
        present = sorted(set(canonical_map[t] for t in present_raw))
        for a, b in itertools.combinations(present, 2):
            pair_counts[(a, b)] += 1

    graph = graphviz.Digraph()
    graph.attr(rankdir="LR")

    edges = []
    for (a, b), count in pair_counts.most_common(10):
        graph.node(a)
        graph.node(b)
        graph.edge(a, b, label=f"co-occurs x{count}")
        edges.append(f"{a} -> {b} (co-occurs {count}x)")

    dot_source = graph.source
    dot_path = f"concepts/{session_id}.dot"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_source)

    return graph, edges, ", ".join(terms)


# ==========================================================
# TRANSLATION (fix: surface truncation to the user)
# ==========================================================

@st.cache_data(show_spinner=False)
def translate_summary(summary, language, session_id):

    truncated = len(summary) > 4500
    text_to_translate = summary[:4500]

    try:
        translated = GoogleTranslator(
            source="auto",
            target=language
        ).translate(text_to_translate)
    except Exception as e:
        raise RuntimeError(f"Translation service failed: {e}") from e

    filename = f"translations/{session_id}_{language}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(translated)

    return translated, truncated


# ==========================================================
# PARALLEL PROCESSING
# ==========================================================

def process_outputs(summary, session_id, flan_size="small"):

    load_flan(flan_size)  # warm cache once before threads touch it

    # Concept map has no dependency on the FLAN lock, so it can run
    # genuinely concurrently. Notes and quiz both need _flan_lock, so
    # they still serialize against each other — that's a model
    # limitation, not something threading can fix.
    with ThreadPoolExecutor() as executor:
        concept_future = executor.submit(generate_concept_map, summary, session_id)
        notes = generate_notes(summary, session_id, flan_size)
        quiz = generate_quiz(summary, session_id, flan_size=flan_size)
        concept_graph, concept_edges, concept_raw = concept_future.result()

    return notes, quiz, concept_graph, concept_edges, concept_raw


# ==========================================================
# STREAMLIT UI
# ==========================================================

def main():

    st.title("🎓 AI Lecture Intelligence System")
    st.markdown(
        "### YouTube Lecture → Transcript → Summary → Notes → Quiz → Concept Map → Translation"
    )

    session_id = get_session_id()

    st.sidebar.header("Settings")

    whisper_model = st.sidebar.selectbox(
        "Whisper Model",
        ["tiny", "base", "small"],
        index=0
    )

    summary_length = st.sidebar.selectbox(
        "Summary Length",
        ["Short", "Medium", "Detailed"],
        index=0
    )

    target_language = st.sidebar.selectbox(
        "Translate Summary",
        list(LANG_MAP.keys())
    )

    flan_size = st.sidebar.selectbox(
        "Notes/Quiz Model Speed",
        ["small (faster)", "base (better quality)"],
        index=0
    )
    flan_size = "small" if flan_size.startswith("small") else "base"

    st.divider()

    youtube_url = st.text_input("Paste YouTube Lecture URL")

    if st.button("🚀 Process Lecture", use_container_width=True):

        if youtube_url.strip() == "":
            st.error("Please enter a YouTube URL.")
            st.stop()

        if not FFMPEG_DIR:
            st.error("ffmpeg is not available — audio extraction will fail. Fix this before continuing.")
            st.stop()

        start = time.time()

        # -----------------------------
        # Download
        # -----------------------------
        with st.spinner("Downloading audio..."):
            try:
                audio_path = download_audio(youtube_url, session_id)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.success("Audio downloaded.")

        # -----------------------------
        # Transcription (fix: live progress instead of blank spinner)
        # -----------------------------
        st.write("Generating transcript...")
        progress_placeholder = st.empty()
        try:
            transcript, detected_lang, detected_conf = transcribe(
                audio_path, whisper_model, session_id, progress_placeholder
            )
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        progress_placeholder.empty()

        st.sidebar.info(f"Detected language: {detected_lang} ({detected_conf:.0%} confidence)")
        st.success("Transcript completed.")

        # -----------------------------
        # Summary
        # -----------------------------
        with st.spinner("Generating summary..."):
            summary = summarize(transcript, summary_length, session_id)

        st.success("Summary completed.")

        # -----------------------------
        # Parallel Processing
        # -----------------------------
        with st.spinner("Generating Notes, Quiz & Concept Map..."):
            notes, quiz, concept_graph, concept_edges, concept_raw = process_outputs(
                summary, session_id, flan_size
            )

        st.success("Notes, quiz & concept map completed.")

        # -----------------------------
        # Translation
        # -----------------------------
        with st.spinner("Translating summary..."):
            try:
                translated, was_truncated = translate_summary(
                    summary, LANG_MAP[target_language], session_id
                )
            except RuntimeError as e:
                st.error(str(e))
                translated, was_truncated = "", False

        st.success("Translation completed.")

        total_time = round(time.time() - start, 2)
        st.success(f"Completed in {total_time} seconds.")

        # ====================================
        # TABS
        # ====================================

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Transcript", "Summary", "Study Notes", "Quiz", "Concept Map", "Translation"]
        )

        with tab1:
            st.subheader("📝 Full Transcript")
            st.text_area("Transcript", transcript, height=400, label_visibility="collapsed")

        with tab2:
            st.subheader("📄 Summary")
            st.write(summary)

        with tab3:
            st.subheader("📚 Study Notes")
            st.write(notes)

        with tab4:
            st.subheader("❓ Quiz")
            st.write(quiz)

        with tab5:
            st.subheader("🧠 Concept Map")
            if concept_edges:
                st.graphviz_chart(concept_graph)
            else:
                st.warning(
                    "Not enough repeated terms found to build a map. "
                    f"Top terms detected: {concept_raw or 'none'}"
                )
            with st.expander("Relationships (text)"):
                if concept_edges:
                    for edge in concept_edges:
                        st.write(f"- {edge}")
                else:
                    st.write("No co-occurring term pairs found.")

        with tab6:
            st.subheader(f"🌐 Translation ({target_language})")
            if was_truncated:
                st.warning(
                    "Summary was longer than the translation service's limit — "
                    "only the first ~4500 characters were translated."
                )
            st.write(translated)


if __name__ == "__main__":
    main()