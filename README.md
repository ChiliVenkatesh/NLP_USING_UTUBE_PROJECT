# 🎓 AI Lecture Intelligence System

Turns any YouTube lecture into a full study kit: transcript, summary, study notes, quiz, concept map, and a translated summary — all from a pasted URL, served through a Streamlit UI.

## Pipeline

```
YouTube URL → Audio → Transcript → Summary → Notes / Quiz / Concept Map → Translation
```

| Stage | How |
|---|---|
| Download | `yt-dlp` pulls audio from the given YouTube URL |
| Transcribe | Faster-Whisper (model size selectable: tiny / base / small) |
| Summarize | Hugging Face `sshleifer/distilbart-cnn-12-6` summarization pipeline |
| Notes & Quiz | Google FLAN-T5 (`small` or `base`), generated in parallel with the concept map |
| Concept Map | TF-IDF keyword extraction + co-occurrence graph, rendered with Graphviz |
| Translate | `deep-translator` (Google Translate) — supports English, Telugu, Hindi, Tamil, Kannada, Malayalam |

## Features

- 🎥 Paste any YouTube lecture URL — audio is downloaded and transcribed automatically
- 📝 Full transcript with detected source language and confidence score
- 📄 Adjustable summary length (Short / Medium / Detailed)
- 📚 Auto-generated study notes and a quiz from the summary
- 🧠 Concept map showing how key terms co-occur across the lecture
- 🌐 Summary translation into six languages
- ⚡ GPU auto-detected and used when available; falls back to CPU
- 🔒 Per-session file naming so concurrent users don't collide on a shared deployment

## Tech Stack

- **UI:** Streamlit
- **Download:** yt-dlp
- **Transcription:** faster-whisper
- **Summarization / Notes / Quiz:** Hugging Face Transformers (DistilBART, FLAN-T5)
- **Concept map:** scikit-learn TF-IDF + Graphviz
- **Translation:** deep-translator
- **Audio processing:** ffmpeg

## Project Structure

```
.
├── app.py                 # Streamlit app — full pipeline + UI
├── requirements.txt
├── downloads/ audio/ transcripts/ summaries/
├── notes/ quiz/ translations/ concepts/   # generated output folders (created at runtime)
```

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install ffmpeg and Graphviz** (required, not pip-installable)
   - ffmpeg: https://ffmpeg.org/download.html
   - Graphviz: https://graphviz.org/download/

   If they're not on your system `PATH`, point the app at them instead:
   ```bash
   export FFMPEG_BIN=/path/to/ffmpeg/bin
   export GRAPHVIZ_BIN=/path/to/graphviz/bin
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` needs filling in — at minimum: `streamlit`, `torch`, `yt-dlp`, `faster-whisper`, `transformers`, `deep-translator`, `scikit-learn`, `graphviz`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Paste a YouTube lecture URL.
2. Pick a Whisper model size, summary length, target translation language, and notes/quiz model speed in the sidebar.
3. Click **🚀 Process Lecture**.
4. Browse results across the Transcript, Summary, Study Notes, Quiz, Concept Map, and Translation tabs.

## ⚠️ Before pushing to GitHub

- **Do NOT commit `cookies.txt`.** It currently contains live YouTube session/auth tokens (`__Secure-3PSID`, `__Secure-3PAPISID`, etc.) exported by yt-dlp. Anyone with this file can access your Google/YouTube session. Delete it from the repo folder or add it to `.gitignore` before pushing — it's only needed for age-restricted or login-gated videos and the app already treats it as optional.
- Also gitignore the generated runtime folders (`downloads/`, `audio/`, `transcripts/`, `summaries/`, `notes/`, `quiz/`, `translations/`, `concepts/`) and any sample media files (`.mp3`, `.mkv`) — these are large and regeneratable, not source.
- The `LICENSE` and `README.txt` currently in the folder are actually ffmpeg's own GPL license and build-info text, not your project's — replace `LICENSE` with your project's chosen license and drop the ffmpeg build info (or move it into a `third_party/` credits file if you want to keep it).

Suggested `.gitignore`:
```
cookies.txt
downloads/
audio/
transcripts/
summaries/
notes/
quiz/
translations/
concepts/
*.mp3
*.mkv
```

## License

MIT (or update as needed — see note above about the current `LICENSE` file).
