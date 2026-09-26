
import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os, re, csv, io

st.set_page_config(page_title="Voiceover → Channel Farm Scene Tool", page_icon="🎬", layout="wide")

STYLE = """historical stickman documentary illustration, simple white oval head, dot eyes,
line mouth, scribble hair, thin black stick limbs, flat simple clothing, bold imperfect black ink,
hand-drawn texture, plain light background, muted ochre sand brown olive dusty-blue palette,
2D doodle style, no realism, no 3D, no gradients, 16:9"""

@st.cache_resource
def load_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")

def fmt(sec):
    sec = max(0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"

def clean(s):
    return re.sub(r"\s+", " ", s).strip()

# Lightweight scene detector designed for free CPU.
# It uses punctuation, sentence boundaries, connector words, and target word range
# to create meaningful visual beats instead of blindly cutting every N words.
def detect_scenes(words, min_words=4, max_words=10, max_seconds=4.8):
    scenes = []
    current = []
    visual_break_words = {
        "now","then","but","however","suddenly","instead","because","so",
        "while","when","after","before","later","eventually","finally",
        "meanwhile","one","another","next"
    }

    def flush():
        nonlocal current
        if not current:
            return
        scenes.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "words": [w["word"] for w in current]
        })
        current = []

    for idx, w in enumerate(words):
        current.append(w)
        text = w["word"].strip()
        joined = clean(" ".join(x["word"] for x in current))
        n = len(current)
        duration = w["end"] - current[0]["start"]
        next_word = words[idx+1]["word"].strip().lower() if idx+1 < len(words) else ""

        # Strong linguistic/visual break.
        strong_punct = bool(re.search(r"[.!?]$", text))
        comma_break = bool(re.search(r"[,;:]$", text)) and n >= min_words
        connector_break = next_word in visual_break_words and n >= min_words

        # Prefer meaningful scenes around 4–8 words, but allow up to 10.
        ideal_break = n >= min_words and (
            strong_punct or
            (n >= 6 and comma_break) or
            (n >= 7 and connector_break) or
            n >= max_words or
            duration >= max_seconds
        )

        if ideal_break:
            flush()

    flush()

    # Merge tiny scenes with the following scene where possible.
    merged = []
    for scene in scenes:
        n = len(scene["words"])
        if merged and n < min_words:
            prev = merged[-1]
            prev["words"].extend(scene["words"])
            prev["end"] = scene["end"]
        else:
            merged.append(scene)

    # Split very long leftovers by phrase-sized chunks.
    final = []
    for scene in merged:
        words2 = scene["words"]
        if len(words2) <= max_words:
            final.append(scene)
            continue
        # We don't have individual timings here after merging, so retain the
        # original scene; this branch mainly protects unusual transcripts.
        final.append(scene)

    for i, s in enumerate(final, 1):
        s["scene"] = i
        s["text"] = clean(" ".join(s["words"]))
    return final

def visual_prompt(text, scene_no):
    return (
        f"{STYLE}. Scene {scene_no}. Create one clear visual beat that illustrates this narration: "
        f"“{text}”. Keep the same recurring stickman character design across scenes. "
        "Show one main action or visual idea, readable silhouette, simple historical environment, "
        "wide cinematic composition, consistent scale, no text, no labels, no watermark."
    )

st.title("🎬 Voiceover → Automatic Scene Detection")
st.caption("Channel Farm-style workflow: audio timing + meaningful visual beats + image prompts")

with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    min_words = st.slider("Minimum words / scene", 3, 7, 4)
    max_words = st.slider("Maximum words / scene", 6, 14, 10)
    max_seconds = st.slider("Maximum scene duration (sec)", 2.5, 8.0, 4.8, 0.1)
    language = st.text_input("Language (optional)", "en")
    st.info("The detector uses punctuation, sentence boundaries, connectors and timing. It does NOT simply cut every N words.")

audio = st.file_uploader("🎙️ Upload voiceover", type=["mp3","wav","m4a","mpeg"])

if audio and st.button("🚀 Detect scenes", type="primary"):
    suffix = os.path.splitext(audio.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio.read())
        audio_path = f.name

    try:
        with st.spinner("Loading speech model..."):
            model = load_model(model_size)

        with st.spinner("Transcribing with word-level timing..."):
            segments, info = model.transcribe(
                audio_path,
                language=language.strip() or None,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5
            )
            words = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        t = clean(w.word)
                        if t:
                            words.append({"start": w.start, "end": w.end, "word": t})

        scenes = detect_scenes(words, min_words, max_words, max_seconds)
        st.session_state["scenes"] = scenes

        st.success(f"Detected {len(scenes)} meaningful visual scenes.")

        for s in scenes:
            st.markdown(
                f"### Scene {s['scene']:03d}  |  {fmt(s['start'])} → {fmt(s['end'])}"
            )
            st.write(s["text"])
            with st.expander("🖼️ Image prompt"):
                st.code(visual_prompt(s["text"], s["scene"]), language="text")

        txt_parts = []
        for s in scenes:
            txt_parts.append(
                f"SCENE {s['scene']:03d}\n"
                f"TIMESTAMP: {fmt(s['start'])} --> {fmt(s['end'])}\n"
                f"VOICE: {s['text']}\n"
                f"IMAGE PROMPT: {visual_prompt(s['text'], s['scene'])}\n"
            )
        txt = "\n".join(txt_parts)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["scene","start","end","duration","voice_text","image_prompt"])
        for s in scenes:
            writer.writerow([
                f"{s['scene']:03d}", fmt(s["start"]), fmt(s["end"]),
                f"{s['end']-s['start']:.2f}", s["text"],
                visual_prompt(s["text"], s["scene"])
            ])

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download scene TXT", txt,
                "channel_farm_scenes.txt", "text/plain")
        with c2:
            st.download_button("⬇️ Download scene CSV", output.getvalue(),
                "channel_farm_scenes.csv", "text/csv")

    except Exception as e:
        st.error("Processing failed.")
        st.exception(e)
    finally:
        try: os.remove(audio_path)
        except: pass
