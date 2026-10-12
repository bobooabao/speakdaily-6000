import json, pathlib, os, subprocess, tempfile, sys
root = pathlib.Path(__file__).resolve().parents[1]
base = root / "quality-v3"
sources = ["sentences.json", "extra-01.json", "extra-02.json", "extra-03.json"]
items = []
for name in sources:
    items.extend(json.loads((base / name).read_text(encoding="utf-8")))
dialogues = json.loads((base / "dialogues.json").read_text(encoding="utf-8"))
essentials = json.loads((base / "essentials.json").read_text(encoding="utf-8"))
jobs = [(f"s{i:04d}", x["en"]) for i, x in enumerate(items, 1)]
jobs += [(f"d{i:03d}-{j}", t["en"]) for i, d in enumerate(dialogues, 1) for j, t in enumerate(d["turns"], 1)]
jobs += [(f"e{i:03d}", x["en"]) for i, x in enumerate(essentials, 1)]
out = base / "audio"
out.mkdir(exist_ok=True)
pending = [(key, text) for key, text in jobs if not (out / (key + ".mp3")).exists() or (out / (key + ".mp3")).stat().st_size == 0]
if "--status" in sys.argv:
    print(f"total={len(jobs)}")
    print(f"ready={len(jobs)-len(pending)}")
    print(f"pending={len(pending)}")
    raise SystemExit(0)
if not pending:
    print("All audio files complete", flush=True)
    raise SystemExit(0)
import numpy as np
import soundfile as sf
from kokoro import KPipeline
limit = int(os.getenv("BATCH_SIZE", "30"))
print(f"Ready {len(jobs)-len(pending)}/{len(jobs)}; generating up to {min(limit,len(pending))}", flush=True)
pipe = KPipeline(lang_code="a")
for n, (key, text) in enumerate(pending[:limit], 1):
    target = out / (key + ".mp3")
    chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipe(text, voice="am_michael", speed=1.0)]
    if not chunks:
        raise RuntimeError(f"Empty output for {key}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / "voice.wav"
        sf.write(wav, np.concatenate(chunks), 24000)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "80k", str(target)], check=True)
    if n % 10 == 0:
        print(f"Generated {n}/{min(limit,len(pending))}", flush=True)
