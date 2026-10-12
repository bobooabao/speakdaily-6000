import json, pathlib, subprocess, tempfile
import numpy as np
import soundfile as sf
from kokoro import KPipeline
root=pathlib.Path(__file__).resolve().parent
items=json.loads((root/'sentences.json').read_text(encoding='utf-8'))
out=root/'audio';out.mkdir(exist_ok=True)
pipe=KPipeline(lang_code='a')
for item in items:
    target=out/f"{item['id']:02d}.mp3"
    if target.exists():continue
    parts=[np.asarray(audio,dtype=np.float32) for _,_,audio in pipe(item['en'],voice='am_michael',speed=1.0)]
    if not parts:raise RuntimeError(f"No audio for {item['id']}")
    with tempfile.TemporaryDirectory() as td:
        wav=pathlib.Path(td)/'voice.wav'
        sf.write(wav,np.concatenate(parts),24000)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-codec:a','libmp3lame','-b:a','128k',str(target)],check=True)
    print(target)
