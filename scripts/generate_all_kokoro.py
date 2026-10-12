import json, pathlib, re, subprocess, tempfile, os
import numpy as np
import soundfile as sf
from kokoro import KPipeline
root=pathlib.Path(__file__).resolve().parents[1]
html=(root/'index.html').read_text(encoding='utf-8')
match=re.search(r'const DATA=(\[.*?\]);\s*const \$=',html,re.S)
if not match: raise RuntimeError('Cannot locate original 6000 sentences in index.html')
items=[x for group in json.loads(match.group(1)) for x in group['items']]
assert len(items)==6000, f'Expected 6000, found {len(items)}'
out=root/'audio';out.mkdir(exist_ok=True)
limit=int(os.getenv('BATCH_SIZE','120'))
pending=[x for x in items if not (out/f"{x['id']:04d}.mp3").exists()]
print(f'Existing: {len(items)-len(pending)}; pending: {len(pending)}; generating: {min(limit,len(pending))}',flush=True)
if not pending:quit()
pipe=KPipeline(lang_code='a')
for i,x in enumerate(pending[:limit],1):
    target=out/f"{x['id']:04d}.mp3"
    chunks=[np.asarray(audio,dtype=np.float32) for _,_,audio in pipe(x['en'],voice='am_michael',speed=1.0)]
    if not chunks: raise RuntimeError(f'Empty audio for sentence {x["id"]}')
    with tempfile.TemporaryDirectory() as tmp:
        wav=pathlib.Path(tmp)/'voice.wav'
        sf.write(wav,np.concatenate(chunks),24000)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-codec:a','libmp3lame','-b:a','96k',str(target)],check=True)
    if i%20==0:print(f'Generated {i}/{min(limit,len(pending))} in this batch',flush=True)
