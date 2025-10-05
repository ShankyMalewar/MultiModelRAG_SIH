# test_asr.py
from faster_whisper import WhisperModel
import time
p = "data/vault/sample.mp3"

print("Loading model (tiny) ...")
t0 = time.time()
model = WhisperModel("tiny", device="cpu")   # tiny is fast for a smoke test
print("Loaded in", time.time()-t0, "s")

print("Transcribing", p)
segments, info = model.transcribe(p, beam_size=2, vad_filter=True)
print("Detected language:", info.language, " duration:", info.duration)
print("Segments (first 3):")
for i,s in enumerate(segments):
    print(i, s.start, s.end, s.text[:200])
    if i>=2:
        break
print("Done.")
