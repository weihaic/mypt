

from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

# JIT compile the forward call - slow, but we only do once
text = pipeline("audio.wav")

# used cached function thereafter - super fast!!
text = pipeline("audio.wav")

print(text)
