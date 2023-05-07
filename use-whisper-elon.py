

from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-base",  dtype=jnp.bfloat16, batch_size=16)

# JIT compile the forward call - slow, but we only do once
text = pipeline("elon.mp3",task="transcribe", return_timestamps=True)

# used cached function thereafter - super fast!!


print(text)
