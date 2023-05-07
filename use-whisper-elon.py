import jax.numpy as jnp

from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-base",  dtype=jnp.bfloat16, batch_size=16)

# JIT compile the forward call - slow, but we only do once
text = pipeline("elon.mp3",task="transcribe", return_timestamps=True)

# used cached function thereafter - super fast!!


#print(text)


# save text to a file
with open("output.txt", "w") as output_file:
    output_file.write(text)
