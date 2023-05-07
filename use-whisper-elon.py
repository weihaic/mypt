import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline
import json

# Instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2",  batch_size=16)

# JIT compile the forward call - slow, but we only do once
text = pipeline("elon.mp3", task="transcribe", return_timestamps=True)

# Print text to the console
# print(text)

# Serialize the dictionary to a JSON-formatted string
text_str = json.dumps(text)

# Save the JSON string to a file
with open("output.json", "w") as output_file:
    output_file.write(text_str)

