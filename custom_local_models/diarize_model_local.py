import os
from pyannote.core import Annotation

HUGGINGFACE_ACCESS_TOKEN = os.environ["HUGGINGFACE_ACCESS_TOKEN"]

# Instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_ACCESS_TOKEN
)

audio_file_path = "LINK_TO_AUDIO_FILE.mp3"


# Run the pipeline on an audio file
diarization = pipeline(audio_file_path, min_speakers=2, max_speakers=5)

# Adjust the diarization object URI for RTTM output
safe_uri = os.path.basename(audio_file_path).replace(" ", "_")
if hasattr(diarization, 'uri'):
    diarization.uri = safe_uri
else:
    # yeah lets make sure it doesn't fail silently
    print("Adapt this step based on the actual structure of the diarization object.")

# Dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
