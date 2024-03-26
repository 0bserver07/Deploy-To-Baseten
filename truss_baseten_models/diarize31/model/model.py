import os
import tempfile
from typing import Dict

import requests
from pyannote.audio import Pipeline

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self.pipeline = None

    def load(self):
        # Load the HuggingFace speaker diarization pipeline
        HUGGINGFACE_ACCESS_TOKEN = self._secrets["hf_access_token"]
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_ACCESS_TOKEN
        )

    def preprocess(self, request: Dict) -> Dict:
        audio_url = request["url"]

        # Stream the download for large files
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()  # Ensure the download succeeded
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            for chunk in r.iter_content(chunk_size=8192):
                temp_audio_file.write(chunk)
            temp_audio_file.close()

        return {"audio_file_path": temp_audio_file.name}

    def predict(self, request: Dict) -> Dict:
        audio_file_path = request["audio_file_path"]

        # Run the diarization pipeline
        diarization_result = self.pipeline(audio_file_path, min_speakers=2, max_speakers=5)

        # Serialize the diarization result to a format that can be converted to JSON
        # Assuming diarization_result is an instance of pyannote.core.Annotation
        result_data = {
            "segments": [
                {
                    "speaker": label,
                    "start": segment.start,
                    "end": segment.end,
                }
                for segment, _, label in diarization_result.itertracks(yield_label=True)
            ]
        }

        # Clean up the temporary file
        os.remove(audio_file_path)

        return {"diarization_result": result_data}


