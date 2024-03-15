import logging
from pathlib import Path

from audio_separator.separator import Separator
from fastapi import FastAPI
from moviepy.editor import *
from pydantic import BaseModel
from pytube import YouTube
from starlette.responses import JSONResponse


def get_env(key, default_val=None):
    val = os.getenv(key)
    if val is None:
        return default_val
    return val


DOWNLOAD_PATH = get_env("UVR_DOWNLOAD_PATH", "audio/uvr_downloads")
PROCESSED_PATH = get_env("UVR_PROCESSED_PATH", "audio/uvr_processed")
MODEL_PATH = get_env("UVR_MODEL_PATH", "Kim_Vocal_2.onnx")
VIDEO_EXT = get_env("UVR_VIDEO_EXT", "mp4")
AUDIO_EXT = get_env("UVR_VIDEO_EXT", "mp3")

logger = logging.getLogger(__name__)

separator = Separator(
    model_file_dir="models",
    output_dir=PROCESSED_PATH,
    output_format=AUDIO_EXT,
    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 24, "enable_denoise": True},
)
separator.load_model(MODEL_PATH)
app = FastAPI()


class UvrInferReq(BaseModel):
    task_id: str
    youtube_path: str
    input_path: Path


@app.post("/api/v1/uvr/infer", tags=["Infer"], response_class=JSONResponse)
async def uvr_infer(req: UvrInferReq) -> JSONResponse:
    print(f"received request: {req}")

    try:
        if req.youtube_path is None:
            separator.separate(req.input_path)

        yt = YouTube(req.youtube_path)
        yt = yt.streams.filter(progressive=True, file_extension=VIDEO_EXT).order_by('resolution').last()
        if yt is None:
            return JSONResponse(content={"message": f"No downloadable {VIDEO_EXT} from provided URL"}, status_code=400)

        file_prefix = req.task_id.replace("-", "_")

        downloaded_file = yt.download(output_path=DOWNLOAD_PATH, filename=f"{file_prefix}.{VIDEO_EXT}")

        audio_file_path = f"{DOWNLOAD_PATH}/{file_prefix}.{AUDIO_EXT}"
        VideoFileClip(downloaded_file).audio.write_audiofile(audio_file_path)

        processed_files = separator.separate(audio_file_path)

        return JSONResponse(content={"message": "Created", "file": f"{processed_files[0]}"}, status_code=201)
    except Exception as e:
        return JSONResponse(content={"message": f"UVR error {e}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
