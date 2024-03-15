import logging
from pathlib import Path

from audio_separator.separator import Separator
from fastapi import FastAPI
from moviepy.editor import *
from pydantic import BaseModel
from pytube import YouTube
from starlette.responses import JSONResponse

DOWNLOAD_PATH = os.getenv("UVR_DOWNLOAD_PATH")
if DOWNLOAD_PATH is None:
    DOWNLOAD_PATH = "audio/uvr_downloads"

PROCESSED_PATH = os.getenv("UVR_PROCESSED_PATH")
if PROCESSED_PATH is None:
    PROCESSED_PATH = "audio/uvr_processed"

MODEL_PATH = os.getenv("UVR_MODEL_PATH")
if MODEL_PATH is None:
    MODEL_PATH = "UVR-MDX-NET-Inst_full_292.onnx"

logger = logging.getLogger(__name__)

separator = Separator(
    model_file_dir="models",
    output_dir=PROCESSED_PATH,
    output_format="mp3",
    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 24, "enable_denoise": True},
)
separator.load_model(MODEL_PATH)
app = FastAPI()

hash = separator.get_model_hash(f"models/{MODEL_PATH}")
print(f"Model hash: {hash}")


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
        yt = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').last()
        if yt is None:
            return JSONResponse(content={"message": "No downloadable mp4 from provided URL"}, status_code=400)

        file_prefix = req.task_id.replace("-", "_")
        downloaded_file = yt.download(output_path=DOWNLOAD_PATH, filename=f"{file_prefix}.mp4")
        VideoFileClip(downloaded_file).audio.write_audiofile(f"{DOWNLOAD_PATH}/{file_prefix}.mp3")

        process_files = separator.separate(f"{DOWNLOAD_PATH}/{file_prefix}.mp3")
        return JSONResponse(content={"message": "Created", "file": f"{process_files[0]}"}, status_code=201)
    except Exception as e:
        return JSONResponse(content={"message": f"UVR error {e}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
