import logging
from pathlib import Path

from audio_separator.separator import Separator
from fastapi import FastAPI
from moviepy.editor import *
from pydantic import BaseModel
from pytube import YouTube
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

separator = Separator(
    model_file_dir="models",
    output_dir="output",
    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.5, "batch_size": 16, "enable_denoise": True},
)
separator.load_model("Kim_Vocal_2.onnx")
app = FastAPI()


class UvrInferReq(BaseModel):
    task_id: str
    youtube_path: str
    output_path: str
    input_path: Path


@app.post("/api/v1/uvr/infer", tags=["Infer"], response_class=JSONResponse)
async def uvr_infer(req: UvrInferReq) -> JSONResponse:
    print(f"received request: {req}")

    if req.youtube_path is None:
        separator.separate(req.input_path)

    yt = YouTube(req.youtube_path)
    yt = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').last()
    if yt is None:
        return JSONResponse(content={"message": "No downloadable mp4 from provided URL"}, status_code=400)

    downloaded_file = yt.download(output_path=req.output_path)
    VideoFileClip(downloaded_file).audio.write_audiofile("downloads/out.mp3")
    separator.separate("downloads/out.mp3")

    return JSONResponse(content={"message": "Created"}, status_code=201)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
