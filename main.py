import logging

from audio_separator.separator import Separator
from fastapi import FastAPI
from moviepy.editor import *
from pydantic import BaseModel
from starlette.responses import JSONResponse


def get_env(key, default_val=None):
    val = os.getenv(key)
    if val is None:
        return default_val
    return val


PROCESSED_PATH = get_env("UVR_PROCESSED_PATH", "audio/save_uvr")
MODEL_PATH = get_env("UVR_MODEL_PATH", "Kim_Vocal_2.onnx")
VIDEO_EXT = get_env("UVR_VIDEO_EXT", ".mp4")
AUDIO_EXT = get_env("UVR_VIDEO_EXT", ".mp3")

logger = logging.getLogger(__name__)

separator = Separator(
    model_file_dir="models",
    output_dir=PROCESSED_PATH,
    output_format="mp3",
    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 24, "enable_denoise": True},
)
separator.load_model(MODEL_PATH)
app = FastAPI()


class UvrInferReq(BaseModel):
    input_path: str


class UvrInferResp(BaseModel):
    out_vocal: str
    out_instr: str


@app.post("/api/v1/uvr/infer", tags=["Infer"], response_class=JSONResponse)
async def uvr_infer(req: UvrInferReq) -> JSONResponse:
    print(f"received request: {req}")

    try:
        audio_file_path = req.input_path
        if audio_file_path.endswith(VIDEO_EXT):
            audio_file_path = audio_file_path.replace(VIDEO_EXT, AUDIO_EXT)
            VideoFileClip(req.input_path).audio.write_audiofile(audio_file_path)

        processed_files = separator.separate(audio_file_path)
        if len(processed_files) != 2:
            raise ValueError("process audio failed")

        resp = UvrInferResp(
            output_instrument_path=f"{PROCESSED_PATH}{processed_files[0]}",
            output_vocal_path=f"{PROCESSED_PATH}{processed_files[1]}"
        )

        return JSONResponse(
            content={"output_vocal_path": resp.out_vocal, "output_instrument_path": resp.out_instr},
            status_code=201)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"message": f"UVR error {e}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
