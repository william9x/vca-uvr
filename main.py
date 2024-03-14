import logging
from pathlib import Path

from audio_separator.separator import Separator
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

separator = Separator(
    model_file_dir="models",
    output_dir="output",
    mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.999, "batch_size": 4, "enable_denoise": True},
)
separator.load_model("Kim_Vocal_2.onnx")
app = FastAPI()


class UvrInferReq(BaseModel):
    input_path: Path


@app.post("/api/v1/uvr/infer", tags=["Infer"], response_class=JSONResponse)
async def uvr_infer(req: UvrInferReq) -> JSONResponse:
    print(f"received request: {req}")

    separator.separate(req.input_path)

    return JSONResponse(content={"message": "Created"}, status_code=201)
