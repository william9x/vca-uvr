import os
from pathlib import Path

import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/liamhvn/voice-models/resolve/main/rvc"

BASE_DIR = Path(__file__).resolve().parent.parent


def dl_model(link, dir_name, output_file):
    with requests.get(f"{link}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / output_file), exist_ok=True)
        with open(dir_name / output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    rvc_models_dir = BASE_DIR / "assets/weights/"
    model_paths = [
        "kim_jong_un",
        "putin",
        "mickey_mouse",
        "batman",
        "messi",
        "godzilla",
    ]
    for model in model_paths:
        print(f"Downloading {model}...")
        dl_model(f"{RVC_DOWNLOAD_LINK}/{model}.pth", rvc_models_dir, f"{model}.pth")
        dl_model(f"{RVC_DOWNLOAD_LINK}/{model}.index", rvc_models_dir, f"{model}.index")

    print("All models downloaded!")
