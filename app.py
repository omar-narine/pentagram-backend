import io
import random
import time
from pathlib import Path
import modal

from diffusers import DiffusionPipeline

hf_pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-3.5-large')

app = modal.App('pentagram-image-gen')

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

with image.imports():
    from diffusers import DiffusionPipeline
    import torch
    from fastapi import Response


@app.cls(
    image=image,
    gpu='A10G',
    timeout=10 * MINUTES,
)
class Inference:
    @modal.build()
    @modal.enter()
    def initalize(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-3.5-large')

    @modal.enter()
    def move_to_gpu(self):
        self.pipe.to('cuda')


@app.function(gpu='A10G')
def something():
    return 'Something'
