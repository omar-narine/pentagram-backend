import io
import random
import time
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-3.5-large')
