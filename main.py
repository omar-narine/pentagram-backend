import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os


def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=torch.float16,
        variant='fp16'
    )


image = (modal.Image.debian_slim()
         .pip_install('fastapi[standard]', 'transformers', 'accelerate', 'diffusers', 'requests')
         .run_function(download_model))

app = modal.App('pentagram', image=image)


@app.cls(
    image=image,
    gpu='A10G',
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name('API_KEY')]
)
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        # Defining the model again
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=torch.float16,
            variant='fp16'
        )

        self.pipe.to('cuda')  # Telling Modal to run on GPU
        self.API_KEY = os.environ['API_KEY']

    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description='The prompt for image generation')):
        '''
        Prompt is a query parameter within the url
        '''

        api_key = request.headers.get('X-API-KEY')
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail='Unauthorized'
            )

        image = self.pipe(prompt, num_inference_steps=1,
                          guidance_scale=0.0).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')

        return Response(content=buffer.getvalue(), media_type='image/jpeg')

    @modal.web_endpoint()
    def health(self):
        '''
        Endpoint to keep container warm and latency down
        '''
        return {'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}


@app.function(
    schedule=modal.Cron('*/60 * * * *'),
    secrets=[modal.Secret.from_name('API_KEY')]
)
def keep_warm():
    health_url = 'https://omar-narine--pentagram-model-health.modal.run'
    generate_url = 'https://omar-narine--pentagram-model-generate.modal.run'

    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    headers = {'X-API-KEY': os.environ['API_KEY']}
    generate_response = requests.get(generate_url, headers=headers)
    print(
        f"Generate endpoint tested successfully at: {[datetime.now(timezone.utc).isoformat()]}")
