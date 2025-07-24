import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_image.py 'your prompt here'")
        return
    prompt = sys.argv[1]

    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Generate image
    image = pipe(prompt).images[0]

    # Save and show image
    image.save("generated_image.png")
    image.show()
    print("Image saved as generated_image.png")

if __name__ == "__main__":
    main() 