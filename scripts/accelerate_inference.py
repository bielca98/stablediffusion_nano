import torch
import json
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, DDIMScheduler
from accelerate import Accelerator
from transformers import AutoTokenizer

PRETRAINED_PATH = "stabilityai/stable-diffusion-2-1-base"


def load_model():
    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_PATH,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_PATH, subfolder="text_encoder"
    )

    scheduler = DDIMScheduler.from_pretrained(PRETRAINED_PATH, subfolder="scheduler")

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_PATH,
        torch_dtype=torch.float16,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    ).to(accelerator.device)

    # Move the pipeline to the accelerator device(s)
    pipeline = accelerator.prepare(pipeline)

    return pipeline, accelerator


def generate_image(
    pipeline, accelerator, prompt, num_inference_steps=50, guidance_scale=9.0, seed=42
):
    device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate the image
    with torch.autocast(device_type=device.type):
        images = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=90,
            generator=generator,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        ).images

    return images


def main():
    # Load the model and accelerator
    pipeline, accelerator = load_model()
    print(accelerator.device)
    # Define the prompt
    prompt = "a professional photograph of an astronaut riding a horse"

    # Generate the image
    images = generate_image(pipeline, accelerator, prompt)

    # Save the image
    # for i, image in enumerate(images):
    #    image.save(f"outputs/delete/generated_image_{i}.png")
    print(f"Saved {len(images)} images.")


if __name__ == "__main__":
    main()
