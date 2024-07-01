import argparse
import torch
import numpy as np
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
import pandas as pd
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoConfig
import os
import shutil
from PIL import Image
from torchvision import models, transforms
from scipy.linalg import sqrtm


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=20,
        help="Number of images generated for validation.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="checkpoint",
        help="The checkpoint directory that will be used to load the unet weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the generated images will be written.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_model(pretrained_path, weights_path, revision):

    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )

    # Load the text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_path,
        subfolder="text_encoder",
        revision=revision,
    )

    # Load the scheduler
    scheduler = DDIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")

    # Load the VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_path, subfolder="vae", revision=revision
    )

    # Load the UNet configuration
    unet = UNet2DConditionModel.from_pretrained(weights_path)

    # Move Unet, Vae and text_encoder to the correspondent device
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Set the unet, VAE and text_encode into eval
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        tokenizer=tokenizer,
        scheduler=scheduler,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=revision,
    )

    # Move the pipeline to the accelerator device(s)
    pipeline = accelerator.prepare(pipeline)

    return pipeline, accelerator


def generate_image(
    pipeline,
    accelerator,
    prompt,
    num_images_per_prompt,
    num_inference_steps=100,
    seed=42,
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
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            height=128,
            width=128,
        ).images

    return images


# Image preprocessing function
def load_and_preprocess_image(img_path):
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_activations(image_folder, model):
    activations = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        if img_path.endswith(("png", "jpg", "jpeg")):
            img = load_and_preprocess_image(img_path)
            with torch.no_grad():
                act = model(img).squeeze().numpy()
            activations.append(act)
    activations = np.array(activations)
    return activations


def main():
    args = parse_args()

    pipeline, accelerator = load_model(
        args.pretrained_model_name_or_path, args.weights_path, args.revision
    )

    images = generate_image(pipeline, accelerator, "", args.num_images_per_prompt)

    # Create a outputs directory
    os.makedirs(args.output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = os.path.join(args.output_dir, f"image_{i}.png")
        img.save(img_path)

    # Load pre-trained InceptionV3 model + higher level layers
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove the last layer
    inception.eval()

    # Load datasets
    path2 = "/projects/static2dynamic/Biel/stablediffusion_nano/data/data/train/DMSO"
    # Get activations
    act1 = get_activations(args.output_dir, inception)
    act2 = get_activations(path2, inception)

    # Calculate FID
    fid = calculate_fid(act1, act2)
    print("FID score:", fid)


if __name__ == "__main__":
    main()
