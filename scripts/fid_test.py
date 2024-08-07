import argparse
import torch
import shutil

import numpy as np
from transformers import CLIPTextModel
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
from pipelines.pipeline_stable_diffusion_class_conditioned import (
    ClassConditionedStableDiffusionPipeline,
)
from accelerate import Accelerator
import os
from PIL import Image
from torchvision import transforms
from scipy.linalg import sqrtm
from utils import load_unet_custom
import torch_fidelity


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
        "--num_images_per_iteration",
        type=int,
        default=20,
        help="Number of images generated during each pipeline iteration.",
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
        nargs="+",
        default="output",
        help="The output directory where the generated images will be written.",
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        nargs="+",
        default="data/data/train/DMSO",
        help="The images directory where the original images are.",
    )
    parser.add_argument(
        "--finetunning_method",
        type=str,
        default=None,
        choices=["full", "lora", "svdiff", "from_scratch", "attention"],
        help=(
            "Finetunning method that will be used to adapt the model to the new dataset."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_model(args):

    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the scheduler
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Load the VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    # Load the UNet configuration
    class_conditioning = len(args.output_dir) > 1
    unet = load_unet_custom(
        args.pretrained_model_name_or_path,
        args.weights_path,
        args.revision,
        subfolder="unet",
        method=args.finetunning_method,
        class_conditioning=class_conditioning,
    )

    # Move Unet and Vae to the correspondent device
    unet.to(accelerator.device)
    vae.to(accelerator.device)

    # Set unet and VAE into eval
    unet.eval()
    vae.eval()

    # Load the Stable Diffusion pipeline
    pipeline = ClassConditionedStableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
    )

    # Move the pipeline to the accelerator device(s)
    pipeline = accelerator.prepare(pipeline)

    # Load text encoder config
    text_encoder_config = CLIPTextModel.config_class.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    encoder_max_position_embeddings = text_encoder_config.max_position_embeddings
    encoder_hidden_size = text_encoder_config.hidden_size

    # Prepare empty text encoder hidden states
    encoder_hidden_states = torch.zeros(
        [
            args.num_images_per_iteration,
            encoder_max_position_embeddings,
            encoder_hidden_size,
        ],
        dtype=accelerator.unwrap_model(unet).dtype,
    ).to(accelerator.device)

    return pipeline, accelerator, encoder_hidden_states


def generate_image(
    pipeline,
    accelerator,
    encoder_hidden_states,
    num_images,
    class_label=None,
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
            encoder_hidden_states=encoder_hidden_states,
            class_label=class_label,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
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


def save_images(images, output_dir):
    # Create a outputs directory
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(img_path)


def main():
    args = parse_args()

    pipeline, accelerator, encoder_hidden_states = load_model(args)

    if len(args.output_dir) > 1:
        for i in range(len(args.output_dir)):
            print("Generating images with image label " + str(i))

            # num_class_images = len(os.listdir(args.original_dir[i]))
            num_class_images = 200
            batch_size = args.num_images_per_iteration
            num_batches = num_class_images // batch_size + (
                num_class_images % batch_size != 0
            )

            images = []
            for j in range(num_batches):
                batch_images = generate_image(
                    pipeline,
                    accelerator,
                    encoder_hidden_states,
                    batch_size,
                    class_label=i,
                    seed=j,
                )
                images.extend(batch_images)
            save_images(images, args.output_dir[i])

            metrics_dict = torch_fidelity.calculate_metrics(
                input1=args.output_dir[i],
                input2=args.original_dir[i],
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                prc=False,
            )
            print(metrics_dict)

            # Remove the output directory if it exists
            if os.path.exists(args.output_dir[i]):
                shutil.rmtree(args.output_dir[i])
    else:
        print("Generating images")
        images = generate_image(
            pipeline, accelerator, encoder_hidden_states, args.num_images_per_iteration
        )
        save_images(images, args.output_dir[0])


if __name__ == "__main__":
    main()
