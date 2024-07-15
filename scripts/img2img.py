import argparse
import torch
from transformers import CLIPTextModel
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
from PIL import Image
from pipelines.pipeline_stable_diffusion_ddim_inversion import (
    StableDiffusionPipelineWithDDIMInversion,
)
from accelerate import Accelerator
import os
from utils import load_unet_custom
from diffusers.utils import is_wandb_available

if is_wandb_available():
    import wandb


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
        "--weights_path",
        type=str,
        default="checkpoint",
        help="The checkpoint directory that will be used to load the unet weights.",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Original class label of the images that will be used for translation.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Dir where there are the images that will be used for translation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="+",
        default="output",
        help="The output directory where the generated images will be written.",
    )
    parser.add_argument(
        "--num_images_per_class",
        type=int,
        default=20,
        help="Number of images generated for validation.",
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
    parser.add_argument(
        "--upload_images",
        action="store_true",
        help="Whether or not to upload images to wandb.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
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

    # Load the Stable Diffusion inversion pipeline
    pipeline_inversion = StableDiffusionPipelineWithDDIMInversion.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
    )

    # Move the inversion pipeline to the accelerator device(s)
    pipeline_inversion = accelerator.prepare(pipeline_inversion)

    # Load text encoder config
    text_encoder_config = CLIPTextModel.config_class.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    encoder_max_position_embeddings = text_encoder_config.max_position_embeddings
    encoder_hidden_size = text_encoder_config.hidden_size

    return (
        pipeline_inversion,
        accelerator,
        encoder_max_position_embeddings,
        encoder_hidden_size,
    )


def generate_translated_image(
    pipeline,
    accelerator,
    images,
    encoder_hidden_states,
    class_label=None,
    num_inference_steps=100,
    seed=42,
):
    device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    with torch.autocast(device_type=device.type):
        # Generate the latents
        print("Generating latents...")
        inv_latents = pipeline.invert(
            class_label=class_label,
            encoder_hidden_states=encoder_hidden_states,
            image=images,
            num_inference_steps=num_inference_steps,
            num_images=len(images),
            generator=generator,
        )

        print("Generating images...")
        image = pipeline(
            encoder_hidden_states=encoder_hidden_states,
            class_label=1 - class_label,
            num_inference_steps=num_inference_steps,
            num_images=len(images),
            generator=generator,
            height=images[0].height,
            width=images[0].width,
            latents=inv_latents,
        ).images

    return image


def save_images(images, output_dir):
    # Create a outputs directory
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(img_path)


def main():
    args = parse_args()

    (
        pipeline_inversion,
        accelerator,
        encoder_max_position_embeddings,
        encoder_hidden_size,
    ) = load_model(args)

    if args.upload_images:
        # Init wandb
        wandb.init(
            project="svdiff_img2img",
            name=f"{args.experiment_name}",
            config={
                "method": args.finetunning_method,
            },
        )

    original_images = []
    translated_images = []

    image_names = os.listdir(args.data_dir)

    # Define batch size
    batch_size = min(len(image_names), args.num_images_per_class)

    # Calculate the number of batches
    num_batches = len(image_names) // batch_size + (len(image_names) % batch_size != 0)

    # Prepare empty text encoder hidden states
    encoder_hidden_states = torch.zeros(
        [
            batch_size,
            encoder_max_position_embeddings,
            encoder_hidden_size,
        ],
        dtype=accelerator.unwrap_model(pipeline_inversion.unet).dtype,
    ).to(accelerator.device)

    for i in range(num_batches):
        batch_images = []
        for j in range(batch_size):
            if i * batch_size + j < len(image_names):
                image = Image.open(
                    os.path.join(args.data_dir, image_names[i * batch_size + j])
                )
                batch_images.append(image)

        print(f"Generating images for batch {i}...")
        translated_image_batch = generate_translated_image(
            pipeline_inversion,
            accelerator,
            batch_images,
            encoder_hidden_states,
            class_label=args.class_label,
        )

        original_images.extend(batch_images)
        translated_images.extend(translated_image_batch)

    if args.upload_images:
        wandb.log(
            {
                "original_images": [
                    wandb.Image(
                        image,
                        caption=f"{i}",
                    )
                    for i, image in enumerate(original_images)
                ],
                "translated_images": [
                    wandb.Image(
                        image,
                        caption=f"{i}",
                    )
                    for i, image in enumerate(translated_images)
                ],
            }
        )
        wandb.finish()
    else:
        save_images(original_images, args.output_dir[0])
        save_images(translated_images, args.output_dir[1])


if __name__ == "__main__":
    main()
