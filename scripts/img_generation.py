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
        choices=[
            "full",
            "lora",
            "svdiff",
            "from_scratch",
            "attention",
            "svdiff_attention",
            "lora_attention",
        ],
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
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


def generate_images(
    pipeline,
    accelerator,
    num_images_to_generate,    
    batch_size,
    encoder_hidden_states,
    resolution,
    class_label=None,
    num_inference_steps=100,
    seed=42,
):
    device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    with torch.autocast(device_type=device.type):

        num_batches = num_images_to_generate // batch_size + (
            num_images_to_generate % batch_size != 0
        )
        
        print("Generating images...")
        images = []
        for j in range(num_batches):
            images_batch = pipeline(
                encoder_hidden_states=encoder_hidden_states,
                class_label=class_label,
                num_inference_steps=num_inference_steps,
                num_images=batch_size,
                generator=generator,
                height=resolution,
                width=resolution,
            ).images
                
            images.extend(images_batch)

    return images


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

    encoder_hidden_states = torch.zeros(
        [
            args.batch_size,
            encoder_max_position_embeddings,
            encoder_hidden_size,
        ],
        dtype=accelerator.unwrap_model(pipeline_inversion.unet).dtype,
    ).to(accelerator.device)

    generated_images_0 = generate_images(
        pipeline_inversion,
        accelerator,
        args.num_images_per_class,
        args.batch_size,
        encoder_hidden_states,
        args.resolution,
        class_label=0,
    )

    generated_images_1 = generate_images(
        pipeline_inversion,
        accelerator,
        args.num_images_per_class,
        args.batch_size,
        encoder_hidden_states,
        args.resolution,
        class_label=1,
    )

    save_images(generated_images_0, args.output_dir[0])
    save_images(generated_images_1, args.output_dir[1])


if __name__ == "__main__":
    main()
