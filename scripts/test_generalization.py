import argparse
import torch
from transformers import CLIPTextModel
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
from pipelines.pipeline_stable_diffusion_ddim_inversion import (
    StableDiffusionPipelineWithDDIMInversion,
)
from accelerate import Accelerator
import os
from utils import load_unet_custom
from diffusers.utils import is_wandb_available
from utils import DownStreamDataset, collate_fn, pil_to_tensor, tensor_to_pil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io

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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="checkpoint",
        help="The checkpoint directory that will be used to load the unet weights.",
    )
    parser.add_argument(
        "--second_weights_path",
        type=str,
        default="checkpoint",
        help=(
            "The checkpoint directory that will be used to load the unet weights of the second model, trained on a"
            " different dataset and that will be used to compare the generated images and the generalization capacity"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Dir where there are the images that will be used for translation.",
    )
    parser.add_argument(
        "--use_local_checkpoints",
        action="store_true",
        help="Whether to use local checkpoints or download them from huggingface.",
    )
    parser.add_argument(
        "--num_images_per_class",
        type=int,
        default=20,
        help="Number of images generated for validation.",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Label of the class that will be used to generate the images.",
    )
    parser.add_argument(
        "--dataloader_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the  dataloader.",
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
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--n_close_images_to_upload",
        type=int,
        default=10,
        help=("Number of top closest images that will be uploaded as an example."),
    )
    parser.add_argument(
        "--data_samples",
        type=int,
        default=10,
        help="Number of samples that will be used from the indicated dataset.",
    )
    parser.add_argument(
        "--data_sampling_seed",
        type=int,
        default=43,
        help="Random seed that will be used to sample from the original dataset.",
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
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        local_files_only=args.use_local_checkpoints,
    )

    # Load the UNet configuration
    unet = load_unet_custom(
        args.pretrained_model_name_or_path,
        args.weights_path,
        args.revision,
        subfolder="unet",
        method=args.finetunning_method,
        class_conditioning=True,
        is_local_checkpoint=args.use_local_checkpoints,
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
        local_files_only=args.use_local_checkpoints,
    )
    encoder_max_position_embeddings = text_encoder_config.max_position_embeddings
    encoder_hidden_size = text_encoder_config.hidden_size

    # Dataset and DataLoaders creation:
    train_dataset = DownStreamDataset(
        data_root=[args.data_dir],
        size=args.resolution,
        center_crop=args.center_crop,
        data_samples=args.data_samples,
        data_sampling_seed=args.data_sampling_seed,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.dataloader_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Load UNET for the second model, trained on a different dataset and that will be used to compare the generated
    # images and the generalization capacity
    second_unet = load_unet_custom(
        args.pretrained_model_name_or_path,
        args.second_weights_path,
        args.revision,
        subfolder="unet",
        method=args.finetunning_method,
        class_conditioning=True,
        is_local_checkpoint=args.use_local_checkpoints,
    )

    # Move Unet to the correspondent device
    second_unet.to(accelerator.device)

    # Set unet into eval
    second_unet.eval()

    # Load the Stable Diffusion inversion pipeline
    second_pipeline_inversion = (
        StableDiffusionPipelineWithDDIMInversion.from_pretrained(
            args.pretrained_model_name_or_path,
            scheduler=scheduler,
            unet=accelerator.unwrap_model(second_unet),
            vae=vae,
            revision=args.revision,
        )
    )

    # Move the inversion pipeline to the accelerator device(s)
    pipeline_inversion = accelerator.prepare(pipeline_inversion)

    return (
        pipeline_inversion,
        second_pipeline_inversion,
        dataloader,
        accelerator,
        encoder_max_position_embeddings,
        encoder_hidden_size,
    )


def cos_similarity(im1, im2):
    """Compute cosine similarity directly between two images."""
    norm_im1 = im1 / im1.norm(dim=(1, 2), keepdim=True).norm(dim=0, keepdim=True)
    norm_im2 = im2 / im2.norm(dim=(1, 2), keepdim=True).norm(dim=0, keepdim=True)
    flattened_im1 = norm_im1.flatten(start_dim=0)
    flattened_im2 = norm_im2.flatten(start_dim=0)
    return torch.matmul(flattened_im1, flattened_im2.T)


def combine_images(image1, image2, label1="Generated", label2="Real"):
    # Assumes image1 and image2 are PIL Image objects and have the same dimensions
    combined = Image.new("RGB", (image1.width + image2.width, image1.height))
    draw = ImageDraw.Draw(combined)

    # Use a default font
    font = ImageFont.load_default()

    # Add labels to the images
    draw.text((10, 10), label1, font=font)
    draw.text((image1.width + 10, 10), label2, font=font)

    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))
    return combined


def generate_and_find_closest_images_direct(
    pipeline,
    second_pipeline,
    accelerator,
    dataloader,
    n_images,
    encoder_hidden_states,
    class_label=None,
    num_inference_steps=100,
    seed=42,
):
    device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    # Save the initial state of the generator
    initial_state = generator.get_state()


    # Generate images using the pipeline
    with torch.no_grad():
        with torch.autocast(device_type=device.type):
            generated_images = pipeline(
                class_label=class_label,
                encoder_hidden_states=encoder_hidden_states,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images=n_images,
            ).images
    
    # Restore the generator state to the saved initial state
    generator.set_state(initial_state)


    # Generate images using the second pipeline
    with torch.no_grad():
        with torch.autocast(device_type=device.type):
            generated_images_B = second_pipeline(
                class_label=class_label,
                encoder_hidden_states=encoder_hidden_states,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images=n_images,
            ).images

    closest_images = []
    similarities = []

    # Compare each generated image to all images in the dataloader
    for generated_img in generated_images:
        generated_img = pil_to_tensor(generated_img).to(accelerator.device)
        max_similarity = -float("inf")
        closest_img = None

        for batch in dataloader:
            batch_images = batch["pixel_values"].to(accelerator.device)

            for original_img in batch_images:
                # Compute cosine similarity
                similarity = cos_similarity(
                    generated_img.unsqueeze(0), original_img.unsqueeze(0)
                ).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_img = original_img

        closest_images.append(tensor_to_pil(closest_img))
        similarities.append(max_similarity)

    # Sort generated images, closest images and similarities based on similarities
    sorted_indices = np.argsort(similarities)[::-1]
    generated_images = [generated_images[i] for i in sorted_indices]
    generated_images_B = [generated_images_B[i] for i in sorted_indices]
    closest_images = [closest_images[i] for i in sorted_indices]

    return generated_images, generated_images_B, closest_images


def remove_im_mean(data):
    return data - data.mean(dim=(1, 2, 3), keepdims=True)


def im_set_corr(set1, set2, remove_mean=True):
    """
    im_set: tensor of size N,C,H, W
    """

    if len(set1.shape) != 4 or len(set2.shape) != 4:
        raise ValueError("Input shape error")
    if remove_mean:
        set1 = remove_im_mean(set1)
        set2 = remove_im_mean(set2)

    norms1 = set1.norm(dim=(2, 3), keepdim=True).norm(dim=1, keepdim=True)
    norms1[norms1 == 0] = 0.001  # to avoid dividing by 0 for blank images
    norms2 = set2.norm(dim=(2, 3), keepdim=True).norm(dim=1, keepdim=True)
    norms2[norms2 == 0] = 0.001  # to avoid dividing by 0 for blank images

    return torch.matmul(
        ((set1 / norms1).flatten(start_dim=1)), ((set2 / norms2).flatten(start_dim=1).T)
    )


def generate_histogram(generated_images_A, generated_images_B, original_images):

    # Plot similarities between generated images A and B, and A and original images
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
    plt.tight_layout()

    # Compute similarities between the two sets of generated images
    sims = im_set_corr(generated_images_A, generated_images_B).diag()
    bins = np.linspace(0.0, 1.0, 100)

    # Plot the histogram of similarities between generated sets A and B
    ax.hist(
        sims.cpu().flatten(),
        bins=bins,
        label="Samples from two denoisers",
        alpha=1,
        density=True,
    )

    # Compute similarities between generated set A and the original images
    corrs = im_set_corr(generated_images_A, original_images)
    values, indices = corrs.max(dim=1)

    ax.hist(
        values.cpu(),
        bins=bins,
        label="Sample and closest train image",
        alpha=0.7,
        density=True,
    )

    ax.legend(fontsize=15)
    ax.set_xlim(-0.1, 1.1)
    ax.set_title("Histogram", fontsize=20)
    ax.tick_params(bottom=True, left=True, labelleft=False, labelbottom=True)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels((np.linspace(0, 1, 5)), fontsize=15)
    plt.subplots_adjust(top=0.85)

    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    histogram = Image.open(buf)

    return histogram


def save_images(images, output_dir):
    # Create a outputs directory
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(img_path)


def main():
    args = parse_args()

    wandb.init(
        project="generalization_tests",
        name=f"{args.experiment_name}",
        config={
            "method": args.finetunning_method,
            "n_train_samples": args.data_samples,
            "class_label": args.class_label,
        },
    )

    (
        pipeline_inversion,
        second_pipeline_inversion,
        dataloader,
        accelerator,
        encoder_max_position_embeddings,
        encoder_hidden_size,
    ) = load_model(args)

    # Prepare empty text encoder hidden states
    encoder_hidden_states = torch.zeros(
        [
            args.num_images_per_class,
            encoder_max_position_embeddings,
            encoder_hidden_size,
        ],
        dtype=accelerator.unwrap_model(pipeline_inversion.unet).dtype,
    ).to(accelerator.device)

    generated_images, generated_images_B, closest_images = (
        generate_and_find_closest_images_direct(
            pipeline=pipeline_inversion,
            second_pipeline=second_pipeline_inversion,
            accelerator=accelerator,
            dataloader=dataloader,
            n_images=args.num_images_per_class,
            encoder_hidden_states=encoder_hidden_states,
            class_label=args.class_label,
            num_inference_steps=100,
            seed=42,
        )
    )

    histogram = generate_histogram(
        torch.stack([pil_to_tensor(image) for image in generated_images]),
        torch.stack([pil_to_tensor(image) for image in generated_images_B]),
        torch.stack([pil_to_tensor(image) for image in closest_images]),
    )

    combined_images = [
        combine_images(gen_img, closest_img, "Generated", "Real")
        for gen_img, closest_img in zip(
            generated_images[: args.n_close_images_to_upload],
            closest_images[: args.n_close_images_to_upload],
        )
    ]

    wandb.log(
        {
            "top closest images": [
                wandb.Image(
                    image,
                    caption=f"top {i}",
                )
                for i, image in enumerate(combined_images)
            ],
            "histogram": wandb.Image(
                histogram,
            ),
        }
    )

    wandb.finish()

    # save_images(combined_images, args.output_dir[0])
    # save_images(closest_images[: args.n_close_images_to_upload], args.output_dir[1])


if __name__ == "__main__":
    main()
