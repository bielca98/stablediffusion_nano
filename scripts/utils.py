from torch.utils.data import Dataset
from pathlib import Path
import os
import inspect
from torchvision import transforms
from PIL import Image
import torch
import random
from diffusers import __version__
from diffusers import UNet2DConditionModel
from accelerate.logging import get_logger
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from svdiff_pytorch import UNet2DConditionModelForSVDiff
from peft import PeftModel, PeftConfig, get_peft_model
from safetensors.torch import save_file
from safetensors.torch import safe_open
import huggingface_hub

logger = get_logger(__name__)


class DownStreamDataset(Dataset):
    """
    A dataset to prepare the images for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root,
        size=512,
        center_crop=False,
        data_samples=10,
        data_sampling_seed=43,
    ):
        self.size = size
        self.center_crop = center_crop

        # Set the seed for reproducibility
        random.seed(data_sampling_seed)

        self.data_root = [Path(root) for root in data_root]
        self.images_paths = []
        for root in self.data_root:
            if not root.exists():
                raise ValueError("Instance images root doesn't exists.")

            # Get all image paths in the current root
            all_images_paths = list(root.iterdir())

            # Sample a specific number of images
            sampled_images_paths = random.sample(all_images_paths, data_samples)

            # Add the sampled paths to the main list as a new sublist
            self.images_paths.append(sampled_images_paths)

        self.num_images = [len(paths) for paths in self.images_paths]
        self._length = sum(self.num_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.RandomCrop(size)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Determine which directory the index falls into
        i = 0
        while index >= self.num_images[i]:
            index -= self.num_images[i]
            i += 1

        example = {}
        image = Image.open(self.images_paths[i][index])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["class_label"] = i
        example["images"] = self.image_transforms(image)

        return example


def collate_fn(examples):
    pixel_values = [example["images"] for example in examples]
    class_labels = [example["class_label"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    class_labels = torch.tensor(class_labels).long()

    batch = {
        "pixel_values": pixel_values,
        "class_labels": class_labels,
    }
    return batch


def check_substring(n, target_modules):
    for module in target_modules:
        if module in n:
            return True
    return False


def save_weights(
    step, unet, accelerator, output_dir, method, class_conditioning, save_path=None
):
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        if save_path is None:
            save_path = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        state_dict = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).state_dict()

        # If using svdiff, only save delta weights
        if method == "svdiff":
            svdiff_state_dict_keys = [
                k
                for k in accelerator.unwrap_model(unet).state_dict().keys()
                if "delta" in k
            ]
            state_dict_svdiff = {k: state_dict[k] for k in svdiff_state_dict_keys}
            save_file(
                state_dict_svdiff,
                os.path.join(save_path, "spectral_shifts.safetensors"),
            )
        elif method == "svdiff_attention":
            weights_to_save = [
                "to_q.delta",
                "to_k.delta",
                "to_v.delta",
                "to_out.0.delta",
            ]
            svdiff_state_dict_keys = [
                k
                for k in accelerator.unwrap_model(unet).state_dict().keys()
                if check_substring(k, weights_to_save)
            ]
            state_dict_svdiff = {k: state_dict[k] for k in svdiff_state_dict_keys}
            save_file(
                state_dict_svdiff,
                os.path.join(save_path, "spectral_shifts.safetensors"),
            )
        else:
            unet.save_pretrained(save_path)

        # Explicetely save class embedding weights
        if class_conditioning and method in [
            "lora",
            "svdiff",
            "svdiff_attention",
            "lora_attention",
        ]:
            class_embedding_state_dict_keys = [
                k
                for k in accelerator.unwrap_model(unet).state_dict().keys()
                if "class_embedding" in k
            ]
            state_dict = {k: state_dict[k] for k in class_embedding_state_dict_keys}
            save_file(
                state_dict, os.path.join(save_path, "class_embedding.safetensors")
            )

        print(f"[*] Weights saved at {save_path}")


def load_safetensors_file(
    pretrained_model_name_or_path,
    file_name,
    model,
    param_device,
    torch_dtype,
    hf_hub_kwargs=None,
):
    if os.path.isdir(pretrained_model_name_or_path):
        pretrained_model_name_or_path = os.path.join(
            pretrained_model_name_or_path, file_name
        )
    elif not os.path.exists(pretrained_model_name_or_path):
        # download from hub
        hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
        pretrained_model_name_or_path = huggingface_hub.hf_hub_download(
            pretrained_model_name_or_path,
            filename=file_name,
            **hf_hub_kwargs,
        )
    assert os.path.exists(pretrained_model_name_or_path)

    with safe_open(pretrained_model_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            accepts_dtype = "dtype" in set(
                inspect.signature(set_module_tensor_to_device).parameters.keys()
            )
            if accepts_dtype:
                set_module_tensor_to_device(
                    model,
                    key,
                    param_device,
                    value=f.get_tensor(key),
                    dtype=torch_dtype,
                )
            else:
                set_module_tensor_to_device(
                    model, key, param_device, value=f.get_tensor(key)
                )
    print(f"Resumed from {pretrained_model_name_or_path}")
    return model


def load_unet_custom(
    pretrained_model_name_or_path,
    weights_path=None,
    revision=None,
    subfolder=None,
    method=None,
    lora_config=None,
    class_conditioning=False,
    is_local_checkpoint=False,
    **kwargs,
):
    user_agent = {
        "diffusers": __version__,
        "file_type": "model",
        "framework": "pytorch",
    }

    # load config
    config, unused_kwargs, commit_hash = UNet2DConditionModel.load_config(
        pretrained_model_name_or_path,
        return_unused_kwargs=True,
        return_commit_hash=True,
        force_download=False,
        lora_config=None,
        revision=revision,
        subfolder=subfolder,
        user_agent=user_agent,
        local_files_only=is_local_checkpoint,
    )

    # Add class embeddings
    initialize_class_embeddings = False
    if class_conditioning and config["num_class_embeds"] is None:
        config["num_class_embeds"] = 2
        initialize_class_embeddings = True

    # Instantiate model with empty weights (or random weights if from_scratch)
    if "svdiff" in method:
        with init_empty_weights():
            model = UNet2DConditionModelForSVDiff.from_config(config)
    elif method == "lora" or method == "lora_attention":
        with init_empty_weights():
            model = UNet2DConditionModel.from_config(config)

            if lora_config is None:
                lora_config = PeftConfig.from_pretrained(weights_path)

            model = get_peft_model(model, lora_config)
    elif method == "from_scratch":
        model = UNet2DConditionModel.from_config(config)
    else:
        with init_empty_weights():
            model = UNet2DConditionModel.from_config(config)

    # Define device and data type
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None

    # Get accepts_dtype
    accepts_dtype = "dtype" in set(
        inspect.signature(set_module_tensor_to_device).parameters.keys()
    )

    if method != "from_scratch" or weights_path is not None:

        # Load pre-trained weights
        if "svdiff" in method:
            original_model = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision
            )
            state_dict = original_model.state_dict()
            shift_weighgts = {
                n: torch.zeros(p.shape)
                for n, p in model.named_parameters()
                if "delta" in n
            }
            state_dict.update(shift_weighgts)
        elif method == "lora" or method == "lora_attention":
            original_model = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision
            )
            if weights_path is not None:
                original_model = PeftModel.from_pretrained(original_model, weights_path)
            else:
                original_model = get_peft_model(original_model, lora_config)
            state_dict = original_model.state_dict()
        else:
            if weights_path is None:
                original_model = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    revision=revision,
                )
            else:
                original_model = UNet2DConditionModel.from_pretrained(weights_path)

            state_dict = original_model.state_dict()

        # Check if all keys are present
        missing_keys = set(
            model.state_dict().keys()
            - set(["class_embedding.weight", "base_model.model.class_embedding.weight"])
        ) - set(state_dict.keys())

        if len(missing_keys) > 0:
            raise ValueError(
                "Cannot load {UNet2DConditionModel} from "
                f"{pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}."
            )

        # Copy base weights
        for param_name, param in state_dict.items():
            if accepts_dtype:
                set_module_tensor_to_device(
                    model, param_name, param_device, value=param, dtype=torch_dtype
                )
            else:
                set_module_tensor_to_device(
                    model, param_name, param_device, value=param
                )

        # Copy svdiff weights
        if weights_path is not None and "svdiff" in method:
            model = load_safetensors_file(
                weights_path,
                "spectral_shifts.safetensors",
                model,
                param_device,
                torch_dtype,
                **kwargs,
            )

        del original_model

    # Initialize class embeddings weight
    if initialize_class_embeddings and weights_path is None:
        param = torch.randn(model.class_embedding.weight.shape).to(param_device)
        if accepts_dtype:
            set_module_tensor_to_device(
                model,
                "class_embedding.weight",
                param_device,
                value=param,
                dtype=torch_dtype,
            )
        else:
            set_module_tensor_to_device(
                model, "class_embedding.weight", param_device, value=param
            )
    elif class_conditioning and method in [
        "svdiff",
        "svdiff_attention",
        "lora",
        "lora_attention",
    ]:
        model = load_safetensors_file(
            weights_path,
            "class_embedding.safetensors",
            model,
            param_device,
            torch_dtype,
            **kwargs,
        )

    if "torch_dtype" in kwargs:
        model = model.to(kwargs["torch_dtype"])

    model.register_to_config(_name_or_path=pretrained_model_name_or_path)

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    torch.cuda.empty_cache()

    return model
