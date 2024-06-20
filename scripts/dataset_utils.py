from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch


class DownStreamDataset(Dataset):
    """
    A dataset to prepare the images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        prompts,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.data_root = [Path(root) for root in data_root]
        for root in self.data_root:
            if not root.exists():
                raise ValueError("Instance images root doesn't exists.")

        self.images_paths = [list(root.iterdir()) for root in self.data_root]
        self.num_images = [len(paths) for paths in self.images_paths]
        self.prompts = prompts
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
        example["images"] = self.image_transforms(image)
        example["prompt_ids"] = self.tokenizer(
            self.prompts[i],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    pixel_values = [example["images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch
