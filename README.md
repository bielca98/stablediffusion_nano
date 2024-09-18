# Image-to-Image Translation and Fine-Tuning with Stable Diffusion 

This project corresponds to a six-month internship conducted at the Institut de Biologie de l’École Normale Supérieure as part of the Master 2 program in Mathematics, Vision, and Learning at ENS Paris-Saclay. The internship was completed in the Computational Bioimaging and Bioinformatics lab, led by Dr. Auguste Genovesio, with guidance from Anis Bourou and Thomas Boyer. The primary focus of the work was on the fine-tuning of conditional diffusion models and their applications in image-to-image translation, specifically for biological images. The report of the internship is included in this repository (CASTANO-SEGADE_BIEL_RAPPORT.pdf).

This project builds upon code from the [SVDiff Project](https://github.com/mkshing/svdiff-pytorch) and [Stable Diffusion 2.1](https://github.com/Stability-AI/stable-diffusion). It incorporates our own custom code for fine-tuning pre-trained models on specific datasets using various fine-tuning techniques. 

## Overview 

The primary goal of this project is to fine-tune pre-trained models, such as Stable Diffusion 2.1, to work with specific image datasets. While Stable Diffusion typically generates images conditioned by text prompts, this project adapts the model to generate images conditioned by integer labels (representing different classes). 

We explore several fine-tuning techniques: 

- **SVDiff** 

- **LoRA (Low-Rank Adaptation)** 

- **Attention Fine-Tuning** 

- **Full Model Fine-Tuning** 

### Datasets 
We used this code to train and evaluate models on various datasets, including: 
- **BBBC021** (Broad Bioimage Benchmark Collection) 
- **Golgi** 
- **LARKK2** 

## Fine-Tuning Methods 

Although the project is based on Stable Diffusion, we focus on adapting the model to handle integer-based class labels rather than text conditioning. These methods allow the model to be fine-tuned or translated between different image classes. 
## Bash Scripts 
This repository contains several bash scripts to handle the model fine-tuning and image generation tasks. Each script has specific purposes, which are briefly explained below: 
- **`train.sh`**: For unconditional training, where the model learns without being conditioned on class labels. 
- **`train_two_classes.sh`**: For conditional training on two classes, where the model is trained to distinguish and generate images between two specific classes. 
- **`train_jeanzay_unconditional.sh` and `train_jeanzay.sh`**: These are equivalent to the previous scripts but specifically adapted to run on the Jean Zay supercomputer. 
- **`test_generalization.sh`**: To test the generalization or memorization capabilities of a fine-tuned model. This script evaluates how well a model can generate or classify unseen images. 
- **`img2img.sh`**: Implements the image-to-image (img2img) translation pipeline. This allows you to translate images from one class to another using a pre-trained model, applying the model's learned transformations. 
- **`img_generation.sh`**: For generating and storing images using a fine-tuned model. This script generates new images from the fine-tuned model, saving them in the specified output folders. Each script comes with detailed usage instructions as comments in the header, so you can easily understand the input parameters and expected outputs. 
## License This project is based on code from SVDiff and Stable Diffusion 2.1. Please refer to their respective licenses for more information. 