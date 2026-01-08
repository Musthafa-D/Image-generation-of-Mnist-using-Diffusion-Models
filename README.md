# MNIST Image Generation Using Diffusion Models (PyTorch)

## Overview

This project implements **image generation on the MNIST dataset using diffusion models** in PyTorch.  
Diffusion models are a class of generative models that learn to produce new images by reversing a noise process. They are a modern alternative to GANs and have shown strong results on image synthesis tasks.

This repository contains code to train and evaluate different diffusion models for generating MNIST images and visualize the results and also to interpret the trained models using various interpretability methods such as attributions, metrics, etc.

---

## Dataset

- **MNIST (Modified National Institute of Standards and Technology)**
- 70,000 grayscale images of handwritten digits (0–9)
  - 60,000 training images
  - 10,000 test images
- Each image is 28×28 pixels

Both conditional and unconditional of denoising diffusion models as well as latent diffsuion models are trained and evaluated.

---

## What This Project Includes

- Data loading and preprocessing hooks
- Diffusion model implementation in PyTorch
- Training loop (forward noise → reverse denoising)
- Image sampling and visualization
- Saving generated images during training
- Interpretability of the trained models.

---
