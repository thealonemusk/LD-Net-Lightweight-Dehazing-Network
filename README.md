# LD-Net: Lightweight Dehazing Network Using Convolutional Autoencoder for Image Dehazing

LD-Net is a deep learning-based algorithm designed to enhance visibility in hazy images and videos, improving visual quality and clarity. This project implements a lightweight dehazing network based on a convolutional autoencoder (CAE) architecture.

## Key Features

- Achieved a 40% improvement in structural similarity index (SSIM) compared to baselines
- Reduced model size by 70% while maintaining performance
- Integrated PyTorch model with a React.js frontend via a Flask API
- Enables real-time image processing with under 2-second latency

## Project Structure

The project consists of two main components:

1. Frontend (React.js)
2. Backend (Python Flask API)

## Installation and Setup

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd Backend
pip install -r requirements.txt
python main.py
```

### Virtual Environment Setup

#### For Linux-based systems:

```bash
python -m venv .venv
source .venv/bin/activate
```

#### For Windows-based systems:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Required Python Modules

Install the following modules in your virtual environment:

```bash
pip install torch opencv-python numpy tqdm matplotlib
```

**Note:** Make sure to compile and run the project using the virtual environment kernel.

## Technical Details

LD-Net utilizes the encoder-decoder structure of a Convolutional Autoencoder (CAE) to learn a latent representation that captures the haze-free content of the image. The decoder then reconstructs the dehazed image from this latent representation.

Key advantages of LD-Net include:
- Lightweight architecture suitable for real-time applications
- Computational efficiency
- Improved image quality metrics (PSNR and SSIM) compared to hazy input images

## Evaluation

The network has been evaluated on a benchmark hazy image dataset, demonstrating its effectiveness in improving image quality metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

## Future Work

- Further optimization of the model for even faster processing
- Expansion of the dataset to improve generalization
- Integration with mobile platforms for on-device dehazing

## Contributors

Ashutosh Jha
Ayush Sehgal

