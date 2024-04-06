# minor_project


## installtion for linux based compiler 

python vnev .vnev
source ./venv/bin/activate

## installtion for windows based compiler 

python vnev .vnev
source ./venv/Scripts/activate

## Installation of required modules 
pip install torch opencv-python numpy tqdm matplotlib

compile using .vnev kernel 


Haze is a scattering phenomenon that reduces the visibility of scenes in images. Dehazing techniques aim to recover the original scene from hazy images. This paper proposes LD-Net, a lightweight dehazing network based on a convolutional autoencoder (CAE) architecture. LD-Net utilizes the encoder-decoder structure of a CAE to learn a latent representation that captures the haze-free content of the image. The decoder then reconstructs the dehazed image from the latent representation. The proposed network is lightweight and computationally efficient, making it suitable for real-time applications. We evaluate LD-Net on a benchmark hazy image dataset and demonstrate its effectiveness in improving image quality metrics like PSNR and SSIM compared to the hazy images.

