# CT Image SuperResolution with Deep Learning
### NeuroEngineering Project Workshop 2021/2022 @ Politecnico di Milano
*Active Contributors: **Alberto Rota**, Carlo Sartori, Federico Monterosso, Iva Milojkovic*

*Supervisors: Prof. Pietro Cerveri, PhD; Matteo Rossi, PhD*
***
A Deep Learning model for superresolution of CT Images. This superresolution task consists in cthe combination of denoising and increasing of spatial extent. Both a higher PSNR and SSIM were archieved when compared to standard non-ML superresolution strategies like bilinear, cubic and quintic interpolation: such methods avereaged a PSNR of 35.2dB and SSIM of 0.88 on the available dataset. The developed dense model reached a 43dB PSNR and 0.95 SSIM on the test set (20% of the total dataset size) only

![superresolution](https://github.com/alberto-rota/CT-SuperResolution-with-Deep-Learning/blob/main/predicted_vs_expected_results.png)

**Input:** 128x128x64 low-res CT-scans, with superimposed gaussian noise

**Output:**  256x256x64 high-res CT-scans, denoised

### The Model
The network implements a variation on the standard DenseNet architecture, where a 4x upsampling followed by a convolutional layer with stride 2 is added at the beginning.
The model has ~45K parameters and trains 1 epoch in 102 seconds on an NVIDIA P100 GPU.
An efficient training earlystopped after 55 of the 100 desired epoch, while using a batch size of 4 slices to avoid overloading the RAM. Even tho multiple *ad hoc* loss functions were tested (which took in consideration the PSNR), the best performance was obtained from MSE.

![densenet](https://github.com/alberto-rota/CT-SuperResolution-with-Deep-Learning/blob/main/BIGDenseNet.png)

The model output can be viewed sliced or in 3D from the provided MATLAB script
