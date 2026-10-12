
# GANoViMaly: A Video Anomaly Detection Framework using Vision Mamba based Generative Adversarial Network

**Authors:** Samprit Bose and Maheshkumar H. Kolekar


## 1. Abstract

Video anomaly detection emphasizes on recognizing unusual activities in video sequences that differ from normal behavioral patterns. In this work, we present a prediction-based framework called GANoViMaly, which combines Generative Adversarial Networks (GANs) with the State Space Model-based Vision Mamba (Vim) architecture for anomaly detection in videos. The framework first applies a spatiotemporal attention mechanism to the input video clips in order to generate 3D tokens that capture both spatial and temporal information. These tokens are then processed by a Vim-based generator that predicts future frames using information from previous frames. A 1x1 patchGAN-based discriminator improves discrimination capabilities. The proposed method is assessed on several benchmark datasets, including UCSD Ped1, UCSD Ped2, CUHK Avenue, and ShanghaiTech. Experimental results show that the framework performs competitively with existing state-of-the-art approaches in both qualitative and quantitative evaluations. In addition to improved detection performance, the model also maintains efficient runtime behavior, achieving an average inference time of 0.029 seconds during the testing phase.


## 2. Methodology

### Overall Architecture

GANoViMaly integrates spatiotemporal attention, Vision Mamba-based future-frame prediction, and a 1×1 PatchGAN discriminator to identify anomalous activities in video sequences.

![GANoViMaly Architecture](Figures/block%20diag.jpg)

*Figure 1. Overall architecture of the proposed GANoViMaly framework.*

## 3. Main Components and Loss Function

- **Spatiotemporal Attention Module:** Processes input video snippets and generates 3D tokens that capture spatial and temporal features.
- **Vision Mamba-based Generator:** Processes the tokens using bidirectional state-space sequence modeling to predict the subsequent video frame.
- **1×1 PatchGAN Discriminator:** Distinguishes actual future frames from predicted frames using pixel-level discrimination.
- **Future-Frame Prediction:** Learns normal spatiotemporal patterns by predicting future frames from previously observed frames.
- **Anomaly Evaluation:** Uses prediction quality and normalized PSNR-based regularity scores to identify deviations from normal behavior.


### Loss Function

The generator is optimized using a combination of mean squared error (MSE), adversarial loss, and L1 loss:

$$
L_{\mathrm{gen}} = \alpha L_{\mathrm{MSE}} + \beta L_{\mathrm{adv}} + \gamma L_{\mathrm{L1}}
$$

where:

- $L_{\mathrm{MSE}}$ measures the squared difference between predicted and ground-truth frames.
- $L_{\mathrm{adv}}$ encourages the generator to produce realistic future frames.
- $L_{\mathrm{L1}}$ penalizes pixel-wise absolute differences.

The discriminator is trained using a least-squares adversarial objective to distinguish real future frames from generated predictions.


## 4. Results

GANoViMaly is evaluated on four benchmark datasets: UCSD Ped1, UCSD Ped2, CUHK Avenue, and ShanghaiTech.

The framework predicts future frames and identifies anomalies through deviations between predicted and actual frames. Prediction errors can be visualized as anomaly heatmaps, while regularity score curves illustrate changes in normality across video frames.

![GANoViMaly Results](Figures/Results%20Graphs.jpg)

*Figure 2. Regularity score curves on testing videos of (a) UCSD Ped1, (b) UCSD Ped2, (c) CUHK Avenue and (d) Shanghai Tech datasets*


## 6. Citation


```

The DOI and final publication details can be added once they are available.
