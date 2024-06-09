# Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models

This readme file is an outcome of the [CENG502 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2024) Project List](https://github.com/CENG502-Projects/CENG502-Spring2024) for a complete list of all paper reproduction projects. The public repository and commit history are available [here](https://github.com/AbtinMogharabin/MakeMe-BNN).


# 1. Introduction

The challenge of effectively quantifying uncertainty in predictions made by deep learning models, particularly Deep Neural Networks (DNNs), is crucial for their safe deployment in risk-sensitive environments. DNNs are typically deterministic, providing point estimates without any measure of confidence. This can lead to overconfident decisions in many real-world applications, particularly in safety-critical domains such as autonomous driving, medical diagnoses, industrial visual inspection, etc. 

Traditional Bayesian Neural Networks (BNNs) represent a probabilistic approach to neural networks where the weights are treated as random variables with specified prior distributions rather than fixed values. This method allows BNNs to not only make predictions but also to estimate the uncertainty of these predictions by integrating over the possible configurations of weights. To compute these integrations, which are generally intractable due to the high dimensionality of the parameter space, BNNs typically employ approximation techniques like Variational Inference (VI) or Markov Chain Monte Carlo (MCMC). These methods, however, introduce significant computational overhead and complexity. VI, for instance, requires the selection of a tractable family of distributions that can limit the expressiveness of the model. At the same time, MCMC is computationally expensive and slow to converge, particularly for large datasets or complex network architectures.

Given these challenges, the deployment of traditional BNNs in real-world applications, especially those requiring real-time operations or running on limited hardware, needs to be more practical. The paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre "trained Models" introduces an innovative approach, termed Adaptable Bayesian Neural Network (ABNN), which allows the conversion of existing pre-trained DNNs into BNNs. This conversion is achieved in a post-hoc manner—after the DNN has been trained—requiring minimal computational resources and avoiding extensive retraining. This paper was published at CVPR (Conference on Computer Vision and Pattern Recognition) in 2024.

ABNN preserves the main predictive properties of DNNs while enhancing their uncertainty quantification abilities. The paper conducts extensive experiments across multiple datasets for image classification and semantic segmentation tasks. It demonstrates that ABNN achieves state-of-the-art performance without the computational budget typically associated with ensemble methods. The following figure shows a brief comparison of ABNN and a number of other uncertainty-based deep learning approaches in literature:

<div align="center">
    <img src="Images/Brief-Evaluation.png" alt="Brief" width="800" height="320">
    <p id="Brief"></p>  
</div>

In this repository, we make an effort to reproduce the methods and results of the paper based on the descriptions provided.

## 1.1. Paper summary

The ABNN approach starts with a pre-trained DNN and transforms it into a Bayesian Neural Network (BNN) by introducing Bayesian Normalization Layers (BNLs) into existing normalization layers such as batch or layer normalization. This transformation involves adding Gaussian noise to the normalization process, thereby incorporating uncertainty in the model's predictions without extensive retraining. The process is designed to be computationally efficient, requiring only minimal additional training (fine-tuning) to adjust the new layers, making it feasible to implement on existing models without significant computational overhead.

<div align="center">
    <img src="Images/Approach-Illustration.png" alt="Approach" width="800" height="300">
    <p id="Approach"></p>  
</div>

The key contributions of this paper are as follows:

- ABNN provides a scalable way to estimate uncertainty by leveraging pre-trained models and transforming them with minimal computational cost. This approach circumvents the traditionally high resource demands of training BNNs from scratch or employing ensemble methods.
- The method is compatible with multiple neural network architectures such as ResNet-50, WideResnet28-10, and ViTs. The only requirement for ABNN to be compatible with a DNN architecture is that the DNN should include normalization layers (such as batch, layer, or instance normalization). This is not a limiting factor as most modern architectures include one type of these layers
- ABNN can estimate the posterior distribution around the local minimum of the pre-trained model in a resource-efficient manner while still achieving competitive uncertainty estimates with diversity. The results indicate that ABNN achieves comparable or superior performance in uncertainty estimation and predictive accuracy compared to existing state-of-the-art methods like Deep Ensembles and other Bayesian methods in both in- and out-of-distribution settings
- Stability and Performance: It is noted that ABNN offers more stable training dynamics compared to traditional BNNs, which are often plagued by training instabilities. The use of Bayesian Normalization Layers helps mitigate these issues, providing a smoother training process and robustness in performance.
- ABNN allows for sequential training of multiple BNNs starting from the same checkpoint, thus modeling various modes within the actual posterior distribution.
- It is also observed that the variance of other transient ABNN parameters is lower compared to that of a classic BNN, resulting in a more stable backpropagation.
- Based on my review, this paper demonstrates one of the very few efforts to translate a deterministic model into a Bayesian version after the training of the deterministic model is finished. To name the two most relevant approaches:
  1. **Deterministic Variational Inference Approach:**
     - One paper employs deterministic variational inference techniques to integrate Bayesian methods into trained deterministic neural networks. It introduces closed-form variance priors for the network weights, allowing the deterministic model to handle uncertainty estimations through a robust Bayesian framework after its initial training [2].
     - Compared to this approach, which requires extensive modifications to the network's inference process to accommodate the new Bayesian priors, the "Make Me a BNN" paper introduces a method that is notably simpler and potentially faster, as it leverages existing normalization layers within pre-trained DNNs to implement Bayesian functionality.

  2. **Decoupled Bayesian Stage Approach:**
     - Another study involves a decoupled Bayesian stage applied to a pre-trained deterministic neural network. This method uses a Bayesian Neural Network to recalibrate the outputs of the deterministic model, thereby improving its predictive uncertainty without retraining the entire network from scratch [3].
     - "Make Me a BNN" paper introduces a quick deployment and straightforward integration into existing models by attaching simple adaptable Bayesian modules directly to the normalization layers. But effective calibration improvement involves adding an entirely new Bayesian processing layer, might not be the most efficient way to introduce Bayesian uncertainty into existing models.


# 2. The method and my interpretation

## 2.1. The original method

### 2.1.1  Model General Overview

The paper conducts a multi-step theoretical analysis of the model's key elements. They explained their reasoning behind each of the chosen methods.

1. In the supplementary material, they show that ABNN exhibits more excellent stability than classical BNNs. This is because, in variational inference BNNs, the gradients crucial for obtaining the Bayesian interpretation vary greatly. This often introduces instability, perturbating the training. ABNN reduces this burden by applying this term to the latent space rather than the weights.

2. In the literature, because of the non-convex nature of the DNN loss, there might exist a need to modify the loss. By adding a new $\varepsilon$ term, they show  empirical benefits for performance and uncertainty quantification

3. Although using BNNs theoretically provides valuable information, they remain unused in practice because of challenges in computing full posteriors. For this reason, ABNN solely samples the sampling noise terms (ϵ) and averages over multiple training terms to generate robust predictions during inference.

After these modifications, the general model training procedure is as follows:

<div align="center">
    <img src="Images/Training-Procedure.png" alt="Procedure" width="430" height="580">
    <p id= "Procedure">Figure 3: ABNN's general training procedure</p>  
</div>

### 2.1.2  Bayesian Normalization Layers (BNLs)

The BNL is the core of the ABNN approach, which adapts conventional normalization layers by incorporating Gaussian noise to model uncertainty. Here's the detailed equation for the BNL:

$$
u_j = \text{BNL}(W^{(j)} h_{j-1})
$$

$$
a_j = a(u_j)
$$

$$
\text{BNL}(h_j) = \frac{(h_j - \hat{\mu}_j)}{\hat{\sigma}_j} \cdot \gamma_j (1 + \epsilon_j) + \beta_j
$$

Where:
- $u_j$: Represents the pre-activation mapping at layer $j$.
- $h_j$: Represents the input to the normalization layer.
- $W^{(j)}$: Weights of layer $j$.
- $\hat{\mu}_j$ and $\hat{\sigma}_j$: Empirical mean and standard deviation computed from the input $h_j$.
- $\gamma_j$ and $\beta_j$: Learnable parameters that scale and shift the normalized input.
- $\epsilon_j \sim \mathcal{N}(0,1)$: Gaussian noise added to introduce randomness and model uncertainty.
- $a(\cdot)$: Activation function applied to $u_j$ to get the activation output $a_j$.

### 2.1.3  Fine-tuning the ABNN

During the fine-tuning phase, ABNN optimizes the network's parameters (more focusing on the parameters introduced in the BNL). The loss function is a combination of the standard training loss (The Maximum A Posteriori (MAP) Loss) and additional $\varepsilon$ term to manage the Bayesian aspects:

The MAP loss, $L_{MAP}(\omega)$, is given by the formula:

$$
L_{MAP}(\omega) = -\sum_{(x_i,y_i) \in D} \log P(y_i | x_i,\omega) - \log P(\omega)
$$

- The first term is the log-likelihood of the data given the parameters, which is typical for maximum likelihood estimation.
- The second term, $-\log P(\omega)$, is the logarithm of the prior probability of the parameters, incorporating prior beliefs about the parameter values into the training process.

The calculation of the extra $\varepsilon$ term is done as below:

$$
ε(\omega) = -\sum_{(x_i,y_i) \in D} \eta_i \log P(y_i | x_i,\omega)
$$

Where:
- $D$: Training dataset consisting of input-output pairs $(x_i, y_i)$.
- $\eta_i$: Class-dependent random weight initialized at the beginning of training.
- $P(y_i | x_i, \omega)$: The probability of target $y_i$ given the input $x_i$ and model parameters $\omega$.

Then, the loss will be calculated:

$$
L(\omega) = L_{MAP}(\omega) + \varepsilon(\omega)
$$

### 2.1.4  Inference with ABNN

During inference, ABNN uses the stochastic nature of BNLs to generate a predictive distribution over outputs for given inputs. They achieved this by sampling from the Gaussian noise components $\epsilon_j$ during each forward pass, thus generating different outputs for the same input. In the end, ABNN averages the results of multiple such stochastic passes and obtains a single prediction.

$$
P(y | x, D) \approx \frac{1}{ML} \sum_{l=1}^L \sum_{m=1}^M P(y | x, \omega_m, \epsilon_l)
$$

Where:
- $P(y | x, D)$: The probability of the output (y) depending on the input (x) and the whole training dataset (D).
- $M$: Number of models (ensemble members).
- $L$: Number of noise samples (stochastic forward passes).
- $\omega_m$: Parameters of the $m$-th model configuration.
- $\epsilon_l$: Noise vector sampled for the $l$-th stochastic forward pass.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The paper demonstrates the efficiency of the ABNN approach to a number of different datasets and backbones in *image classification* and *semantic segmentation* tasks. A group of both in- and out-distribution datasets were used.

### 3.1.1  Datasets and General Details

### 3.1.1.1  Image Classification

1. **[CIFAR-10 and CIFAR-100 ](https://www.cs.toronto.edu/~kriz/cifar.html)** [5]:
   - **CIFAR-10**: CIFAR-10 contains 60,000 images divided into 10 classes. The images are colored, with a resolution of 32x32 pixels. The 10 classes in CIFAR-10 are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It contains 50,000 training images and 10,000 test images
   - **CIFAR-100**: This dataset contains 60,000 images across 100 classes, with 600 images per class. Like CIFAR-10, these are 32x32 pixel color images. The 100 classes are grouped into 20 superclasses, with each superclass containing five related classes. There are 50,000 training images and 10,000 test images, distributed equally across the 100 classes.
   - **Usage**: In the paper, this dataset was used to evaluate image classification tasks with ResNet-50 and WideResNet-28x10 backbones. They did the training from scratch but pre-trained models would have similar results.
   - **Results**: The ABNN model achieved competitive performance on CIFAR-10 and CIFAR-100 when using ResNet-50 and WideResNet28-10 architectures. Specifically, for CIFAR-10 with WideResNet-28×10, they achieved the best AUPR and AUC (while having the smaller number of parameters), and for CIFAR-100 with ResNet-50, they had the best AUPR, AUC, and FPR95. The paper also provides details on the fast computation of the models compared to other uncertainty-based models. The training time for CIFAR-10 and CIFAR-100 datasets on ResNet-50 and WideResNet-28x10 backbones was 12 hours in total on a single RTX 3090.

<div align="center">
    <img src="Images/CIFAR-10.png" alt="CIFAR-10" width="350" height="300">
    <p id="CIFAR-10">Figure 4: Sample images from the CIFAR-10 dataset </p>  
</div>
  
2. **[Street View House Numbers (SVHN) ](http://ufldl.stanford.edu/housenumbers/)** [7]:
   - **Description**:  SVHN is a real-world image dataset obtained from house numbers in Google Street View images. Images in SVHN, like CIFAR, are small, often 32x32 pixels. This dataset contains over 600,000 images.
   - **Usage**: The paper used this dataset as the out-of-distribution dataset for models trained on CIFAR-10/100 to test their generalization and uncertainty estimation. 
   - **Results**: The paper didn't share the specific performance metrics on this dataset.

<div align="center">
    <img src="Images/SVHN.png" alt="SVHN" width="450" height="400">
    <p id="SVHN">Figure 5: Sample images from the SVHN dataset in full numbers format </p>     
</div>

3. **[ImageNet ](https://www.image-net.org/download.php)** [6]:
   - **Description**: It has 1,000 classes, each with varying numbers of images, but generally several hundred to a few thousand images per class. The classes represent a broad range of objects on everyday items and the training set contains approximately 1.2 million images, while the test set has about 50,000 images. The dimensions are typically resized to 224x224 pixels for model training.
   - **Usage**: In the paper, ResNet-50 and Vision Transformer (ViT) were used for experiments on ImageNet for image classification tasks. For these backbones, they used torchvision pre-trained models.
   - **Results**: ABNN demonstrated an accuracy of 79.5% with ResNet-50 and 80.6% with ViT. For ViT, their approach achieved better FPR95 and ECE compared to other models.

<div align="center">
    <img src="Images/ImageNet.png" alt="ImageNet" width="600" height="200">
    <p id="ImageNet">Figure 6: Sample images from the ImageNet dataset </p>     
</div>

4. **[Describable Textures Dataset (DTD) ](https://www.robots.ox.ac.uk/~vgg/data/dtd/)** [8]:
   - **Description**:  DTD is a dataset of textural images organized according to a list of 47 terms (categories) inspired by human perception. There are a total of 5,640 images. The data is typically processed to fit the input size (224x224 pixels) requirements of the ImageNet model. This dataset contains over 600,000 images.
   - **Usage**: The paper used this dataset as the out-of-distribution dataset for models trained on ImageNet-trained models. 
   - **Results**: Specific performance metrics were not shared in the paper.

<div align="center">
    <img src="Images/DTD.png" alt="DTD" width="800" height="200">
    <p id="DTD">Figure 7: **Sample images from the DTD dataset</p>     
</div>


### 3.1.1.2  Semantic Segmentation

1. **[StreetHazards ](https://github.com/hendrycks/anomaly-seg)** [9]:
   - **Description**: This dataset is a part of the larger BDD100K dataset, explicitly designed for benchmarking anomaly detection in the context of street scenes for 13 classes.  The classes represent various street elements. In total, there are 5,125 training images and 1,500 test images of around 512x256 pixels. The test set also contains an additional 250 out-of-distribution classes. 
   - **Usage**: This dataset is designed for semantic segmentation tasks. The paper employed DeepLabv3+ with a ResNet-50 encoder as a backbone, as introduced by Chen et al. [4].
   - **Results**: In semantic segmentation tasks, ABNN achieved a small improvement in AUC compared to other models.

<div align="center">
    <img src="Images/StreetHazards.png" alt="StreetHazards" width="500" height="390">
    <p id="StreetHazards">Figure 8: Sample images from the StreetHazards dataset</p>     
</div>

2. **[BDD-Anomaly ](https://github.com/daniel-bogdoll/anomaly_datasets/blob/main/datasets/bdd-anomaly.py)** [10]:
   - **Description**: A challenging real-world dataset for street scene segmentation that includes diverse conditions such as weather and nighttime scenes. BDD-Anomaly is a subset of the BDD100K dataset, focusing on street scenes with 17 distinct classes in the training set. The test set also introduces two additional out-of-distribution (OOD) classes, namely motorcycle and train.
   - **Usage**: The paper employed the ResNet-50 encoder as a backbone and evaluated the results for semantic segmentation.
   - **Results**: For semantic segmentation, ABNN successfully increased the AUPR and AUC compared to the past state-of-the-art. ECE was also decreased successfully.

3. **[MUAD](https://muad-dataset.github.io/)** [11]:
   - **Description**: MUAD is a synthetic dataset for autonomous driving with multiple uncertainty types and tasks. It contains 10413 in total: 3420 images in the train set, 492 in the validation set and 6501 in the test set. There are a total of 21 classes: 19 classes taken from the CityScapes dataset [12] and two OOD classes representing object anomalies and animals. All these sets cover both day and night conditions.
   - **Usage**: The paper employed this dataset for semantic segmentation tasks focusing on both normal and out-of-distribution scenarios. In the study, a DeepLabV3+ with a ResNet50 encoder was used for the backbone.
   - **Results**: ABNN showed impressive results, achieving a significant increase in mean IoU and AUC, while significantly lowering FPR95 and ECE.

<div align="center">
    <img src="Images/MUAD.png" alt="MUAD" width="1200" height="185">
    <p id="MUAD">Figure 9: Sample images from the MUAD dataset</p>     
</div>


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

| Task               | Dataset (also used for backbone training)  | Method   | Acc ↑    | NLL ↓    | ECE ↑     | AUPR ↑    | AUC ↑     | FPR95 ↓   | ΔParam ↓   | Time (h) ↓ | mIoU ↑    |
|--------------------|--------------------------------------------|----------|----------|----------|-----------|-----------|-----------|-----------|------------|------------|-----------|
| Image Classification | CIFAR-10 (ResNet-50)                     | ABNN     | **95.4** | 0.215    | **0.845** | **97.0**  | **94.7**  | **15.1**  | 0.16       | **2.0**    | -         |
| Image Classification | WideResNet-28x10 (CIFAR-10)              | ABNN     | 93.7     | 0.198    | 1.8       | **98.5**  | **96.9**  | **12.6**  | **0.05**   | **5.0**    | -         |
| Image Classification | CIFAR-100 (ResNet-50)                    | ABNN     | 78.2     | 0.889    | **5.5**   | **89.4**  | **81.1**  | **50.1**  | **0.16**   | **2.0**    | -         |
| Image Classification | CIFAR-10 (WideResNet-28x10)              | ABNN     | 80.4     | 1.08     | **5.5**   | **85.0**  | **75.0**  | **57.7**  | **0.05**   | **5.0**    | -         |
| Image Classification | ImageNet (ResNet-50) including OOD       | ABNN     | **79.5** | -        | **9.65**  | 17.8      | **82.0**  | **65.2**  | -          | -          | -         |
| Image Classification | ImageNet (ViT) including OOD             | ABNN     | 80.6     | -        | **4.32**  | **21.7**  | **85.4**  | **55.1**  | -          | -          | -         |
| Image Segmentation  | StreetHazards including OOD               | ABNN     | -        | -        | 6.09      | 7.85      | **88.39** | 32.02     | -          | -          | 53.82     |
| Image Segmentation  | BDD-Anomaly including OOD                 | ABNN     | -        | -        | **14.03** | **5.98**  | **85.74** | 29.01     | -          | -          | 48.76     |
| Image Segmentation  | MVAD including OOD                        | ABNN     | -        | -        | **5.58**  | 24.37     | **91.55** | **21.68** | -          | -          | **61.96** |

In this table, bolded values shows a result above or equivalent to the past state-of-the-art. For image classification tasks, ABNN was compared with BatchEnsemble, MIMO (ρ = 1), LPBNN, Deep Ensembles, and Laplace models. For image segmentation tasks, ABNN results were compared with TRADI, Deep Ensembles, MIMO, BatchEnsemble, and LP-BNN.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

1. Franchi, G., Laurent, O., Leguéry, M., Bursuc, A., Pilzer, A., & Yao, A. Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models. In CVPR, 2024

2. Wu, A., Nowozin, S., Meeds, E., Turner, R.E., Hernández-Lobato, J.M. & Gaunt, A.L. Deterministic variational inference for robust Bayesian neural networks. In ICLR, 2018

3. Maronas, J., Paredes, R., & Ramos, D. Calibration of deep probabilistic models with decoupled Bayesian neural networks. Neurocomputing, 407, 194-205, 2020

4. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018

5. Krizhevsky, Alex. Learning multiple layers of features from tiny images. Technical report, MIT, 2009

6. Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." In CVPR, 2009

7. Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning." In NeurIPSW, 2011

8. Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. Describing textures in the wild. In Proceedings of the CVPR, 2014

9. Hendrycks, Dan, et al. "A benchmark for anomaly segmentation." arXiv preprint arXiv:1911.11132, 2019.

10. Yu, Fisher, et al. "Bdd100k: A diverse driving dataset for heterogeneous multitask learning." In CVPR, 2020.

11. Franchi, Gianni, et al. "Muad: Multiple uncertainties for autonomous driving, a benchmark for multiple uncertainty types and tasks." In BMVC, 2022.

12. Cordts, Marius, et al. "The cityscapes dataset for semantic urban scene understanding." In CVPR, 2016.

# 6. Contact

**Name: Abtin Mogharabin**

[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:atbinmogharabin@gmail.com)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abtinmogharabin/)
[![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AbtinMogharabin)

**Name: Abduallah Damash**

[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:eng.abduallah1@gmail.com)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/engabduallah/)
[![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/engabduallah)
[![gitlab](https://img.shields.io/badge/GitLab-330F63?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/engabduallah)
