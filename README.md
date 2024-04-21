# Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models

This readme file is an outcome of the [CENG502 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2024) Project List](https://github.com/CENG502-Projects/CENG502-Spring2024) for a complete list of all paper reproduction projects. The public repository and commit history is available in the account here: https://github.com/AbtinMogharabin/MakeMe-BNN.

# 1. Introduction

The challenge of effectively quantifying uncertainty in predictions made by deep learning models, particularly Deep Neural Networks (DNNs), is crucial for their safe deployment in risk-sensitive environments. DNNs are typically deterministic, providing point estimates without any measure of confidence, which can lead to overconfident decisions in many real-world applications, particularly in safety-critical domains such as autonomous driving, medical diagnoses, industrial visual inspection, etc. 

Traditional Bayesian Neural Networks (BNNs) represent a probabilistic approach to neural networks where the weights are treated as random variables with specified prior distributions, rather than fixed values. This method allows BNNs to not only make predictions but also to estimate the uncertainty of these predictions by integrating over the possible configurations of weights. To compute these integrations, which are generally intractable due to the high dimensionality of the parameter space, BNNs typically employ approximation techniques like Variational Inference (VI) or Markov Chain Monte Carlo (MCMC). These methods, however, introduce significant computational overhead and complexity. VI, for instance, requires the selection of a tractable family of distributions that can limit the expressiveness of the model, while MCMC is computationally expensive and slow to converge, particularly for large datasets or complex network architectures.

Given these challenges, the deployment of traditional BNNs in real-world applications, especially those requiring real-time operations or running on limited hardware, becomes impractical. The paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models" introduces an innovative approach, termed Adaptable Bayesian Neural Network (ABNN), which allows the conversion of existing pre-trained DNNs into BNNs. This conversion is achieved in a post-hoc manner—after the DNN has been trained—requiring minimal computational resources and avoiding extensive retraining. This paper was published at CVPR (Conference on Computer Vision and Pattern Recognition) in 2024.

ABNN preserves the main predictive properties of DNNs while enhancing their uncertainty quantification abilities. The paper conducts extensive experiments across multiple datasets for image classification and semantic segmentation tasks, and demonstrates that ABNN achieves state-of-the-art performance without the computational budget typically associated with ensemble methods. The following figure shows a brief comparison of ABNN and a number of other uncertainty-based deep learning approaches in literature:

<div align="center">
    <img src="Images/Brief-Evaluation.png" alt="Image" width="800" height="320">
</div>

In this repository, we make an effort to reproduce the methods and results of the paper based on the descriptions provided.

## 1.1. Paper summary

The ABNN approach starts with a pre-trained DNN and transforms it into a Bayesian Neural Network (BNN) by introducing Bayesian Normalization Layers (BNLs) to the existing normalization layers (like batch or layer normalization). This transformation involves adding Gaussian noise to the normalization process, thereby incorporating uncertainty into the model's predictions without extensive retraining. The process is designed to be computationally efficient, requiring only minimal additional training (fine-tuning) to adjust the new layers, making it feasible to implement on existing models without significant computational overhead.

<div align="center">
    <img src="Images/Approach-Illustration.png" alt="Image" width="800" height="300">
</div>

The key contributions of this paper are as follows:

- ABNN provides a scalable way to estimate uncertainty by leveraging pre-trained models and transforming them with minimal computational cost. This approach circumvents the traditionally high resource demands of training BNNs from scratch or employing ensemble methods.
- The method is compatible with multiple neural network architectures such ResNet-50, WideResnet28-10; and ViTs. The only requirement for ABNN to have compatiblity with a DNN architectures is that the DNN should include normalization layers (such as batch, layer, or instance normalization). This is not a limiting factor as most modern architectures include one type of these layers
- ABNN can estimate the posterior distribution around the local minimum of the pre-trained model in a resource efficient manner while still achieving competitive uncertainty estimates with diversity. The results indicate that ABNN achieves comparable or superior performance in uncertainty estimation and predictive accuracy compared to existing state-of-the-art methods like Deep Ensembles and other Bayesian methods in both in- and out-of-distribution settings
- Stability and Performance: It is noted that ABNN offers more stable training dynamics compared to traditional BNNs, which are often plagued by training instabilities. The use of Bayesian Normalization Layers helps mitigate these issues, providing a smoother training process and robustness in performance.
- ABNN allows for sequential training of multiple BNNs starting from the same checkpoint, thus modeling various modes within the true posterior distribution.
- It is also observed that the variance of the gradient for ABNN’s parameters is lower compared to that of a classic BNN, resulting in a more stable backpropagation.
- Based on my review, this paper demonstrates one of the very few efforts on translating a deterministic model into a bayesian version after the training of the deterministic model is finished. To name 2 most relevant approaches:
  1. **Deterministic Variational Inference Approach:**
     - One paper employs deterministic variational inference techniques to integrate Bayesian methods into trained deterministic neural networks. It introduces closed-form variance priors for the network weights, allowing the deterministic model to handle uncertainty estimations through a robust Bayesian framework after its initial training [2].
     - Compared to this approach that requires extensive modifications to the network’s inference process to accommodate the new Bayesian priors, the "Make Me a BNN" paper introduces a method that is notably simpler and potentially faster, as it leverages existing normalization layers within pre-trained DNNs to implement Bayesian functionality.

  2. **Decoupled Bayesian Stage Approach:**
     - Another study involves a decoupled Bayesian stage applied to a pre-trained deterministic neural network. This method uses a Bayesian Neural Network to recalibrate the outputs of the deterministic model, thereby improving its predictive uncertainty without retraining the entire network from scratch [3].
     - Unlike the "Make Me a BNN" paper's straightforward and simple approach, this method, while effective for improving calibration, involves adding an entirely new Bayesian processing layer, which might not be as efficient or straightforward in terms of retrofitting existing models with Bayesian capabilities.


# 2. The method and my interpretation

## 2.1. The original method

### 2.1.1  Model General Overview

The paper conducts a multi-step theoreical analysis on the model key elements. Explaining their reasoninng behind each of the chosen methods.

1. In the supplementary material they show that ABNN exhibits greater stability than classical BNNs. This is because in variational inference BNNs the gradients, crucial for obtaining the Bayesian interpretation, vary greatly. This often introduces instability, perturbating the training. ABNN reduces this burden by applying this term on the latent space rather than the weights.

2. In the literature, because of the non-convex nature of the DNN loss, there might exist a need to modify the loss. By adding a new $\varepsilon$ term, they show  show empirical benefits for performance and uncertainty quantification

3. Although using BNNs theoretically provide valuable information, they remain unused in practice because of challenges in computing full posteriors. For this reason, ABNN solely samples the sampling noise terms (ϵ) and average over multiple training terms to generate robust predictions during inference.

After these modifications, the general model training procedure is as follows:

<div align="center">
    <img src="Images/Training-Procedure.png" alt="Image" width="400" height="580">
</div>

### 2.1.2  Bayesian Normalization Layers (BNLs)

The BNL is the core of the ABNN approach, which adapts conventional normalization layers by incorporating Gaussian noise to model uncertainty. Here’s the detailed equation for the BNL:

$$
u_j = \text{BNL}(W_j h_{j-1})
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
- $W_j$: Weights of layer $j$.
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

- The first term is the log likelihood of the data given the parameters, which is typical for maximum likelihood estimation.
- The second term, $-\log P(\omega)$, is the logarithm of the prior probability of the parameters, incorporating prior beliefs about the parameter values into the training process.

The calclation of the extra $\varepsilon$ term is done as below:

$$
ε(\omega) = -\sum_{(x_i,y_i) \in D} \eta_i \log P(y_i | x_i,\omega)
$$

Where:
- $D$: Training dataset consisting of input-output pairs $(x_i, y_i)$.
- $\eta_i$: Class-dependent random weight initialized at the beginning of training.
- $P(y_i | x_i, \omega)$: The probability of target $y_i$ given the input $x_i$ and model parameters $\omega$.

And then, the loss will be calculated:

$$
L(\omega) = L_{MAP}(\omega) + \varepsilon(\omega)
$$

### 2.1.4  Inference with ABNN

During inference, ABNN uses the stochastic nature of BNLs to generate a predictive distribution over outputs for given inputs. They achieved this by sampling from the Gaussian noise components $\epsilon_j$ during each forward pass, thus generating different outputs for the same input. In the end, ABNN averages the results of multiple such stochastic passes and obtains a single prediction.

$$
P(y | x, D) \approx \frac{1}{ML} \sum_{l=1}^L \sum_{m=1}^M P(y | x, \omega_m, \epsilon_l)
$$

Where:
- $P(y | x, D)$: The probbility of the output (y) depending on the input (x) and the whole training dataset (D).
- $M$: Number of models (ensemble members).
- $L$: Number of noise samples (stochastic forward passes).
- $\omega_m$: Parameters of the $m$-th model configuration.
- $\epsilon_l$: Noise vector sampled for the $l$-th stochastic forward pass.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.
### Datasets

The authors used the following datasets for their experiments on uncertainty quantification using Bayesian Neural Networks:

1. **CIFAR-10 and CIFAR-100**:
   - **Description**: CIFAR-10 and CIFAR-100 are popular image classification datasets containing 60,000 images divided into 10 and 100 classes respectively.
   - **Usage**: They are utilized for image classification tasks with models such as ResNet-50 and WideResNet28-10.
   - **Dimension**: Each image is 32x32 pixels.
   - **Data Split**: Training and test splits are standard; CIFAR-10 and CIFAR-100 typically consist of 50,000 training images and 10,000 testing images.   
   - **Results**: The ABNN model achieved competitive performance on CIFAR-10 and CIFAR-100 when using ResNet-50 and WideResNet28-10 architectures. Specifically, for CIFAR-10 with ResNet-50, the accuracy was 95.4%, and for CIFAR-100, the accuracy was 78.9%.
   - **Reference**: [Krizhevsky, Alex. Learning multiple layers of features from tiny images. Technical report, MIT, 2009](https://www.cs.toronto.edu/~kriz/cifar.html) 

2. **ImageNet**:
   - **Description**: A large dataset designed for use in visual object recognition software research, more than 14 million images have been hand-annotated to indicate what objects are pictured and in at least one million of the images, bounding boxes are also provided.
   - **Usage**: Utilized for image classification tasks, specifically tested with ResNet-50 and Vision Transformers (ViT).
   - **Dimension**: There are various dimensions in ImageNet, but typically resized to 224x224 pixels for model training.
   - **Data Split**: It is significantly larger, with over 1.2 million training images and 50,000 validation images.   
   - **Results**: ABNN demonstrated an accuracy of 79.5% with ResNet-50 and 80.6% with ViT.
   - **Reference**: [Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." In CVPR, 2009](https://www.image-net.org/download.php)

3. **SVHN (Street View House Numbers)**:
   - **Description**: A real-world image dataset obtained from house numbers in Google Street View images.
   - **Usage**: Used as an out-of-distribution dataset for models trained on CIFAR-10/100 to test their generalization and uncertainty estimation.
   - **Dimension**: Images in SVHN, like CIFAR, are small, often 32x32 pixels.
   - **Data Split**: Contains over 600,000 images used typically for testing generalization beyond the training data of different datasets.
   - **Results**: Used as an out-of-distribution dataset to test the generalization of models trained on CIFAR-10/100. Specific performance metrics in this context were not detailed for SVHN in the paper.
   - **Reference**: [Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning." In NeurIPSW, 2011](http://ufldl.stanford.edu/housenumbers/)

4. **Describable Textures Dataset (DTD)**:
   - **Description**: A dataset of textural images organized according to a list of 47 terms (categories) inspired by human perception.
   - **Usage**: Used as an out-of-distribution dataset in experiments with models trained on ImageNet.
   - **Dimension**: Typically processed to fit the input size (224x224 pixels) requirements of the ImageNet model.
   - **Data Split**: Consists of 5,640 images categorized into 47 categories.   
   - **Results**: Employed as an OOD dataset for ImageNet trained models. Performance details specific to DTD were not detailed in the paper.
   - **Reference**: [Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. Describing textures in the wild. In Proceedings of the CVPR. 2014](https://www.robots.ox.ac.uk/~vgg/data/dtd/) 
  
5. **StreetHazards**:
   - **Description**: Part of the larger BDD100K dataset, designed specifically for benchmarking anomaly detection in the context of street scenes.
   - **Usage**: Used for assessing model performance in semantic segmentation tasks and uncertainty estimation.
   - **Dimension**: Typically used dimensions might range around 512x256 pixels becouse it includes complex street scenes.
   - **Data Split**: Comprises 5,125 training images and 1,500 testing images, including additional out-of-distribution objects.   
   - **Results**: In semantic segmentation tasks, ABNN achieved a mean IoU of 53.82% and was able to improve OOD detection, reducing FPR95 to 32.02%.
   - **Reference**: [Hendrycks, Dan, et al. "A benchmark for anomaly segmentation." arXiv preprint arXiv:1911.11132, 2019](https://github.com/hendrycks/anomaly-seg) 

6. **BDD-Anomaly**:
   - **Description**: A challenging real-world dataset for street scene segmentation that includes diverse conditions such as weather and nighttime scenes.
   - **Usage**: A subset of the BDD100K used for evaluating semantic segmentation and anomaly detection models.
   - **Dimension**: Typically used dimensions might range around 1280x720 pixels becouse it includes diverse street scenes captured at various times.
   - **Data Split**: Features 6,688 images for training and 361 for testing.   
   - **Results**: For semantic segmentation, ABNN obtained a mean IoU of 48.76% on this dataset.
   - **Reference**: Yu, Fisher, et al. "Bdd100k: A diverse driving dataset for heterogeneous multitask learning." In CVPR, 2020 .

7. **MUAD**:
   - **Description**: Specifically designed for evaluating autonomous driving systems, focusing on both normal and anomalous objects encountered in urban environments.
   - **Usage**: Employed for semantic segmentation tasks focusing on both normal and out-of-distribution scenarios.
   - **Dimension**: Typically used dimensions might range around 2048x1024 pixels becouse it includes diverse urban scenes.
   - **Data Split**: Contains 3,420 training images, 492 validation images, and 6,501 test images, which include normal and OOD samples.   
   - **Results**: ABNN showed impressive results, achieving a mean IoU of 61.96% and significantly lowering the FPR95 to 21.68%.
   - **Reference**: [Franchi, Gianni, et al. "Muad: Multiple uncertainties for autonomous driving, a benchmark for multiple uncertainty types and tasks." In BMVC, 2022](https://muad-dataset.github.io/) 

8. **CityScapes**:
   - **Description**: A dataset for semantic urban scene understanding that provides pixel-level annotations.
   - **Usage**: Commonly used for semantic segmentation tasks in urban settings.
   - **Dimension**: Images are high-resolution, typically used at 2048x1024 pixels, providing detailed urban scenes for segmentation.
   - **Data Split**: Includes 3,475 images split across training, validation, and test sets.   
   - **Results**: Specific results for ABNN on CityScapes were not detailed in the paper.
   - **Reference**: [Cordts, Marius, et al. "The cityscapes dataset for semantic urban scene understanding." In CVPR, 2016](https://www.cityscapes-dataset.com/)






## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

1. Franchi, G., Laurent, O., Leguéry, M., Bursuc, A., Pilzer, A., & Yao, A. (2024). Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models. Conference on Computer Vision and Pattern Recognition.
2. Wu, A., Nowozin, S., Meeds, E., Turner, R.E., Hernández-Lobato, J.M. & Gaunt, A.L. (2018). Deterministic variational inference for robust Bayesian neural networks. In International Conference on Learning Representations.
3. Maronas, J., Paredes, R., & Ramos, D. (2020). Calibration of deep probabilistic models with decoupled Bayesian neural networks. Neurocomputing, 407, 194-205.

# Contact

Name: Abtin Mogharabin email: atbinmogharabin@gmail.com

Name: Abduallah Damash email: 
