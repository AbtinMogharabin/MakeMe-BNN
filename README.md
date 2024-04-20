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
     - One paper employs deterministic variational inference techniques to integrate Bayesian methods into trained deterministic neural networks. It introduces closed-form variance priors for the network weights, allowing the deterministic model to handle uncertainty estimations through a robust Bayesian framework after its initial training [1].
     - Compared to this approach that requires extensive modifications to the network’s inference process to accommodate the new Bayesian priors, the "Make Me a BNN" paper introduces a method that is notably simpler and potentially faster, as it leverages existing normalization layers within pre-trained DNNs to implement Bayesian functionality.

  2. **Decoupled Bayesian Stage Approach:**
     - Another study involves a decoupled Bayesian stage applied to a pre-trained deterministic neural network. This method uses a Bayesian Neural Network to recalibrate the outputs of the deterministic model, thereby improving its predictive uncertainty without retraining the entire network from scratch [2].
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

There are mulitple datasets: 

1- CIFAR-10 and CIFAR-100 for ResNet-50 and WideResNet28-10.  https://www.cs.toronto.edu/~kriz/cifar.html

2- ImageNet whichthey reported the results for ABNN on it with ResNet-50 and ViT. https://www.image-net.org/download.php

They used the datesets and transform the initial problem into binary classification between in-distribution
and out-of-distribution data using the maximum softmax probability as the criterion.

3- SVHN dataset as the out-of-distribution dataset for OOD detection tasks on CIFAR-10 and CIFAR-100. http://ufldl.stanford.edu/housenumbers/

4- Describable Texture as the out-of-distribution dataset on ImageNet. https://www.robots.ox.ac.uk/~vgg/data/dtd/


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.

