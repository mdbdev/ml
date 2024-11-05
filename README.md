# ml
MDB Machine Learning Live Coding Demo Files for the New Member Training Program.
# Machine Learning with Classification, Regression, and Generative AI

This project is part of the Mobile Developers of Berkeley (MDB) club’s **New Member Training Program**. It provides a hands-on introduction to essential machine learning concepts, covering **regression**, **classification**, and **generative AI**. Using Python, PyTorch, and pre-trained models, this notebook allows members to explore core ML techniques without the need for high computational resources.

## Project Overview

This notebook is designed as a comprehensive guide to:

1. **Regression**: 
   - We implement **linear regression** using the least squares method for a simple approach.
   - We train a **Multilayer Perceptron (MLP)** for non-linear regression, highlighting neural networks’ flexibility in capturing complex relationships.

2. **Classification**:
   - We train an **MLP on the MNIST dataset** for handwritten digit classification, introducing the basics of feedforward neural networks.
   - We implement a **Convolutional Neural Network (CNN) on the CIFAR-10 dataset** to classify images across multiple categories, showcasing the power of CNNs for image recognition tasks.

3. **Generative AI**:
   - We introduce generative AI concepts and use a **pre-trained model** to generate images from text prompts. This section highlights generative models’ potential in creative applications and makes advanced AI accessible by leveraging pre-trained models.

By the end of this notebook, you’ll have practical experience with supervised learning (regression and classification) and generative AI concepts, forming a foundation for more advanced machine learning and AI projects.

## Dataset Information

- **MNIST**: A dataset of 28x28 grayscale images of handwritten digits (0-9), used to train and evaluate the MLP classification model.
- **CIFAR-10**: A dataset of 32x32 color images across 10 classes (e.g., airplane, car, bird), used to train the CNN for multi-class image classification.

## Notebook Sections

1. **Setup and Data Loading**:
   - Load and preprocess the MNIST and CIFAR-10 datasets.
   - Visualize samples from each dataset to understand their structure and classes.
   
2. **Regression**:
   - Implement linear regression with the least squares method.
   - Train an MLP for non-linear regression, demonstrating the use of neural networks in continuous prediction tasks.

3. **Classification**:
   - Train an MLP on the MNIST dataset for digit classification.
   - Implement and train a CNN on the CIFAR-10 dataset to classify images into multiple categories.
   - Track and plot training and testing losses to visualize model performance.

4. **Generative AI**:
   - Introduce generative models and their applications.
   - Use a pre-trained generative model (e.g., Stable Diffusion or DALL-E Mini) to generate images from text prompts.
   
5. **Visualization**:
   - Visualize sample images from the MNIST and CIFAR-10 datasets.
   - Plot training and testing losses to evaluate the models’ learning progress.

## Requirements

To run this notebook, you’ll need the following Python libraries:

- `torch`
- `torchvision`
- `matplotlib`
- `tqdm`
- `transformers` (for generative AI)
- `diffusers` (for Stable Diffusion or DALL-E Mini)

Install dependencies with:
```bash
pip install torch torchvision matplotlib tqdm transformers diffusers
```

## Usage

	1.	Clone this repository and navigate to the project directory.
	2.	Open the notebook in Jupyter or Google Colab.
	3.	Follow each cell in sequence to load data, train models, and visualize results.

## Results and Visualizations

Throughout the notebook, you’ll find plots of training and testing losses, visualizations of sample images from the datasets, and outputs from the generative models. These results will help you:

	•	Understand how the models are learning and generalizing.
	•	Gain insight into the application of generative AI for creative tasks.

This project provides a solid foundation in machine learning and AI, introducing MDB members to essential techniques and encouraging further exploration and experimentation.

