# Plant Disease Detection Using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to classify plant leaf images into 38 categories of plant diseases and healthy leaves. It uses the PlantVillage dataset and TensorFlow/Keras for model development and training.

## Objectives
- Detect plant diseases from leaf images.
- Classify leaves into 38 disease and healthy categories.
- Apply image preprocessing and data augmentation.
- Evaluate the trained model using validation and test datasets.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Hugging Face Datasets
- Google Colab

## Dataset
The project uses the PlantVillage dataset, containing 54,304 images across 38 classes. Images are resized to 128 × 128 pixels and normalized before being fed into the CNN.

Dataset source: [PlantVillage Full Dataset on Hugging Face](https://huggingface.co/datasets/geraldmc/plantvillage-full)

## Model Architecture
The CNN includes:
- Image augmentation using random flipping, rotation, and zoom
- Three convolutional layers with 32, 64, and 128 filters
- Max-pooling layers
- Global average pooling
- A dense layer with dropout
- A 38-unit softmax output layer

## Results
- **Test Accuracy:** 87.58%
- **Validation Accuracy:** 87.99%
- **Test Loss:** 0.3896

Training and validation accuracy and loss graphs are included in this repository.

## Project Files
- `Plant_Disease_Detection.ipynb` — notebook containing dataset preparation, model building, training, and evaluation
- `plant_disease_model.keras` — saved trained model, if included
- `class_names.json` — class labels in prediction order
- `accuracy_plot.png` — accuracy graph
- `loss_plot.png` — loss graph

## Limitations
The model is trained on PlantVillage images, which are often captured under controlled conditions. Its performance on real-world field images may differ. Predictions should be treated as estimates rather than definitive diagnoses.

## Future Improvements
- Evaluate performance across individual disease classes.
- Test on diverse, real-world leaf images.
- Explore transfer learning to improve generalization.
