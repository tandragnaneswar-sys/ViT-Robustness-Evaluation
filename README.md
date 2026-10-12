# Robust Crop–Weed Classification Using Vision Transformers Under Image Distortions

## 1. Project Overview

Crop–weed classification is an important task in precision agriculture. Weeds compete with crops for essential resources such as water, nutrients, sunlight, and space. Therefore, identifying crops and weeds accurately can help improve agricultural management.

Deep learning models can classify plant images, but their performance may change when images are affected by noise, blur, illumination changes, compression, or geometric transformations.

This project evaluates the robustness of a pretrained Vision Transformer (ViT-B/16) for binary crop–weed classification using the Plant Seedlings Dataset.

The model is evaluated on clean test images and images affected by different distortions. Each distortion is tested at five severity levels using the same trained model without retraining.

The main objective is to understand how different image distortions affect the classification performance of the Vision Transformer.

## 2. Objectives

The main objectives of this project are:

- To develop a Vision Transformer-based binary crop–weed classification model.
- To use transfer learning with a pretrained ViT-B/16 model.
- To evaluate model performance on clean test images.
- To generate distorted images using nine different distortion techniques.
- To evaluate each distortion at five severity levels.
- To calculate accuracy, precision, recall, and F1-score.
- To analyze classification errors using confusion matrices.
- To compare the robustness of the model across different distortion types.

## 3. Dataset

### Plant Seedlings Dataset

This project uses the Plant Seedlings Dataset, which contains images of 12 plant species.

The original species are grouped into two categories:

- **Crop:** Maize, Common Wheat, and Sugar Beet.
- **Weed:** Black-grass, Charlock, Cleavers, Common Chickweed, Fat Hen, Loose Silky-bent, Scentless Mayweed, Shepherd's Purse, and Small-flowered Cranesbill.

### Dataset Distribution

| Dataset Split | Crop Images | Weed Images | Total Images |
|---|---:|---:|---:|
| Training | 681 | 3,196 | 3,877 |
| Validation | 146 | 685 | 831 |
| Testing | 146 | 685 | 831 |
| **Total** | **973** | **4,566** | **5,539** |

The dataset is imbalanced because the number of weed images is greater than the number of crop images.

### Preprocessing

The following preprocessing operations are used:

- Images are resized to 224 × 224 pixels.
- Images are converted into tensors.
- Random horizontal flipping and rotation are used for training augmentation.
- Validation and test images are resized and converted into tensors without random augmentation.

## 4. Model Architecture

The project uses a pretrained Vision Transformer Base model with a patch size of 16 (ViT-B/16).

### Main Components

1. **Input Image:** An RGB image resized to 224 × 224 pixels.
2. **Patch Embedding:** The image is divided into 196 non-overlapping patches of size 16 × 16.
3. **Linear Projection:** Each patch is converted into an embedding.
4. **Positional Embedding:** Positional information is added to the patch embeddings.
5. **Transformer Encoder:** Self-attention and feed-forward layers learn relationships between image regions.
6. **Classification Head:** The final representation is used to predict one of two classes: Crop or Weed.

The pretrained model is fine-tuned for binary crop–weed classification.

## 5. Image Distortions

Nine image distortion techniques are evaluated.

| Category | Distortion |
|---|---|
| Noise | Gaussian Noise |
| Noise | Salt-and-Pepper Noise |
| Blur | Gaussian Blur |
| Blur | Motion Blur |
| Lighting | Brightness Variation |
| Lighting | Contrast Variation |
| Compression | JPEG Compression |
| Geometric | Rotation |
| Geometric | Scaling |

Each distortion is generated at five severity levels, from Level 1 to Level 5.

The distorted test images are evaluated using the same trained model without retraining.

## 6. Experimental Workflow

The project follows these steps:

1. Load the Plant Seedlings Dataset.
2. Group the plant species into Crop and Weed classes.
3. Split the data into training, validation, and testing sets.
4. Preprocess the images.
5. Load the pretrained ViT-B/16 model.
6. Fine-tune the model for binary classification.
7. Evaluate the model on clean test images.
8. Generate distorted test datasets for all nine distortion types.
9. Apply five severity levels to each distortion.
10. Evaluate the same trained model on every distorted dataset.
11. Calculate classification metrics and confusion matrices.
12. Compare the results to study model robustness.

## 7. Evaluation Metrics

The following metrics are used to evaluate the model.

### Accuracy

Measures the proportion of correctly classified images.

Accuracy = Correct Predictions / Total Predictions

### Precision

Measures how many predictions for a class are correct.

Precision = True Positives / (True Positives + False Positives)

### Recall

Measures how many actual samples of a class are correctly identified.

Recall = True Positives / (True Positives + False Negatives)

### F1-Score

The harmonic mean of precision and recall.

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

### Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions for each class.

It helps identify whether crop images are incorrectly classified as weeds or weed images are incorrectly classified as crops.

Weighted and macro averages are also calculated to assess performance on the imbalanced dataset.

## 8. Experimental Results

The model is evaluated on the clean test dataset and on all distorted test datasets.

The evaluation records:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrix
- Class-wise classification report

Results are compared across distortion types and severity levels to identify which image conditions affect the model most strongly.

The final numerical results and conclusions should be taken from the verified outputs of the completed experiments.

## 9. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Scikit-learn
- Pandas
- Matplotlib
- Google Colab
- LaTeX / Overleaf

## 10. Project Structure

A suggested project structure is:

```text
Crop-Weed-ViT-Robustness/
│
├── README.md
│
├── notebooks/
│   └── vit_robustness_evaluation.ipynb
│
├── results/
│   ├── rotation_evaluation_results.csv
│   ├── scaling_evaluation_results.csv
│   └── other_evaluation_results.csv
│
├── figures/
│   ├── vit_architecture.png
│   ├── accuracy_comparison.png
│   └── confusion_matrix.png
│
├── report/
│   └── research_report.pdf
│
└── requirements.txt
