# Joint Detection of AI-Generated Images and Post-Processing Alterations

## Overview

This project explores the use of Multi-Task Learning (MTL) for image forensics. The objective is to train a single deep learning model capable of simultaneously:

1. Detecting whether an image is **real** or **AI-generated**.
2. Identifying the type of post-processing transformation applied to the image:

   * Original
   * Internet Transfer
   * Screen Re-digitization

The project was developed as part of the **Computer Vision** course in the Master of Science in Artificial Intelligence at Sapienza University of Rome.

---

## Dataset

The experiments were conducted using a balanced subset of the RRDataset.

To reduce computational requirements while maintaining class balance, 300 images were selected from each category, resulting in a total of 1,800 images.

Dataset distribution:

| Category            | Images |
| ------------------- | -----: |
| Real Images         |    900 |
| AI-Generated Images |    900 |
| Total               |  1,800 |

The dataset was split into:

* Training set: 1,440 images (80%)
* Validation set: 360 images (20%)

---

## Dataset and Reproducibility

The dataset used in this project is not included in this repository due to its size.

Dataset folder:

(https://drive.google.com/drive/folders/1pzCS_3iHQTbyWkKazOEZXYJJD53Lc-8f?usp=sharing)

### Running the Notebook

This project was developed and tested using Google Colab.

To reproduce the experiments:

1. Download the dataset from the Google Drive folder linked above.
2. Upload the dataset to your own Google Drive.
3. Open the notebook in Google Colab.
4. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Update the dataset path variable if necessary:

```python
DATASET_PATH = "/content/drive/MyDrive/RRDataset"
```
Here you have to copy the path where you have your drive folder.

6. Run all notebook cells sequentially.

### Notes

The exact Google Drive location may differ depending on where the dataset is stored. If needed, update the dataset path accordingly before running the notebook.


## Methodology

### Data Preprocessing

All images were:

* Resized to 224 × 224 pixels
* Converted to PyTorch tensors
* Normalized using ImageNet statistics

### Multi-Task Learning Framework

Instead of training separate models, a single neural network was designed to solve both tasks simultaneously.

The architecture consists of:

* Shared feature extractor:

  * Pre-trained ResNet18 backbone
* Task-specific heads:

  * Binary classification head (Real vs AI-generated)
  * Multi-class transformation head (Original, Internet Transfer, Re-digitized)

This design allows both tasks to benefit from shared visual representations while reducing computational cost.

---

## Model Architecture

```text
Input Image
     │
     ▼
Pretrained ResNet18 Backbone
     │
     ├──────────────► Real / AI Classification Head
     │
     └──────────────► Transformation Classification Head
```

The original fully connected layer of ResNet18 was replaced by two independent classification heads.

---

## Training Configuration

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Optimizer     | Adam                       |
| Learning Rate | 1e-4                       |
| Batch Size    | 32                         |
| Epochs        | 10                         |
| Loss Function | Cross Entropy              |
| Loss Weights  | λclass = 0.5, λtrans = 0.5 |

The total optimization objective is defined as:

Ltotal = λclass Lclass + λtrans Ltrans

where both tasks contribute equally to the final loss.

---

## Evaluation

The following metrics were computed for the Real/Fake classification task:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Additionally, qualitative visual inspections were performed on randomly selected validation images.

---

## Results

### Real vs AI Detection

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Real         |      0.85 |   0.89 |     0.87 |
| AI-generated |      0.90 |   0.87 |     0.88 |

**Overall Accuracy: 88%**

The results demonstrate that the proposed multi-task framework can learn meaningful forensic representations even under challenging real-world conditions involving internet transmission and re-digitization.

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows a balanced performance across both classes, with similar precision and recall values for real and AI-generated images.

---

## Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── P2CV.ipynb
├── images/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
└── requirements.txt
```

---

## Requirements

Main dependencies:

```bash
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Authors

Paola Cortés Domínguez  
Carmen Rojas Gallego

---

## Course Information

**Computer Vision**

Master of Science in Artificial Intelligence

Sapienza University of Rome

Professor: Irene Amerini
