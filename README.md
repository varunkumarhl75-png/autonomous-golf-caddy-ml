# Autonomous Golf Caddy – Traversability Prediction

## 1. Project Overview

This mini project develops a machine-learning based prototype for an autonomous golf caddy.

The system takes a single RGB camera image and predicts which image regions are **traversable**. The predicted traversability map is then used by a simple region-based navigation heuristic to generate one of four decisions:

- FORWARD
- LEFT
- RIGHT
- STOP

The project is inspired by the problem formulation of autonomous golf-caddy navigation, but the model and implementation in this repository are our own.

---

## 2. Problem Statement

An autonomous golf caddy needs to understand the terrain visible in front of it and determine where it can safely move.

The objective of this project is to:

1. Process a camera image.
2. Predict traversable and non-traversable regions.
3. Use the prediction to estimate whether the left, center, or right region is suitable for movement.
4. Produce a basic navigation decision.

This is a prototype navigation system rather than a complete physical autonomous-robot controller.

---

## 3. Dataset

The project uses the provided golf-caddy image dataset containing RGB images and corresponding labelled supervision images.

The labelled images use semantic colors representing different scene classes. For this project, the classes are converted into a binary task:

- **Traversable** = traversable terrain + pavement
- **Non-traversable** = all remaining labelled classes

Only the labelled image pairs that could be matched with their original RGB images were used for supervised training.

### Dataset split

There are 11 matched labelled image pairs used in the experiment.

- **9 images** → training
- **2 images** → held-out testing

Test images:

- `left25057.jpg`
- `left25343.jpg`

Because the held-out test set contains only two images, the reported metrics should be considered experimental results for this mini project and not a claim of broad real-world generalization.

---

## 4. Label Processing

The original supervision images contain several semantic colors.

The relevant mapping used in this project is:

| Semantic class | Binary label |
|---|---|
| Traversable terrain | Traversable |
| Pavement | Traversable |
| Untraversable terrain | Non-traversable |
| Putting green / tee box | Non-traversable |
| Target golfer | Non-traversable |
| Sky / clouds | Non-traversable |

The binary formulation was selected because the main navigation requirement is to distinguish terrain that can be used for movement from terrain that should not be treated as traversable.

---

## 5. Preprocessing

Each RGB image is resized to:

**496 × 279 pixels**

For every pixel, a **3 × 3 neighborhood** is extracted.

Each pixel in the neighborhood has three RGB values:

`3 × 3 × 3 = 27 features`

Therefore, every training example contains **27 input features**.

The center pixel of each 3 × 3 neighborhood receives the corresponding binary ground-truth label.

For a 496 × 279 image:

`(496 - 2) × (279 - 2) = 494 × 277 = 136,838`

usable pixel-neighborhood samples are produced.

---

## 6. Training Data Balancing

The original pixel distribution contains substantially more non-traversable samples than traversable samples.

To reduce class imbalance during training:

- 100,000 traversable samples were selected.
- 100,000 non-traversable samples were selected.
- Total balanced training samples = **200,000**
- Random seed = **42**

The test set was not balanced or sampled in this way; it was kept as the complete held-out image data.

---

## 7. Models

Two models were evaluated.

### Logistic Regression

Logistic Regression was used as the baseline model.

Results:

- Accuracy: **57.36%**
- Precision: **35.75%**
- Recall: **93.10%**
- F1-score: **51.66%**

### Random Forest

The final selected model is a Random Forest classifier.

Configuration:

- Number of trees: **100**
- Maximum depth: **20**
- Random state: **42**
- Parallel training: `n_jobs=-1`

Results:

- Accuracy: **77.31%**
- Precision: **52.16%**
- Recall: **87.95%**
- F1-score: **65.48%**

Random Forest produced a substantially better overall F1-score and accuracy than the Logistic Regression baseline.

---

## 8. Prediction Pipeline

The final prediction pipeline is:

```text
Input RGB Image
      ↓
Resize to 496 × 279
      ↓
3 × 3 RGB Neighborhood Extraction
      ↓
27 Features per Pixel
      ↓
Random Forest
      ↓
Binary Traversability Prediction
      ↓
Traversability Mask
```

The trained model is stored at:

```text
src/model/random_forest_model.pkl
```

---

## 9. Navigation Heuristic

The binary prediction mask is converted into a simple navigation decision.

Only the **bottom 45%** of the prediction mask is considered because this region represents terrain closer to the caddy.

The selected region is divided into:

```text
LEFT | CENTER | RIGHT
```

The percentage of predicted traversable pixels is calculated for each region.

A center-priority rule is then applied:

- If center traversability ≥ 60% → **FORWARD**
- Otherwise, if left is higher than right and left ≥ 60% → **LEFT**
- Otherwise, if right is higher than left and right ≥ 60% → **RIGHT**
- Otherwise → **STOP**

The 60% threshold is a prototype heuristic. The dataset does not contain explicit ground-truth navigation labels, so this navigation rule has not been experimentally optimized against labelled driving decisions.

---

## 10. Demo Results

The prediction demo was run on the two held-out test images.

### `left25057.jpg`

```text
Left:     88.72%
Center:   71.26%
Right:    87.56%
Decision: FORWARD
```

### `left25343.jpg`

```text
Decision: STOP
```

The second result demonstrates an important limitation of the current prototype: the pixel classifier can produce noisy predictions in scenes containing vegetation, people, and complex backgrounds.

---

## 11. Repository Structure

```text
ML mini project/
│
├── Data/
│   └── Data/
│       ├── Original RGB images
│       └── Label/supervision images
│
├── notebooks/
│   └── Training and experimentation notebook
│
├── src/
│   ├── preprocessing/
│   │
│   ├── model/
│   │   └── random_forest_model.pkl
│   │
│   └── prediction/
│       ├── predict.py
│       └── demo.py
│
├── output data set/
│
└── README.md
```

---

## 12. Running the Project

### Requirements

Install the required Python packages:

```bash
pip install numpy pillow scikit-learn joblib matplotlib
```

### Run a single prediction

From the project root:

```bash
python src/prediction/predict.py
```

### Run the demo

```bash
python src/prediction/demo.py
```

The demo displays the input image and the predicted binary traversability mask.

---

## 13. Important Project Constraints

This project does **not** use:

- YOLO
- Pretrained object-detection models
- Pretrained neural-network weights
- A pretrained segmentation model

The Random Forest model is trained from the provided labelled image data.

The reference material is used for the problem formulation and semantic interpretation of the dataset; the implementation, binary classification approach, model training, evaluation, and navigation heuristic are developed for this project.

---

## 14. Limitations

1. Only 11 labelled image pairs were available for supervised training/testing.
2. Only two images are used as the held-out test set.
3. The binary model combines traversable terrain and pavement into one class.
4. The model can produce noisy pixel-level predictions.
5. No explicit navigation labels are available.
6. The LEFT/RIGHT/FORWARD/STOP logic is therefore a heuristic rather than a learned navigation policy.
7. The current prototype processes individual images rather than a continuous camera/video stream.
8. The project does not include physical motor control or robot hardware integration.

---

## 15. Future Work

Possible improvements include:

- Collecting or obtaining more labelled images.
- Increasing the diversity of lighting, terrain, and viewpoints.
- Using a richer multi-class terrain classifier.
- Improving spatial features beyond a 3 × 3 RGB neighborhood.
- Reducing prediction noise with post-processing.
- Learning navigation decisions from explicitly labelled driving examples.
- Testing on continuous video.
- Integrating the prediction system with a real caddy platform.

---

## 16. Conclusion

This project demonstrates an end-to-end prototype for image-based terrain traversability prediction.

A Random Forest classifier trained on 3 × 3 RGB neighborhoods achieved:

**77.31% accuracy and 65.48% F1-score**

on the held-out image samples used in this experiment.

The resulting binary prediction is further processed using a simple region-based heuristic to produce navigation decisions. The implementation provides a foundation for future work with larger datasets, improved spatial modelling, and explicit navigation supervision.
