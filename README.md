# fruit-classification with HOG, LBP & Histograms

📌 Description

Classical image classification project on fruit images using hand-crafted features (HOG, LBP, grayscale/RGB histograms) and machine learning models (SVM and Gaussian Naive Bayes).
The goal is to engineer robust features, compare models, and visualize results with confusion matrices.

🎯 Objectives

Perform feature engineering (HOG, LBP, Histograms) on small image datasets.

Train and compare SVM (RBF) vs Naive Bayes.

Evaluate with accuracy and macro Precision / Recall / F1.

Visualize confusion matrices and key examples.

📊 Dataset

Folder-based dataset organized by class (e.g., apple/, banana/, orange/).
Images are resized and (optionally) augmented (grayscale copy, horizontal flip, ±15° rotation) to improve robustness and class balance.

⚙️ Tech Stack

Python

Libraries: OpenCV, scikit-image, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

Scripts/CLI: feature extraction, training, evaluation

🔎 Methodology

Data Preparation

Resize to a fixed size (e.g., 128×128).

Optional augmentation: grayscale, flips, small rotations.

Feature Extraction

Grayscale histogram (e.g., 16 bins) and RGB/HSV histograms (e.g., 8 bins/channel).

LBP (uniform) with P=8, R=1 → compact texture histogram.

HOG with 9 orientations, 8×8 cells, 2×2 blocks.

Concatenate and normalize all features per image.

Modeling

SVM (RBF) with a small grid on C and gamma.

Gaussian Naive Bayes as a fast baseline (pipelines with standardization).

Evaluation

Stratified split (train/val/test).

Metrics: accuracy, macro-P/R/F1, and confusion matrices saved to reports/figures/.

📈 Results

SVM (RBF) typically achieves higher macro-F1 than Naive Bayes when using HOG + LBP + Histograms.
