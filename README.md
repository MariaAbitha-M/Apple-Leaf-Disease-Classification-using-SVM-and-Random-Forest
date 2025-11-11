🍎 Apple Leaf Disease Classification Using SVM and Random Forest
📖 Overview

This project focuses on the detection and classification of apple leaf diseases using machine learning algorithms. By analyzing images of apple leaves, the system can identify whether a leaf is healthy or affected by diseases such as Apple Scab, Black Rot, or Cedar Apple Rust.

The project uses Support Vector Machine (SVM) and Random Forest classifiers to achieve accurate predictions, providing farmers and researchers with an efficient tool for early disease detection and crop management.

🎯 Objectives

Detect apple leaf diseases automatically from images.

Compare classification performance between SVM and Random Forest models.

Improve agricultural productivity by enabling early disease diagnosis.

🧠 Technologies Used

Programming Language: Python

Libraries & Frameworks:

NumPy

Pandas

Matplotlib

Scikit-learn

OpenCV

IDE: Jupyter Notebook / VS Code

📊 Dataset

Dataset Name: Apple Leaf Disease Dataset

Source: [Kaggle / Custom Collected Dataset] (replace with actual source)

Classes:

Apple Scab

Black Rot

Cedar Apple Rust

Healthy

⚙️ Project Workflow
1. Data Collection & Preprocessing

Loaded the dataset and performed data cleaning.

Resized images and converted them into feature vectors using image processing techniques (e.g., grayscale, histogram, edge detection).

2. Feature Extraction

Extracted color, texture, and shape features.

Normalized feature data for training consistency.

3. Model Training

Trained models using Support Vector Machine (SVM) and Random Forest classifiers.

Tuned hyperparameters for better performance.

4. Model Evaluation

Evaluated models using metrics such as accuracy, precision, recall, and F1-score.

Compared SVM vs. Random Forest results.

5. Result Visualization

Displayed confusion matrices and performance graphs using Matplotlib.

📈 Results
Model        	Accuracy	  Precision  	Recall	 F1-Score
SVM	            94%          93%	     94%    	93%
Random Forest	  96%	         95%	     96%	    95%
