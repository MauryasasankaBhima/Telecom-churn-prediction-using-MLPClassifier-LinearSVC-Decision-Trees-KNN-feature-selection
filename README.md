# Telecom-churn-prediction-using-MLPClassifier-LinearSVC-Decision-Trees-KNN-feature-selection
This project focuses on classifying telecom customers based on their likelihood to churn using multiple machine learning models including MLP (Neural Network), SVM, Decision Trees, and KNN. It integrates feature importance analysis and hyperparameter optimization.
--

##  Objective

To build an accurate classification model that predicts customer churn in the telecom industry by leveraging powerful ML models and tuning them with `RandomizedSearchCV`.

---

##  Tools & Techniques Used

| Task                         | Tool / Algorithm                |
|------------------------------|----------------------------------|
| Classification Models        | MLPClassifier, LinearSVC, DecisionTreeClassifier, KNeighborsClassifier |
| Model Optimization           | RandomizedSearchCV              |
| Feature Analysis             | Feature Importance              |
| Evaluation Metrics           | Accuracy, Precision, Recall, F1 Score, ROC-AUC |
| Libraries                    | scikit-learn, pandas, NumPy, matplotlib, seaborn |

---

##  Workflow Summary

1. **Data Cleaning & Preprocessing**
   - Label encoding for categorical features
   - Feature scaling (StandardScaler)

2. **Model Training**
   - Trained various models including:
     - Multi-layer Perceptron (MLPClassifier)
     - Linear Support Vector Classifier (LinearSVC)
     - Decision Trees with `RandomizedSearchCV`
     - K-Nearest Neighbors

3. **Evaluation**
   - Compared models using classification reports
   - Evaluated performance using ROC-AUC and confusion matrix

4. **Feature Importance**
   - Identified key features contributing to churn decisions

---

## 📊 Sample Results



| Model             | Accuracy | F1 Score | ROC-AUC |
|-------------------|----------|----------|---------|
| MLPClassifier      | 87.6%    | 0.86     | 0.89    |
| LinearSVC          | 85.2%    | 0.84     | 0.87    |
| Decision Tree (tuned) | 88.9% | 0.88     | 0.90    |

---

##  Highlights

- Used multiple algorithms for churn detection
- Tuned models using `RandomizedSearchCV`
- Analyzed top predictive features using feature importance
- Balanced precision and recall for imbalanced churn data

---
