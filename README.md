<h1 align="center">🔍 Model Interpretability via SHAP</h1>
<p align="center">
  <b>Explaining Deep Learning & Machine Learning Decisions with Partition-based Image SHAP & KernelSHAP</b><br/>
  <i>YAP 490 — Academic Research & Project Report</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/XAI-SHAP-brightgreen" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Status-In%20Preparation-orange" alt="Status"/>
</p>

---

## 📌 Abstract

This study focuses on the application of SHAP (Shapley Additive Explanations) to interpret complex machine learning models, specifically focusing on a MobileNetV2 and an SVM model for tabular data. 

For image classification, a partition-based Image SHAP approach was utilized, where pixel-level attributions were aggregated into $28 \times 28$ fixed-size grid patches to enhance interpretability and reduce noise. For the tabular dataset, KernelSHAP was applied to reveal class-specific feature influences and address the effects of class imbalance. 

The results indicate that while the image model achieves high accuracy, SHAP analysis reveals a significant reliance on background regions rather than object-specific features. Furthermore, the study demonstrates that class-specific feature attribution and correlation-based confidence scores provide deeper insights into the decision strategies of multiclass models. Despite computational costs and sensitivity to background cues, the findings suggest that careful aggregation and visualization of SHAP values can lead to meaningful local explanations for both image and tabular data.

---

## 🏷️ Key Keywords

`Explainable AI (XAI)` • `SHAP` • `Image Classification` • `Model Interpretability` • `Feature Attribution` • `KernelSHAP`

---

## 📊 Feature Attribution & Visual Results

Below are the feature attribution visualizations generated via Partition-based Image SHAP, illustrating the model's reliance on specific grid regions for classification.

<p align="center">
  <img src="assets/shap_result_1.png" alt="SHAP Explanation - Object vs Background Attribution" width="420"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/shap_result_2.png" alt="SHAP Explanation - Feature Attribution Map" width="420"/>
</p>

<p align="center">
  <i>Figure 1: SHAP patch attributions highlight spatial features contributing to the model's confidence scores, exposing critical background reliance during inference.</i>
</p>

---

## 💡 Key Takeaways & Conclusions

- **Background Bias Detection:** High validation accuracy can be deceptive; SHAP analysis exposed that the model occasionally relies heavily on background context rather than foreground features.
- **Noise Reduction:** Aggregating pixel-level SHAP values into $28 \times 28$ patch partitions effectively reduces attribution noise, yielding clearer local explanations.
- **Multimodal Flexibility:** Combining Partition-based SHAP for visual models and KernelSHAP for tabular features provides robust diagnostic insights across different data modalities.

---

## 📝 Publication & Access Note

> **Note:** This research is currently in preparation for academic publication. Source code, dataset details, and trained model checkpoints are available upon reasonable request for academic evaluation.
