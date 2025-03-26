# DiffV2IQA

This repository contains the official implementation of **DiffV2IQA**:  
**Diffusion Model Based Visual Compensation Guidance and Visual Difference Analysis for No-Reference Image Quality Assessment**  
**Authors**: Zhaoyang Wang, Bo Hu, Mingyang Zhang, Jie Li, Leida Li, Maoguo Gong, Xinbo Gao

> The complete codebase, including pretrained weights, will be released upon acceptance of the paper.

---

## 🧠 Overview

**DiffV2IQA** is the first approach to leverage **diffusion models** for **No-Reference Image Quality Assessment (NR-IQA)**. It addresses the limitations of previous free-energy-guided methods by introducing a visually interpretable and semantically rich guidance mechanism derived from diffusion-based image enhancement. The proposed method restores distorted images using a novel diffusion model and guides the quality assessment process with clearer high-level features.  

---

## 🧱 Network Architecture  
![main_model.png](main_model.png)

---

## 📌 Motivation

> *Despite recent advances, existing free-energy-based NR-IQA methods struggle with restoring heavily distorted images and suffer from limited interpretability. The extracted features often fail to capture meaningful high-level semantics, making it difficult to accurately assess image quality.*

DiffV2IQA addresses this gap by introducing a **diffusion-based restoration framework** that generates high-quality, semantically aligned representations. The intermediate states in the denoising process provide clearer and more meaningful visual guidance, allowing the network to better understand and assess image degradation. A **dual-branch architecture** is then designed to collaboratively exploit these features for accurate quality prediction.

---

## 🌟 Key Contributions

- ✅ **First Application of Diffusion Models in NR-IQA**: We pioneer the integration of diffusion models into the NR-IQA field, achieving both better restoration and quality estimation.
- ✅ **Improved Interpretability & Visual Guidance**: The intermediate features from the diffusion process closely resemble the human visual system’s self-repair behavior, enhancing both interpretability and performance.
- ✅ **Dual-Branch Quality Assessment Framework**: Our architecture includes two complementary branches—one for restoration, one for evaluation—enabling collaborative and efficient NR-IQA.

---

## 🚀 Getting Started

### 🔧 Prerequisites

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 📷 Step 1: Generate Visual Compensation Images

Run the following script to generate **high-level visual compensation images** using the diffusion model:

```bash
python diffusion/sr.py
```

Adjust configuration parameters in the corresponding config file as needed.

---

### 🧪 Step 2: Train the Network

Train the full quality assessment network by running:

```bash
python main.py
```

Make sure all necessary file paths and hyperparameters are properly set in the config file before training.

---

## 📂 Pretrained Models

Pretrained weights for the diffusion module and NR-IQA model will be made available in future updates.

---

## 📌 Acknowledgements

This work builds upon the foundations of **SR3**, **MANIQA**, and **VCRNet**. We sincerely thank the authors of these projects for their valuable contributions and open-source implementations.

---


