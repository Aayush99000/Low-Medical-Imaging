# Low-Medical-Imaging

This project tackles one of medical AI challenges: developing reliable diagnostic models when data is scarce. Through rigorous experimentation, we empirically validate whether transfer learning can overcome the "small data problem" that plagues specialized medical imaging applications. By comparing models trained from scratch against those leveraging pre-trained knowledge, we provide concrete evidence for best practices in low-resource clinical AI deployment.

# 🎯 Problem Statement
Medical AI development faces a critical data scarcity crisis. Creating effective diagnostic tools requires large, expertly annotated datasets, which are often impossible to obtain due to:

🔒 HIPAA privacy regulations limiting data sharing
💰 High annotation costs requiring medical expert time
🏥 Data rarity for specialized conditions and rare diseases

Training complex CNNs from scratch on small datasets leads to severe overfitting and unreliable performance on new patients. Transfer learning offers a promising solution, but requires rigorous empirical validation.

# 🔬 Project Overview
This project provides empirical validation of transfer learning efficacy in low-resource medical imaging scenarios through a controlled three-way comparison:
Three Model Architecture Comparison
ModelDescriptionPurposeModel A (Scratch)Simple CNN trained only on small datasetBaseline performanceModel B (ImageNet Transfer)ResNet-50 pre-trained on ImageNet, fine-tuned on medical dataGeneral feature transferModel C (Medical Transfer)Model pre-trained on broader medical corpusDomain-specific transfer
All models are evaluated under identical conditions to ensure fair comparison.

We utilize three distinct medical imaging datasets to simulate real-world transfer scenarios:  
**1. Chest X-ray Images (Pneumonia)**
Source: Guangzhou Women and Children's Medical Center  
Task: Pneumonia detection from chest radiographs  
Link: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

**2. HAM10000 Dermatoscopic Images**
Source: Harvard Dataverse  
Task: Classification of common pigmented skin lesions  
Link: [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)  

**3. CheXpert Dataset**
Source: Stanford ML Group  
Task: Multi-label chest radiograph classification  
Link: [Stanford ML Group](https://arxiv.org/abs/1901.07031)  

**Data Strategy**

Training Set: Deliberately constrained to 100-200 labeled images per condition  
Test Set: Large, unbiased holdout for reliable evaluation  
Augmentation: Aggressive augmentation  

# Dataset Description

**1.Chest X-Ray Images (Pneumonia)**  
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.  
<img width="1084" height="551" alt="Screenshot 2025-11-12 at 15 57 13" src="https://github.com/user-attachments/assets/fed6b50f-50a9-4899-ad63-21ddc3ce958a" />  

**2.HAM10000 Dermatoscopic**
The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
<img width="1049" height="706" alt="Screenshot 2025-11-12 at 16 03 42" src="https://github.com/user-attachments/assets/796b5aec-d7f8-4826-867e-e537760fa488" />



