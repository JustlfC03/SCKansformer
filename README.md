# BMC-FGCD

## 1. Abstract

In this study, we established the first high-quality Bone Marrow Cell Fine-Grained Classification Dataset (BMC-FGCD),
comprising nearly 100,000 samples and encompassing 40 detailed classifications. Accurate classification of malignant diseases,
such as acute leukemia, is critical in medical diagnosis. Traditionally, peripheral blood smears have been utilized to conduct cytology on a limited scope,
typically examining only three to five leukocyte types. This approach is used to determine the specific leukemia subtype to which a patient belongs,
thereby informing targeted treatment strategies. However, for precise diagnosis—particularly for identifying diseases with significant risks—the exclusive reliance on peripheral blood smears is insufficient.
Bone marrow smear cytomorphology is required to supplement the diagnostic process. Consequently, our dataset provides comprehensive information on 40 types of bone marrow aspiration blood cell.
The construction of the dataset involved a meticulous process of sample acquisition, image processing, and high-precision labeling performed by the professional physicians.
Furthermore, in the baseline experiments, we employed nine well-regarded deep-learning models for evaluation. The results indicated that the ViT model exhibited commendable performance,
as evidenced by its accuracy, precision, recall, and F1 scores. The BMC-FGCD dataset holds the potential not only to enhance the performance of future bone marrow blood cells classification models but also to offer clinicians a more comprehensive diagnostic tool in medical practice.
This is particularly crucial for the classification of rare or atypical blood cells, where conventional diagnostic methods may fall short.
![image](img/fig1combined.png)

## 2. Methods

Workflow of the establishment of the proposed dataset. (**a**) Data were extracted from patients' bone marrow fluid samples obtained at the Department of Hematology at Zhejiang Hospital.
The preparation and examination of the smears were carried out using two types of Olympus microscopes. (**b**) Images deemed to be of low quality underwent standardization through image processing software and systems to ensure their suitability for algorithmic processing.
(**c**) Expert physicians at Zhejiang Hospital meticulously identified and manually annotated each image with the respective cell type. Ultimately, 40 types of bone marrow blood cells were classified within the dataset.
![image](img/fig2process.png)

## 3. Dataset

If you want to use our private dataset, please cite our article.

Donwload link of Bone Marrow Cell Fine-Grained Classification Dataset (BMC-FGCD) is available at [https://drive.google.com/file/d/1SWXL4V5iH--KBAex-H7Fbq1CMrqF6_Oh/view?usp=drive_link](https://drive.google.com/file/d/1SWXL4V5iH--KBAex-H7Fbq1CMrqF6_Oh/view?usp=drive_link).

## 4. Usage

**Below, we delineate the specific utility of this dataset in various application contexts:**

- Training of Deep Learning Models and Automatic Blood Cell Identification.
- Integrated Diagnosis with Clinical Data.
- Identification of Rare and Atypical Blood Cells.

## 5. Citation
```
Yifei Chen
```
