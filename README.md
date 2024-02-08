# BMC-FGCD

## 1. Abstract

When physicians diagnose malignant tumors, such as acute leukemia, the rapid and precise classification of the disease is imperative for guiding subsequent treatment protocols. Traditional diagnostic methods, which rely on peripheral blood smears, are limited to assessing a narrow spectrum of cell types, thereby constraining the accurate determination of the disease subtype. Hence, a more exhaustive cytomorphologic analysis, supplemented by a bone marrow smear, is essential to devise a tailored treatment approach. Given the scarcity of comprehensive bone marrow blood smear datasets, we have developed the inaugural high-quality bone marrow cell fine-grained classification dataset (BMC-FGCD). This dataset comprises nearly 100,000 samples, encompassing 40 detailed categories. The assembly of the BMC-FGCD was overseen by a cadre of expert physicians, who meticulously managed the sample collection, image processing, and high-precision annotation. Furthermore, we employed nine well-established deep learning models to conduct baseline evaluations. These experiments provided a uniform benchmark for the assessment of bone marrow blood cell classification models. The BMC-FGCD dataset is poised not only to enhance the accuracy of bone marrow blood cell classification models in the future but also to significantly contribute to the field of rare or atypical blood cell classification due to its granularity and comprehensive data characteristics.
![image](img/fig1combined.png)

## 2. Methods

Workflow of the establishment of the proposed dataset. (**a**) Data were extracted from patients' bone marrow fluid samples obtained at the Department of Hematology at Zhejiang Hospital. The preparation and examination of the smears were carried out using two types of Olympus microscopes. (**b**) Images deemed to be of low quality underwent standardization through image processing software and systems to ensure their suitability for algorithmic processing. (**c**) Expert physicians at Zhejiang Hospital meticulously identified and manually annotated each image with the respective cell type. Ultimately, 40 types of bone marrow blood cells were classified within the dataset.
![image](img/fig2process.png)

## 3. Dataset

If you want to use our private dataset, please cite our article.

- You can view the downloadable dataset via figshare: [https://doi.org/10.6084/m9.figshare.25182506](https://doi.org/10.6084/m9.figshare.25182506)

- Download link of Bone Marrow Cell Fine-Grained Classification Dataset (BMC-FGCD) is available at [https://drive.google.com/file/d/1SWXL4V5iH--KBAex-H7Fbq1CMrqF6_Oh/view?usp=drive_link](https://drive.google.com/file/d/1SWXL4V5iH--KBAex-H7Fbq1CMrqF6_Oh/view?usp=drive_link).

## 4. Usage

**Below, we delineate the specific utility of this dataset in various application contexts:**

- Training of Deep Learning Models and Automatic Blood Cell Identification.
- Integrated Diagnosis with Clinical Data.
- Identification of Rare and Atypical Blood Cells.

## 5. Citation
```
Yifei Chen
```
