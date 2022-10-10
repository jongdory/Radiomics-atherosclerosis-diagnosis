# Radiomics feature extraction and classification for atherosclerosis diagnosis



## Description

 The purpose of this software is to extract radiomic features from the region of interest in carotid artery and to diagnose arteriosclerosis in patients through machine learning-based classifiers based on them. 
 Radiomic feature extraction was performed using **pyradiomics**, and the results are analyzed using four classifiers: logistic regression, soft vector machine, random forest, and xgboost.



## Usage

#### Dataset 

Dataset: Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis Challenge, MICCAI 2022 
Reference: https://zenodo.org/record/6481870#.Y0JjZXZBzdk



#### Requirements

---

```shell
pip install -r requirment.txt
```



#### Dataset conversion

---

Change the format of the dicom file to a nifti file. It also converts the control of the carotid blood vessels into a region.

```shell
python convert.py
```

The original data is stored in the /dataset directory. An MRI image converted to nifti format is created in the /data/VISTA directory, and an region of interest mask is created in the /ROI directory. In addition, /data/label.json creates a label for the data.



#### Radiomics feature extraction

---

Extracts radiomic features of slices with region of interest from the MRI

```shell
python feature_extraction.py
```

Extract features from images in the /data directory. The extracted radiomics features are stored in /feature directory.



#### Model training

---

Use four classifiers (logistic regression, soft vector machine, random forest, xgboost) to classify radiomics features. The machine learning models determines whether the carotid slice is arteriosclerosis or normal.

```shell
python train.py
```

Models that have been trained are saved in /train directory.



#### Run inference

---

```shell
python predict.py
```

Predict lesions for the patient's mri slice. Prediction results are stored in /results directory in json file format.



#### **Evaluation**

---

```shell
python eval.py
```

Evaluate the performance of the learned model.
