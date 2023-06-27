# CDL-Outlier-Drift-Detection-Practicum

## Team Members
Ishu Kalra, Kiran Jyothi Sheena, Chunxue(April) Yang, Andrew Yaholkovsky, Yifei Wang

## Objective

The objective of this project was to develop efficient modules for outlier and drift detection in streaming data. Three distinct modules were created to accomplish this objective.

1. The first module focuses on Concept Drift Detection, which arises when the relationship between dependent and independent variables undergoes changes over time. Multiple algorithms were developed within this module, utilizing both sliding and fixed windows. Importantly, these algorithms operate independently of the true label associated with the incoming data.

2. The second module addresses Data Drift Detection, which occurs when the distribution of the dependent variable changes over time. To identify data drift, statistical tests such as KL divergence, JS convergence, or PSI are employed. A module was developed to implement an algorithm that detects data drift using a sliding window approach. Additionally, an algorithm was devised to optimize the window size for better detection accuracy.

3. The third module focuses on Outlier Detection, recognizing that streaming data often contains features that are susceptible to outliers. To address this, a module was designed to detect outliers in the data. This module utilizes the algorithms, iForestASD and Half Space Tree, which have proven to be effective in outlier detection scenarios.

Step into any of the following folders for more details:
 - [Concept Drift Detection](/Concept%20Drift%20Detection)
 - [Data Drift Detection](/Data%20Drift%20Detection)
 - [Outlier Detection](/Outlier%20Detection)

<br>
<br>

## Project Structure
<pre>
 ├── Concept Drift Detection
│   ├── Readme.md
│   ├── Semi-supervised_Concept_Drift.ipynb
│   ├── drift_injection
│   │   ├── CreateGradualDriftDatasets.py
│   │   ├── CreateSuddenDriftDatasets.py
│   │   ├── Gradual30high200.csv
│   │   ├── InjectGradualDrift.py
│   │   ├── InjectSuddenDrift.py
│   │   ├── MotionSenseData.csv
│   │   ├── PythonNotebookReference
│   │   │   ├── ArtificialDriftInjection.ipynb
│   │   │   ├── ExperimentingWithMultiflow.ipynb
│   │   │   └── TestingDriftOnDatasets.ipynb
│   │   └── Sudden30high200.csv
│   ├── images
│   │   ├── DriftInjectionExample.png
│   │   ├── GradualDriftInjection.png
│   │   └── InjectedDriftbyExerciseType.png
│   ├── reference
│   │   └── Concept_Drift_no_labels.pdf
│   ├── src
│   │   ├── Dynamic_window.py
│   │   ├── Ensemble_reference_window.py
│   │   ├── Fixed_reference_window.py
│   │   ├── Moving_reference_window.py
│   │   └── __init__.py
│   └── test_dataset
│       ├── NoDrift.csv
│       ├── sub_1_combined_data.csv
│       ├── sub_2_combined_data.csv
│       └── sudden30high200.csv
├── Data Drift Detection
│   ├── Examples
│   │   ├── Example_1.ipynb
│   │   ├── Example_2.ipynb
│   │   └── Window_Variation_Results.ipynb
│   ├── README.md
│   ├── config
│   │   └── default-config.yaml
│   ├── datasets
│   │   ├── N=10000_Subject_1_10_sudden_drift.csv
│   │   ├── N=1000_Subject_2_11_distributed_drift_2000_1000_50.csv
│   │   ├── N=100_Subject_2_11_distributed_drift_2000_1000_50.csv
│   │   ├── N=16000_Subject_1_10_sudden_drift.csv
│   │   ├── N=20000_Subject_1_10_sudden_drift.csv
│   │   ├── N=2000_Subject_2_11_distributed_drift_2000_1000_50.csv
│   │   ├── N=200_Subject_2_11_distributed_drift_2000_1000_50.csv
│   │   ├── N=4000_Subject_1_10_sudden_drift.csv
│   │   ├── N=500_Subject_2_11_distributed_drift_2000_1000_50.csv
│   │   ├── N=8000_Subject_1_10_sudden_drift.csv
│   │   ├── Subject_10_random.csv
│   │   ├── Subject_1_random.csv
│   │   ├── Subject_2_11_distributed_lambda=2000_size=1000_50.csv
│   │   ├── Subject_2_11_distributed_lambda=2000_size=1000_50_indicators.csv
│   │   └── sub_2.csv
│   ├── main.py
│   └── src
│       ├── __init__.py
│       ├── data_drift_detection.py
│       └── window_optimization.py
├── Outlier Detection
│   ├── README.md
│   ├── datasets
│   │   ├── 202301-divvy-tripdata.csv
│   │   ├── 202302-divvy-tripdata.csv
│   │   ├── 202303-divvy-tripdata.csv
│   │   ├── motion_sense_combined_data
│   │   │   ├── sub_10_combined_data.csv
│   │   │   ├── sub_11_combined_data.csv
│   │   │   ├── sub_12_combined_data.csv
│   │   │   ├── sub_13_combined_data.csv
│   │   │   ├── sub_14_combined_data.csv
│   │   │   ├── sub_15_combined_data.csv
│   │   │   ├── sub_16_combined_data.csv
│   │   │   ├── sub_17_combined_data.csv
│   │   │   ├── sub_18_combined_data.csv
│   │   │   ├── sub_19_combined_data.csv
│   │   │   ├── sub_1_combined_data.csv
│   │   │   ├── sub_20_combined_data.csv
│   │   │   ├── sub_21_combined_data.csv
│   │   │   ├── sub_22_combined_data.csv
│   │   │   ├── sub_23_combined_data.csv
│   │   │   ├── sub_24_combined_data.csv
│   │   │   ├── sub_2_combined_data.csv
│   │   │   ├── sub_3_combined_data.csv
│   │   │   ├── sub_4_combined_data.csv
│   │   │   ├── sub_5_combined_data.csv
│   │   │   ├── sub_6_combined_data.csv
│   │   │   ├── sub_7_combined_data.csv
│   │   │   ├── sub_8_combined_data.csv
│   │   │   └── sub_9_combined_data.csv
│   │   ├── revised_sub20_cleandata_forStream.csv
│   │   ├── revised_sub20_data_2ndV.csv
│   │   └── revised_sub20_data_Ishu_modification.csv
│   ├── notebooks
│   │   ├── Dimensionality_Reduction.ipynb
│   │   ├── DivvyBike-Outlier-Detection.ipynb
│   │   ├── ExploringDivvyBikeDataset.ipynb
│   │   ├── Hierarchical_clustering.ipynb
│   │   ├── MotionSenseDataset-HSTree&iForestASD.ipynb
│   │   └── MotionSense_LOF_confusion_matrix_incorporated.ipynb
│   └── src
│       ├── DivvyBike-Outlier-Detection.py
│       ├── Hierarchical_clustering.py
│       ├── MotionSense_HSTree.py
│       ├── MotionSense_LOF_confusion_matrix_incorporated.py
│       └── source
│           ├── functions.py
│           └── iforestasd_scikitmultiflow.py
└── README.md

</pre>
