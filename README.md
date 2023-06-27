# CDL-Outlier-Drift-Detection-Practicum

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
Practicum Group Members: Ishu Kalra, Kiran Jyothi Sheena, April Yang, Andrew Yaholkovsky, Yifei Wang
