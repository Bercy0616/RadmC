Integrated multiomics signatures to optimize the accurate diagnosis of lung cancer  
===================================================================================

[![]](./fig/flow1.png)  

## Abstract
Rapid identification of lung cancer from indeterminate pulmonary lesions (IPLs) and avoiding unnecessary invasive biopsies in patients with benign diseases are equally desirable outcomes that are often at odds with one another. A multi-omics deep learning model (RadmC) was proposed in our study to facilitate these outcomes. In this multicenter, prospective and observational study, we developed and validated a multiomics model (RadmC) to diagnose lung cancer. RadmC was constructed by deep learning algorithms on the basis of integration of CT-based radiomics features and multidimensional epigenomics features from plasma-derived 5-methylcytosines (5mC), 5-hydroxymethylcytosines (5hmC), and ratio between 5mC and 5hmC, showing superiority performance to single-omics prediction model with favorable sensitivity and specificity. Furthermore, its detection ability is consistently sufficient across different age, sex, lesion locations, lesion types, size, substage of adenocarcinoma, suggesting the superiority in various clinical scenarios. The potential of RadmC was further revealed in risk reclassification for indeterminate pulmonary lesions, thus, leading to the reduction of unnecessary invasive diagnostic procedures and delayed treatment for cancers. The multiomics deep learning model in this study offers a more effective, robust and noninvasive tool for lung cancer early diagnosis and management in clinical practice.  


## News
- 2024.11.22: Code, Models, and Sample data have been released.
- 2024.2.1: Code, Models, and Sample will be released. Stay tuned.

We will continue to optimize the code to improve readability.


## Performance



## Usage
You can download our sample data from [Baidu Disk](https://pan.baidu.com/s/1ZpjHTxwp17uLfldlhn3pqw) (code: miso) 

Otherwise, you can modify the hyperparameters for your dataset, and train or test like:  

```python  
python3 train.py
python3 test.py 
```

Recently, we will package our environment as **environment.yml** . It will be easier for you to run all the code.

## Acknowledgement
We would like to thank the MultiomIcs classifier for pulmOnary Nodules (MISSION) Collaborative Group for their supports and efforts.   
We are also grateful to the clinical research coordinators, Xiaomin Zhu and Chong Zhu, who helped to collect the plasma samples.   
Then, we thank all of the participants, without whom this research would not have been possible. 
