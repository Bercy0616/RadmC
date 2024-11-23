Integrated multiomics signatures to optimize the accurate diagnosis of lung cancer  
===================================================================================

<div align="center">
  <img src="https://github.com/Bercy0616/RadmC/blob/main/fig/flow1.png">
</div>



<div align="center">
  <img src="https://github.com/Bercy0616/RadmC/blob/main/fig/flow2.png">
</div>    


## Abstract
Diagnosing lung cancer from indeterminate pulmonary nodules (IPLs) remains challenging. In this multi-institutional study involving 2032 participants with IPLs, we integrate the clinical, radiomic with circulating cell-free DNA fragmentomic features in 5-methylcytosine (5mC)-enriched regions to establish a multiomics model (clinic-RadmC) for predicting the malignancy risk of IPLs. The clinic-RadmC yields an area-under-the-curve (AUC) of 0.923 on the external test set, outperforming the single-omics models, and models that only combine clinical features with radiomic, or fragmentomic features in 5mC-enriched regions (p<0.050 for all). The superiority of the clinic-RadmC maintains well even after adjusting for clinic-radiological variables. Furthermore, the clinic-RadmC-guided strategy could reduce the unnecessary invasive procedures for benign IPLs by 10.9%~35%, and avoid the delayed treatment for lung cancer by 3.1%~38.8%. In summary, our study indicates that the clinic-RadmC provides a more effective and noninvasive tool for optimizing lung cancer diagnoses, thus facilitating the precision interventions.


## News
- 2024.11.22: Code, Models, and Sample data have been released.
- 2024.2.1: Code, Models, and Sample will be released. Stay tuned.

We will continue to optimize the code to improve readability.


## Performance

<div align="center">
  <img src="https://github.com/Bercy0616/RadmC/blob/main/fig/result1.png">
</div>


<div align="center">
  <img src="https://github.com/Bercy0616/RadmC/blob/main/fig/result2.png">
</div>    

<div align="center">
  <img src="https://github.com/Bercy0616/RadmC/blob/main/fig/result3.png">
</div>    

## Usage
You can download our sample data from [Baidu Disk](https://pan.baidu.com/s/1ZpjHTxwp17uLfldlhn3pqw) (code: miso) 

You can download our weights from [Baidu Disk](https://pan.baidu.com/s/1jxNr77jJ6W_xfoTxwAQhqw) (code: miso) 

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
