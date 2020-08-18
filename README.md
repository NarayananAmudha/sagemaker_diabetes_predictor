
# Udacity AWS SageMaker Capstone Project 

## Diabetes Prediction with XGBoost container of SageMaker

### Objective

The motive of this study is to design a model which can prognosticate the likelihood of diabetes in patients with maximum accuracy just by invoking Endpoint of Sagemaker with parameters they got from the Glucose Tolerance Test(GTT) result  

To build multiple supervised learning models of Scitkit learn from linear classifier, tree classifier, naive bayes, neighbors classifier type as base models to compare with XGBoost models build with pre-built container image from AWS Sagemaker for training and deployment.

### Dataset

The Pima Indian Diabetes Dataset with 768 patient’s data has been used in this study, provided by the UCI Machine Learning Repository (https://www.kaggle.com/uciml/pima-indians-diabetes-database). The dataset has been originally collected from the National Institute of Diabetes and Digestive and Kidney Diseases. The number of true cases are 268 (34.90%) and the number of false cases are 500 (65.10%) in this dataset.

### Libraries
Following Python libraries are used im Implementation
  
 - pandas 
 - numpy
 - matplotlib
 - scikit-learn
 - scipy	
 
### Steps
1. Clone the repository.
	```
		git clone https://github.com/NarayananAmudha/sagemaker_diabetes_predictor.git
	```
2. To pre-process raw data , save the processed data and to build all baseline models
   Open the `preprocess_baseline_models.ipynb` file.
	```
		jupyter notebook preprocess_baseline_models.ipynb
		
3.  To Split data and to run XGBoost Model of AWS Sagemaker container
    Run the 'sagemaker_xgboost_model.ipynb' in jupyter notebook with AWS SageMaker account.
	

### Implementation
1.	Analyzed the problem through visualizations and data exploration to have a better understanding of data and features that are appropriate for solving it by plotting correlation matrix.

2.	Pre-Processed data, as it plays an important role in improving performance of the model for small dataset such as by replacing null values with mean/median, and selected important features through feature selection. 
As XGBoost is able to perform pre-processing of replacing null values with median, class imbalance and feature selection by itself, implemented pre-processing and feature selection for building base models alone. Splitted the processed dataset into Train and Test dataset (80: 20 ratio) after scaling/normalizing the features with values between 0 to 1 

3.	Implemented Models Baseline Models with scikit library and XGboost in SageMaker by loading Training, Test and Validation data in S3 Bucket, followed by selecting Algorithm Container Registry Path, Configure Estimator for training by specifying Algorithm container, instance count, instance type, model output location to train model.  
Then tuned the model by HyperParameterTuner object by specifying ranges for hyperparameter such as max depth, learning rate, etc to find best performing model from the model search feature of Sagemaker pre-built container of XGBoost.

4.	Performed Evaluation of model as explained in Evaluation Metric section of this report by computing Accuracy Score, F1 score and ROC-AUC score.

5.	Finally Deployed model as SageMaker endpoint to run prediction on test data by specifying instance count, instance type and endpoint name in Sagemaker.


### Outcome and Future Improvement

Able to achieve Accuracy score of 75 % and ROC-AUC score of 83%  through XGBoost hyperparameter tuned model implemented with AWS Sagemaker pre-built container which is better than multiple baseline models with pre-processed dataset and I still believe, it is possible to improve further by varying parameter range of various parameters of XGBoost through further hyperparameter tuning.

One other consideration for this dataset is that it is collected specifically from a study using members of the Pima Indian tribe. It is entirely conceivable that there are genetic or lifestyle differences within this population that make any model built using this data not robust enough or valid for a larger more general population. With this in mind, any resulting model would need to be validated against populations of patients of either their respective population subgroup or against the general population before being widely deployed or used. In future  I also would like to try this model on the dataset of diabetes with 2000 patient’s data, taken from the hospital Frankfurt (https://www.kaggle.com/johndasilva/diabetes) also to compare the performance of model with benchmark Pima Indian Diabetes Dataset which also has same proportion of true cases (684 – 34.20% and false cases (1316 – 65.80%) . Both the dataset consists of same eight medical distinct measurement variables. 

Possible to enhance model as End to End web/mobile solution with Flask integrating via API to AWS SageMaker Predictor endpoint


