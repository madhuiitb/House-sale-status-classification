
## Description: 
Implementation of residence sale status 


## Table of Contents: 
### 1. Data Pre-Processing
    * Identified the columns without real/integer data
    * Then made a list of non real data columns to converting categorical data into real data for prices prediction
    * After that have removed last 3 columns from the train data because we have to train without having sale status
    * Have normalized the with this MinMaxScaler function to feature range=(-1, 1)
    * Filled all NANs/NAs with 0 to avoid errors and to avoid noise data to get more accuracy
    * After above step, there was significant change in training time and accuracy of RFC and SVM
### 2. Choice of Classifiers Used
    * Used five classification models as mentioned below
    * In which Random Forest and Support Vector Machines giving best results(Best Cross Validation Score/Accuracy)
    * But took largest training time when executed the algorithm / program
    
    
        1. Random Forest Classifier
        2. Support Vector Machines Classifier
        3. K-Nearest Neighbors Classifier
        4. Stochastic Gradient Descent Classifier
        5. Logistic Regression Classifier
### 3. Hyper Parameters and Graphs
    * Used different hyper-parameters for each and every classifier
    * Each model had it's own set of hyper-parameters
    * Change of one hyper-parameter for each and every classifier, along with Cross Validation and Kfold(k=9)
    * I have used cross validation technique which has 9 fold split data technique
    * Which will divide the data into 9 parts and it will iterate 9 times
    * Each time it will take one part as test data in each iteration till it completes all its 9 parts
    * For accuracy it will take mean of all these 9 parts
    * And then it will finalize the accuracy of that particular hyper-parameter
    * Please refer below figures for full understanding of the problem

1. Random Forest Classifier
    *  Hyper parameter used - n estimators
    * Best hyper parameter - n estimators = 9
    ![Figure 1: Hyper-Parameter Tuning vs Cross validation accuracy of RFC](/images/rfc3.jpeg)
2. Support Vector Machines Classifier
    * Hyper parameter used - C(Slack Variable)
    * Best hyper parameter - C = 60
    ![Figure 2: Hyper-Parameter Tuning vs Cross validation accuracy of SVM](/images/svm3.jpeg)
3. K-Nearest Neighbors Classifier
    * Hyper parameter used - n neighbors
    * Best hyper parameter - n neighbors = 11
    ![Figure 3: Hyper-Parameter Tuning vs Cross validation accuracy of KNN](/images/knn3.jpeg)
4. Stochastic Gradient Descent Classifier
    * Hyper parameter used - max iter
    * Best hyper parameter - max iter = 835
    ![Figure 4: Hyper-Parameter Tuning vs Cross validation accuracy of SGDC](/images/sgdc_1.jpeg)
5. Logistic Regression Classifier
    * Hyper parameter used - C(Slack Variable)
    * Best hyper parameter - C = 9
    ![Figure 5: Hyper-Parameter Tuning vs Cross validation accuracy of LRC](/images/figure_1.jpeg)
    
## Installation:
#### * Python
#### * Numpy
#### * Pandas
#### * matplot

Usage: The next section is usage, in which you instruct other people on how to use your project after they’ve installed it. This would also be a good place to include screenshots of your project in action.

## Contributing: 
    I have written entire code, in the train.py and test.py files for machine learning, simple classification
    I have taken help of internet for usage of classifier libraries and functions usability

## Credits: 
    Have taken some insights to proceed with Data Preprocessing part from some of my friends
    http://scikit-learn.org/ - was very helpful for easy implementation and understanding of scikit-learn
    stats.stackexchange.com - for various scikit-learn's for more and easy classifiers implementation
    pandas documentation - for csv files reading and to store the data in csvs's and it was very useful
