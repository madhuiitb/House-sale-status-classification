
## Description: 
Implementation of residence sale status 


## Table of Contents: 
### 1. Data Pre-Processing
### 2. Choice of Classifiers Used
    1. Random Forest Classifier
    2. Support Vector Machines Classifier
    3. K-Nearest Neighbors Classifier
    4. Stochastic Gradient Descent Classifier
    5. Logistic Regression Classifier
### 3. Hyper Parameters and Graphs
I have used different hyper-parameters for each and every classifier out of the five methods that were
implemented, each model had it's own set of hyper-parameters. Change of one hyper-parameter for each
and every classifier, along with Cross Validation and Kfold(k=9) was done, and these chosen parameters
were updated to increase the cross validation accuracy. I have used cross validation technique which has
9 fold split data technique, which will divide the data into 9 parts and it will iterate 9 times each time
it will take one part as test data in each iteration till it completes all its 9 parts and for accuracy it will
take mean of all these 9 parts then it will finalize the accuracy of that particular hyper-parameter.
Please refer Figures 1, 2, 3, 4, 5 for this.

    1. Random Forest Classifier
      *  Hyper parameter used - n estimators
      * Best hyper parameter - n estimators = 9
      * ![RFC](/images/rfc3.jpeg)
      ![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)

    2. Support Vector Machines Classifier
    3. K-Nearest Neighbors Classifier
    4. Stochastic Gradient Descent Classifier
    5. Logistic Regression Classifier
### 4. Conclusion

## Installation:
#### * Python
#### * Numpy
#### * Pandas
#### * matplot

Usage: The next section is usage, in which you instruct other people on how to use your project after they’ve installed it. This would also be a good place to include screenshots of your project in action.

Contributing: Larger projects often have sections on contributing to their project, in which contribution instructions are outlined. Sometimes, this is a separate file. If you have specific contribution preferences, explain them so that other developers know how to best contribute to your work. To learn more about how to help others contribute, check out the guide for setting guidelines for repository contributors.

Credits: Include a section for credits in order to highlight and link to the authors of your project.

License: Finally, include a section for the license of your project. For more information on choosing a license, check out GitHub’s licensing guide!
