# Importing all the libraries
import numpy as npy 
import pandas as pdy
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pickle as pk
from sklearn.preprocessing import MinMaxScaler

#_____________________________________________________________________________
#
#				PRE PROCESSING THE DATA
#_____________________________________________________________________________
train=pdy.read_csv('trainSold.csv') 					# Reading the train.csv to train variable
train_cols_transform=['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition','SaleStatus']
train_data_frame_dummies=pdy.get_dummies(train,columns=train_cols_transform) # Converting categorical data into real data to predict house prices
y_train=train['SaleStatus']
x_train=train_data_frame_dummies.iloc[:,:-3]								

test=pdy.read_csv('testSold.csv') 					# Reading the test.csv to test variable
test_cols_transform=['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition']
test_data_frame_dummies=pdy.get_dummies(test, columns=test_cols_transform)  # Converting categorical data into real data to predict house prices
test_data_frame_dummies.to_csv("outTest.csv") 				# Its not required but it may useful thats why I am saving it in other final
#y_test=test['SaleStatus']
#x_test=test[:,:]
x_train.fillna(0, inplace=True) 					# Filling all NANs/NAs with 0 to avoid errors and to avoid noise data
#x_test.fillna(0, inplace=True) 					# Filling all NANs/NAs with 0 to avoid errors and to avoid noise data

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
x_train = scaling.transform(x_train)

#_____________________________________________________________________________
#
#				Final Models are 
#
#			1. Random Forest Classifier its almost giving ~92% at hyper parameter of n_estimators=15
#			2. Support Vector Machines Classifier its almost giving ~91% at hyper parameter of c=60
#_____________________________________________________________________________

#_____________________________________________________________________________
#
#				Random Forest Classifier
#_____________________________________________________________________________

from sklearn.ensemble import RandomForestClassifier
rfc_range=list(range(4,20))                      			  # Its hyper-parameter to explore different hyper-parameter values what will be the accuracy
rfc_accuracy_scores=[]
rfc_maxim=0								  # To get maximum value of accuracy when used different parameters 
rfc_index=0							  	  # To get maximum index of hyper-parameter 
for i in rfc_range:							  # For loop runnig to change different hyper parameter values
	rfc=RandomForestClassifier(n_estimators=i) 			  # Its a Classifier function with hyper-parameter which has change of hyper-parameter value
	accuracy_scores=cross_val_score(rfc,x_train,y_train,cv=9,scoring='accuracy').mean() # Calculating the accuracy of the RFC with change of hyper-parameter
	if(accuracy_scores>rfc_maxim): 					  # ................... 
		rfc_maxim=accuracy_scores 				  # To get maximum index of hyper-parameter 
		rfc_index=i 						  # ...................
	#normalizaion_scores=cross_val_score(rfc,x_train_normalizaion,y_train,cv=9,scoring='accuracy').mean()
	rfc_accuracy_scores.append(accuracy_scores.mean())
	#knn_normalizaion_scores.append(normalizaion_scores.mean())
#print(rfc_index)
plt.plot(rfc_range,rfc_accuracy_scores) 				   # Graph plot between rfc_range=(4,20) and rfc_accuracy_scores from an array
plt.xlabel('Value of n_estimatorsfor Random Forest')			   # Graph plot x label with n estimators
plt.ylabel('Cross Validation Accuracy')				           # Graph plot y label with Cross Validation Accuracy
plt.grid(True)								   # Graph plot with grid labes visible as true to visialize properly
plt.show()								   # It will show the graph with given parameters
rfc=RandomForestClassifier(n_estimators=rfc_index)   			   # Its a Classifier function with hyper-parameter which has high acuuracy parameter value
rfc.fit(x_train,y_train)                   			           # Fitting the data of x_train and y_train to evaluate the accuracy
x=open("finalModel1.pkl", "wb")						   # Opening the file to Writting bytes to finalmodel file which was fiited data of Random Forest Classifier
pk.dump(rfc, x)								   # Writting bytes from rfc to x(finalmodel file)
x.close()								   # Closing the x file to avoid garbage  data flow / any other error flow
print(cross_val_score(rfc,x_train,y_train,cv=9,scoring='accuracy').mean()) # Printing the accuracy of the RFC with higher value of hyper-parameter

# # #_____________________________________________________________________________
# # # #
# # # #				 Support Vector Machines Classifier
# # # #_____________________________________________________________________________
from sklearn import svm
svc_range=list(range(1,5))						  # Its hiper-parameter to explore different hyper-parameter values what will be the accuracy
svc_accuracy_scores=[]
svc_maxim=0								  # To get maximum value of accuracy when used different parameters 
svc_index=0								  # To get maximum index of hyper-parameter
for i in svc_range:							  # For loop runnig to change different hyper parameter values
	#for j in svc_range2:
	svc=svm.SVC(kernel='linear',C=i*20,gamma=0.02) 		  	  # Its a Classifier function with hyper-parameter which has change of hyper-parameter value
	accuracy_scores=cross_val_score(svc,x_train,y_train,cv=9,scoring='accuracy').mean()# Calculating the accuracy of the SVM with change of hyper-parameter
	if(accuracy_scores>svc_maxim):					  # ................... 
		svc_maxim=accuracy_scores 				  # To get maximum index of hyper-parameter 
		svc_index=(i-1)*20							# ................... 
	#normalizaion_scores=cross_val_score(svc,x_train_normalizaion,y_train,cv=9,scoring='accuracy').mean()
	svc_accuracy_scores.append(accuracy_scores.mean())
#print(svc_index)
plt.plot(svc_range,svc_accuracy_scores)  			 	   # Graph plot between rfc_range=(4,20) and rfc_accuracy_scores from an array
plt.xlabel('Value of C for Support Vector Machines') 			   # Graph plot x label with C svm
plt.ylabel('Cross Validation Accuracy') 			 	   # Graph plot y label with Cross Validation Accuracy
plt.grid(True)								   # Graph plot with grid labes visible as true to visialize properly
plt.show()								   # It will show the graph with given parameters
svc=svm.SVC(kernel='linear',C=svc_index,gamma=0.02)  			   # Its a Classifier function with hyper-parameter which has high acuuracy parameter value
svc.fit(x_train,y_train)						   # Fitting the data of x_train and y_train to evaluate the accuracy
x=open("finalModel2.pkl", "wb")						   # Opening the file to Writting bytes to finalmodel file which was fiited data of Random Forest Classifier
pk.dump(svc, x)								   # Writting bytes from rfc to x(finalmodel file)
x.close()								   # Closing the x file to avoid garbage  data flow / any other error flow
print(cross_val_score(svc,x_train,y_train,cv=9,scoring='accuracy').mean()) # Printing the accuracy of the SVM with higher value of hyper-parameter
