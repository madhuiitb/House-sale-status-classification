# Importing libraries
import pickle as pk 
import numpy as npy 
import pandas as pdy 
from sklearn.metrics import accuracy_score

#_____________________________________________________________________________
#
#				PRE PROCESSING THE DATA
#_____________________________________________________________________________
train=pdy.read_csv('trainSold.csv') # Reading the train.csv to train variable
train_cols_transform=['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition','SaleStatus']
train_data_frame_dummies=pdy.get_dummies(train,columns=train_cols_transform) # Converting categorical data into real data to predict house prices
train_data_frame_dummies.to_csv("outTrain.csv")# Its not required but it may useful thats why I am saving it in other final
y_train=train['SaleStatus']                   
x_train=train_data_frame_dummies.iloc[:,:-3]   # Taking all the colums into x_train data to train the data except last 3 colums because these 3 values are salestatus

test=pdy.read_csv('testSold.csv') # Reading the test.csv to test variable
test_cols_transform=['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition']
test_data_frame_dummies=pdy.get_dummies(test, columns=test_cols_transform) # Converting categorical data into real data to predict house prices
test_data_frame_dummies.to_csv("outTest.csv")  # Its not required but it may useful thats why I am saving it in other final
x_test=test_data_frame_dummies.iloc[:,:]       # Taking all the colums into x_test data to test
x_test.fillna(0, inplace=True)				   # Filling all NANs/NAs with 0 to avoid errors and to avoid noise data
x_train.fillna(0, inplace=True)				   # Filling all NANs/NAs with 0 to avoid errors and to avoid noise data


#_____________________________________________________________________________
#
#Adjusting the columns of the x_test data and x_train data				
#_____________________________________________________________________________
for i in list(set(x_train.columns)-set(x_test.columns)): # Converting x_train.colums as set data and removing x_testcolums as set are as list of each iteration
	x_test[i]=0											 # Adding that each i in x_test and making that colums as 0
for i in list(set(x_test.columns)-set(x_train.columns)): # Converting x_testcolums as set and removing x_train.colums as set dataare as list of each iteration
	x_test[i]=x_test.drop([i],axis=1)					 # Droping that each i in x_test to get exact colums as x_train have

actual=pdy.read_csv('gt.csv')							 # Ground thruth values gt.csv reading to actual file to predict accuracy and to calculate the accuracy			
actual_final=actual['SaleStatus']						 # actual_final will have only salestatus colums to check accuracy to match with that of prediction value



#_____________________________________________________________________________
# #
# #					Final Models are 
#
#			1. Random Forest Classifier its almost giving ~92% at hyper parameter of n_estimators=15
#			2. Support Vector Machines Classifier its almost giving ~91% at hyper parameter of c=60
#			
# #_____________________________________________________________________________

#_____________________________________________________________________________
# #
# #				Random Forest Classifier
# #_____________________________________________________________________________
with open('finalModel1.pkl', 'rb') as model3:   # Reading bytes from finalmodel file which was written by Support Vector Machines Classifier
    res3 = pk.load(model3) 						# Loading bytes to res3 to predict the data using this res3
model3.close()                                  #closing the modle3 file to avoid garbage collection/data flow
pred3=res3.predict(x_test)                      #Its predicting the data between x_test and res3 and storing it in pred3 to calculate accuracy of the RFC
npy.savetxt("out1.csv", pred3, fmt="%s", delimiter=",")# Saving predicted data in out1.csv with salesatus
acc=accuracy_score(pred3,actual_final)          # Finding accuracy between predicted data and actual_ground truth data
print(acc)                                      # printing the accuracy of the Random Forest Classifier

# #_____________________________________________________________________________
# #
# #				Support Vector Machines Classifier
# #_____________________________________________________________________________
with open('finalModel2.pkl', 'rb') as model5:   # Reading bytes from finalmodel file which was written by Random Forest Classifier
    res5 = pk.load(model5) 						# Loading bytes to res5 to predict the data using this res5
model5.close()                                  #closing the modle5 file to avoid garbage collection/data flow
pred5=res5.predict(x_test)                      #Its predicting the data between x_test and res5 and storing it in pred5 to calculate accuracy of the SVM
npy.savetxt("out2.csv", pred5, fmt="%s", delimiter=",")# Saving predicted data in out1.csv with salesatus
acc=accuracy_score(pred5,actual_final)			# Finding accuracy between predicted data and actual_ground truth data
print(acc)										# printing the accuracy of the Random Forest Classifier
