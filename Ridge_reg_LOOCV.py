
# coding: utf-8

# In[198]:


import pandas as pd
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from numpy.linalg import norm


def RidgeReg(X_train,Y_train,lamda):
    I = np.identity(X_train.shape[0]-1);
    # adding column to a matrix
#    print(np.zeros((X_train.shape[0]-1,1)))
    I1 = np.append(I,np.zeros((X_train.shape[0]-1,1)),axis =1);
    # adding row to matrix
    I2 = np.append(I1,np.zeros((1,X_train.shape[0])),axis=0);
#    print(I2)
    C = np.add(np.dot(X_train, np.transpose(X_train)) , I2 * lamda); #  final C  mat
    C1 = np.linalg.inv(C);
    d = np.dot(X_train,Y_train);
    W = np.dot(inv(C),d);
    
    obj = np.subtract((np.dot(np.transpose(X_train),W)) ,Y_train)
    
   # obj1 = np.sum(obj);
    obj = np.square(obj);
#    print(obj.shape)
    obj1 = np.sum(obj)
#    obj1 = np.sum(obj) / X_train.shape[1];
#    obj1  = np.sqrt(obj1);
    
    obj1 = lamda * np.square(np.linalg.norm(W))+ obj1;
#    print(obj1)
    # LOOCV Validation
#    deno = np.subtract(np.identity(X_train.shape[1]) , np.dot(np.dot(np.transpose(X_train),C1),X_train)) #
    deno =  np.dot(np.dot(np.transpose(X_train),C1),X_train)
    deno = np.diag(deno);
    print(np.ones(X_train.shape[1]).shape)
    print(deno.shape)
    deno= np.subtract(np.ones(X_train.shape[1]),deno)
#    print(deno.shape)
    cvErrs = np.divide(np.subtract((np.dot(np.transpose(X_train),W)) ,Y_train)[:,0],deno)
#    print(cvErrs)
    return [W,W[-1],obj1,cvErrs]

    


    
    

training_file = "D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\traindata.csv";
test_file = "D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\testdata.csv";


train_x_df = pd.read_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\traindata.csv",header=None);
train_y_df = pd.read_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\trainlabels.csv",header=None);
valid_x_df = pd.read_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\valdata.csv",header=None);
valid_y_df = pd.read_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\vallabels.csv",header=None);



# for training data
X_train = train_x_df.values;
Y_train = train_y_df.values;
# for test data

#Y_test = train_x_df.values;

# for validation data
X_valid = valid_x_df.values;
Y_valid = valid_y_df.values;

# channging train values
X_train = np.delete(X_train,0,1);
Y_train = np.delete(Y_train,0,1);
X_train = np.transpose(X_train);
Y = np.ones((1,X_train.shape[1]))
X_train = np.append(X_train,Y,axis=0)

# changing validation values
X_valid = np.delete(X_valid,0,1);
Y_valid = np.delete(Y_valid,0,1);
X_valid = np.transpose(X_valid);
Y = np.ones((1,X_valid.shape[1]))
X_valid = np.append(X_valid,Y,axis=0)

#chaging test values


rmse_train=[];
rmse_valid = [];
rmse_train_loocv = [];
W_l = [];
b_l=[];
obj_l=[];
cvErrs_l=[];
lamda = np.array([0.01,0.1,1,10,100,1000]); # the weight for regularization term of Ridge regression
for l in lamda:
    print(l);
    [W,b,obj,cvErrs] = RidgeReg(X_train,Y_train,l);
    print(W.shape)
#    plt.figure(l)
#    plt.plot(np.arange(3001).reshape(3001,1),W)
    W_l.append(W);
    b_l.append(b); 
    obj_l.append(obj);
    cvErrs_l.append(cvErrs);
    
    cvErrs = np.square(cvErrs)
    rmse_train_loocv.append(np.sqrt(np.sum(cvErrs)/X_train.shape[1]));
    y_pred_train = np.dot(np.transpose(X_train),W)
    y_pred_valid = np.dot(np.transpose(X_valid),W)
    rmse_valid.append(np.sqrt(np.sum(np.square(y_pred_valid - Y_valid))/X_valid.shape[1]))
    rmse_train.append(np.sqrt(np.sum(np.square(y_pred_train - Y_train))/X_train.shape[1]))

print(rmse_train)
print(rmse_valid)
print(rmse_train_loocv)
# plotting curves    
plt.figure(1)
plt.plot(lamda,rmse_train)
plt.figure(2)
plt.plot(lamda,rmse_valid) 
plt.figure(3)
plt.plot(lamda,rmse_train_loocv)
#plt.figure(4)

plt.show()    




#X_train = np.asarray();



# In[216]:


cool_lamda =1;
print(X_train.shape,X_train.shape)
X_train_all = np.append(X_train,X_valid,axis =1)
print(X_train_all.shape)
Y_train_all = np.append(Y_train,Y_valid,axis=0)
print(X_train_all.shape,Y_train_all.shape)
for l in range(1):
    print(l);
    [W_final,b_final,obj_final,cvErrs_final] = RidgeReg(X_train_all,Y_train_all,0.789);
    cvErrs_final = np.square(cvErrs_final)
    rmse_loocv=np.sqrt(np.sum(cvErrs_final)/(X_train_all.shape[1]));
    y = np.dot(np.transpose(X_train_all),W_final)
    rmse_train_all= np.sqrt(np.sum(np.square(y - Y_train_all))/(X_train_all.shape[1]))
    print(rmse_loocv)
#Y_test = np.dot(np.transpose(X_test),W_final)

np
#print(rmse_loocv)
#print(rmse_train_all)





# In[217]:


test_x_df = pd.read_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\testdata.csv", header=None);
print(test_x_df.shape)
X_test = test_x_df.values;
X_test = np.delete(X_test,0,1);
print(X_test.shape)
X_test = np.transpose(X_test);
Y = np.ones((1,X_test.shape[1]))
X_test = np.append(X_test,Y,axis=0)
print(X_test.shape)
Y_test = np.dot(np.transpose(X_test),W_final)
print(Y_test.shape,X_test.shape,W_final.shape)
df = pd.DataFrame(Y_test)
df.to_csv("D:\\Courses\\1st_semester\\ML\\Assignment2\\data\\predTestLabels.csv")


# In[219]:


plt.plot(W_final)
plt.show()
print(W_final)

