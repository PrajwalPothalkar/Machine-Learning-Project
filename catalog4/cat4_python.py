#--------Machine Learning Project----------
#------------------------------------------

#--------KNN Algorithm---------------------
#------------------------------------------

#--------Importing Modules-----------------
import math
import operator
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
#-------------------------------------------

#Function for finding the Euclidean Distance for KNN Algorithm
def euclidean_distance_knn(part1,part2,length):
    my_distance=0
    for y in range(length):
        my_distance+=abs(part1[y]-part2[y])**2
    return math.sqrt(my_distance)

#Function to split the dataset into training and test data
def my_split_data(df,train_percentage=0.8):
    df['train']=np.random.rand(len(df))< train_percentage
    train=df[df.train == 1]
    test=df[df.train == 0]
    my_split_data={'train':train,'test':test}
    return my_split_data

#Opening the requires csv file

def load_split(file):
    data= pd.read_csv(file,index_col=False)
    
    data= data.drop(data.columns[0],axis=1)
    data= data[['u','g','r','i','z','extinction_u','extinction_g','extinction_r','extinction_i','extinction_z','nuv-u','nuv-g','nuv-r','nuv-i','nuv-z','u-g','u-r','u-i','u-z','g-r','g-i','g-z','r-i','r-z','i-z','class']]                              

    map_data=my_split_data(data)

    train=map_data['train']
    test=map_data['test']

    train=train.drop(train.columns[-1],axis=1)
    test=test.drop(test.columns[-1],axis=1)

    train=train.values
    test=test.values
    return data,train, test

#Finding the neighbours in KNN Algorithm
def kNN_neighbours(train,test,k):
    d=[]
    length=len(test)-1
    for x in range(len(train)):
        dist=euclidean_distance_knn(test,train[x],length)
        d.append((train[x],dist))
    d.sort(key=operator.itemgetter(1))
    nei=[]
    for x in range(k):
        nei.append(d[x][0])
    return nei

#Obtaining results for the neighbours in KNN Algorithm
def res(neighbours):
    kNN_classVotes={}
    for x in range(len(neighbours)):
        kNN_output=neighbours[x][-1]
        if kNN_output in kNN_classVotes:
            kNN_classVotes[kNN_output]+=1
        else:
            kNN_classVotes[kNN_output]=1
    kNN_sortedVotes=sorted(kNN_classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return kNN_sortedVotes[0][0]

#Function for finding the accuracy
def accuracy(confusion_matrix):
    tp=confusion_matrix[0][0]
    tn=confusion_matrix[1][1]
    fp=confusion_matrix[0][1]
    fn=confusion_matrix[1][0]
    return (tp+tn)/(tp+tn+fp+fn)

#Function for finding the precision
def precision(confusion_matrix):
    tp=confusion_matrix[0][0]
    fp=confusion_matrix[0][1]
    return tp/(tp+fp)

#Function for finding recall
def recall(confusion_matrix):
    tp=confusion_matrix[0][0]
    fn=confusion_matrix[1][0]
    return tp/(tp+fn)

#Function for finding specificity
def specificity(confusion_matrix):
    fp=confusion_matrix[0][1]
    tn=confusion_matrix[1][1]
    return tn/(tn+fp)

#Function for finding the sensitivity
def sensitivity(confusion_matrix):
    tp=confusion_matrix[0][0]
    fn=confusion_matrix[1][0]
    return tp/(tp+fn)

#Function for finding the f1_score
def f1_score(confusion_matrix):
    prec=precision(confusion_matrix)
    rec=recall(confusion_matrix)
    return 2*(prec*rec)/(prec+rec)

def knn(test_data, train_data,k):
    predictions=[]
    
    trueValue=[]

    for x in range(len(test_data)):
        nei=kNN_neighbours(train_data,test_data[x],k)
        result=res(nei)
        predictions.append(result)
        trueValue.append(test_data[x][-1])
    cm=confusion_matrix(trueValue,predictions)
    return (accuracy(cm),cm)

def cross_validation(df,s,k):
    accuracyy = 0
    for i in range(3):
        n_rows = int(df.shape[0]/3)
        test_index = random.sample(s,n_rows)
        test_data = df.iloc[test_index]
        train_data = df.drop(test_index)
        
        train_data= train_data.drop(train_data.columns[-1],axis=1)
        test_data= test_data.drop(test_data.columns[-1],axis=1)
        
        test_data=test_data.values
        train_data=train_data.values
        
        accuracyy+= knn(test_data,train_data,k)[0]
        
    accuracyy = accuracyy / 3
    print("Cross Validation accracy: ",accuracyy)



data,train,test= load_split("cat4.csv")
returning= knn(test,train,7)
cm = returning[1]
print("Normal Validation accuracy: ",returning[0])

print('f1-score', f1_score(cm))

print('precision ',precision(cm))

print('recall ',recall(cm))

print('sensitivity ',sensitivity(cm))

print('Specificity ',specificity(cm))

print()
s=data.index
cross_validation(data,list(s),7)


