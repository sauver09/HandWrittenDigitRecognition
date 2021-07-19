import numpy as np
import pandas as pd 
import matplotlib . pyplot as plt
import scipy
from scipy import stats
from sklearn.metrics import accuracy_score

#This function is a helper function which return Euclidean or Manhatten distance based on the argument dis_type
def calculateDistance(X_test_i,X_train_i,dis_type):
    list_=scipy.spatial.distance.cdist(X_test_i,X_train_i,dis_type)
    return list_

################################################################################################################
# Input :
# • X_train: a two dimensional Numpy array of size n × d, where n is the number of training data points, and d the dimension of the feature vectors.
# • y_train: a Numpy vector of length n. y[i] is a categorical label corresponding to the data point X[i, :], y[i] ∈ {0,1,...,k − 1}. You can assume that the number of classes k is the maximum entry of y plus 1.
# • X_test: a two dimensional Numpy array of size m × d, where m is the number of test data points, and d the dimension of the feature vectors.
# • k: the number of nearest neighbors to use for classification

# Output:
# • yHat: a Numpy vector of length m for the predicted labels for the test data
# • Idxs: a Numpy array of size m × k for the indexes of the k nearest neighbor data points. Each row corresponds to a test data point.
################################################################################################################
def knn_classifier(X_train, y_train, X_test, k):

    k_nn=[]
    yHat=[]
    Idxs=[]
    nearestNeighbors=np.array(calculateDistance(X_test,X_train,"euclidean"))
    indexes=nearestNeighbors.argsort()
    Idxs=np.array(indexes[:,:k])
    yHat=np.array([stats.mode(i)[0][0] for i in np.array(y_train[indexes[:,:k]]) ])
    return yHat,Idxs
    

#Implemented this code to use manhantten distance with the same function signature
################################################################################################################
# Input :
# • X_train: a two dimensional Numpy array of size n × d, where n is the number of training data points, and d the dimension of the feature vectors.
# • y_train: a Numpy vector of length n. y[i] is a categorical label corresponding to the data point X[i, :], y[i] ∈ {0,1,...,k − 1}. You can assume that the number of classes k is the maximum entry of y plus 1.
# • X_test: a two dimensional Numpy array of size m × d, where m is the number of test data points, and d the dimension of the feature vectors.
# • k: the number of nearest neighbors to use for classification

# Output:
# • yHat: a Numpy vector of length m for the predicted labels for the test data
# • Idxs: a Numpy array of size m × k for the indexes of the k nearest neighbor data points. Each row corresponds to a test data point.
################################################################################################################

def knn_classifier_manhatten(X_train, y_train, X_test, k):
    
    k_nn=[]
    yHat=[]
    Idxs=[]
    nearestNeighbors=np.array(calculateDistance(X_test,X_train,"cityblock"))
    indexes=nearestNeighbors.argsort()
    Idxs=np.array(indexes[:,:k])
    yHat=np.array([stats.mode(i)[0][0] for i in np.array(y_train[indexes[:,:k]]) ])
    return yHat,Idxs
    



if __name__ == "__main__":

    ################################### Test Code #####################################################################################

    #Step1 : Loading data 

    data_train=pd.read_csv('mnist_train.csv')
    data_test=pd.read_csv('mnist_test.csv')

    X_train=data_train.drop('label',axis=1)
    y_train=np.array(data_train['label'])

    X_test=data_test.drop('label',axis=1)
    y_test=np.array(data_test['label'])

    X_train=np.array(X_train)/255
    X_test=np.array(X_test)/255


    #Step 2: Visualize some of the samples to better understand your dataset 

    fig = plt.figure()
    
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(np.reshape(X_train[i],(28, 28)))
        plt.title(y_train[i])
        plt.axis('off')

    plt.tight_layout() 
    plt.show()



    # Ans 2.2.1
    print("\n##################Ans 2.2.1####################\n")
    print("KNN Accuracy across different K\n")

    m_list=[0,1,2,4,8,16,32]
    accuracy_list=[]
    for m in m_list:    
        targetClass,K_indexes=knn_classifier(X_train,y_train,X_test,(2*m)+1)
        accuracy_list.append(accuracy_score(y_test,targetClass))
        print("Accuracy when k={} is {}".format(2*m+1,accuracy_list[-1]))





    m_list=np.array([0,1,2,4,8,16,32])
    k=[2*i+1 for i in m_list]

    plt.figure(figsize=(12,8))
    p1,=plt.plot(k,accuracy_list,color='blue',label='Accuracy',marker="*",markersize=20)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.legend(handles=[p1], title='KNN Accuracy across different K', bbox_to_anchor=(0.6, 1), loc='upper left')
    plt.title("Ans 2.2.1")
    plt.grid()

    print("\n######################################\n")



    # Ans 2.2.2
    print("\n##################Ans 2.2.2####################\n")
    print("KNN Accuracy across different number of samples\n")

    n_list=[100,200,400,600,800,1000]
    accuracy_list=[]
    for n in n_list:
        targetClass,K_indexes=knn_classifier(X_train[:n],y_train,X_test,3)
        accuracy_list.append(accuracy_score(y_test,targetClass))
        print("Accuracy when k={} and sampleSize={} is {}".format(3,n,accuracy_list[-1]))

    plt.figure(figsize=(12,8))
    p1,=plt.plot(n_list,accuracy_list,color='blue',label='Accuracy',marker="*",markersize=20)
    plt.xlabel("Number of Samples(n)")
    plt.ylabel("Accuracy")
    plt.legend(handles=[p1], title='KNN Accuracy across different number of samples', bbox_to_anchor=(0.2, 1), loc='upper left')
    plt.title("Ans 2.2.2")
    plt.grid()

    print("\n######################################\n")



    # Ans 2.2.3
    print("\n##################Ans 2.2.3####################\n")
    print("Manhatten vs Euclidean\n")

    targetClass_manhat,K_indexes=knn_classifier_manhatten(X_train,y_train,X_test,3)
    targetClass_eucli,K_indexes=knn_classifier(X_train,y_train,X_test,3)

    print("Manhatten Accuracy when k={} is {}".format(3,accuracy_score(y_test,targetClass_manhat)))
    print("Euclidean Accuracy when k={} is {}".format(3,accuracy_score(y_test,targetClass_eucli)))

    print("\n######################################\n")






    # Ans 2.2.4
    print("\n##################Ans 2.2.4####################\n")
    print(" 3 test samples that are wrongly classified.\n")

    targetClass,K_indexes=knn_classifier(X_train,y_train,X_test,5)
    ans=[ (i,(K_indexes[i])) for i,j in enumerate(zip(y_test,targetClass)) if j[0]!=j[1]]


    fig = plt.figure(figsize=(12,12))

    count=0

    for i in range(3):
        plt.subplot(5,6,count+1)
        title="True Value: {}".format(y_test[ans[i][0]])
        plt.title(title)
        count=count+1
        plt.imshow(np.reshape(X_test[ans[i][0]],(28, 28)))
        for j in range(5):
            plt.subplot(5,6,count+1)
            count=count+1
            title="Neighbor: {}".format(y_train[ans[i][1][j]])
            plt.title(title)
            plt.imshow(np.reshape(X_train[ans[i][1][j]],(28, 28)))
        
    plt.tight_layout() 
    plt.show()


    print("\n######################################\n")
