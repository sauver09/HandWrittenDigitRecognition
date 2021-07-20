<h1> Hand-Written Digits Recognition</h1>

This project provides working python code which can be used to detect Hand written digits.


## Steps to Run:
* Make sure you have following libraries installed
  * numpy
  * pandas
  * matplotlib
  * scipy
  * sklearn
* Make sure that you have following files in the directoy where you are trying to run this project
  <br>
  ├── main.py [Link](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/main.py) 
  <br>
  ├── mnist_train.csv [Link](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/mnist_train.csv) 
  <br>
  ├── mnist_test.csv [Link](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/mnist_test.csv) 

* Use the following command on terminal in the current directory where above files are present
```bash
python main.py
```

## Function:

- knn_classifier(X_train, y_train, X_test, k)
<pre>
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
</pre>


## Sample Ouput:

- #### Prediction on some sample test data ####
![alt text](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/screenshots/Screen%20Shot%202021-07-20%20at%201.36.05%20AM.png?raw=true)

- #### Accuracy across different number of Samples ####
![alt text](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/screenshots/Screen%20Shot%202021-07-20%20at%201.36.50%20AM.png?raw=true)

- #### Samples where model is wrong ####
![alt text](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/screenshots/Screen%20Shot%202021-07-20%20at%201.36.28%20AM.png?raw=true)

- #### Accuracy across different values of K in knn Classifier ####
![alt text](https://github.com/sauver09/Hand_Written_Digits_Recognition/blob/master/screenshots/Screen%20Shot%202021-07-20%20at%201.37.12%20AM.png?raw=true)

