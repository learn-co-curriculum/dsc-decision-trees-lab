
# Building Trees using scikit-learn - Lab

## Introduction
Following the toy example we saw in the previous lesson, we shall now grow a decision tree for a more complex dataset. This lab covers all major areas of standard machine learning practice , from data acquisition to evaluation of results. We shall use scikit learn, pandas and graphviz libraries to conduct this analysis following the example of play tennis dataset.

## Objectives

You will be able to:

- Using pandas to prepare the data for the scikit-learn decision tree algorithm
- Train the classifier with a training dataset and evaluate performance using different measures
- Visualize the decision tree and interpret the visualization

## UCI Banknote Authentication Data Set

In this lab we shall work with a popular dataset for classification called the ["UCI Bank Note Authentication Dataset'](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). This Data were extracted from images that were taken from genuine and forged banknote-like specimens. The notes were first digitized, followed by a numerical transformation using DSP techniques. The final set of engineered features are all continuous in nature (visit the UCI link to learn about feature engineering in detail). 

We have following attributes in the dataset. 

1. __Variance__ of Wavelet Transformed image (continuous) 
2. __Skewness__ of Wavelet Transformed image (continuous) 
3. __Curtosis__ of Wavelet Transformed image (continuous) 
4. __Entropy__ of image (continuous) 
5. __Class__ (integer) - Target/Label 

## Step 1: Import necessary Libraries
- Import necessary libraries as we saw in previous lesson


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import tree 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd 
import numpy as np
%matplotlib inline
```

## Step 2: Import Data
- Read the file `"data_banknote_authentication.csv"` as a pandas dataframe. Note that there is no header information in this dataset.
- Assign column names 'Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class' to dataset in the given order.
- View the basic statistics and shape of dataset.
- Check for frequency of positive and negative examples in the target variable


```python
# Create Dataframe
dataset = pd.read_csv("data_banknote_authentication.csv", header=None) 
dataset.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
```


```python
# Describe the dataset
dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variance</th>
      <th>Skewness</th>
      <th>Curtosis</th>
      <th>Entropy</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1372.000000</td>
      <td>1372.000000</td>
      <td>1372.000000</td>
      <td>1372.000000</td>
      <td>1372.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.433735</td>
      <td>1.922353</td>
      <td>1.397627</td>
      <td>-1.191657</td>
      <td>0.444606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.842763</td>
      <td>5.869047</td>
      <td>4.310030</td>
      <td>2.101013</td>
      <td>0.497103</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.042100</td>
      <td>-13.773100</td>
      <td>-5.286100</td>
      <td>-8.548200</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.773000</td>
      <td>-1.708200</td>
      <td>-1.574975</td>
      <td>-2.413450</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.496180</td>
      <td>2.319650</td>
      <td>0.616630</td>
      <td>-0.586650</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.821475</td>
      <td>6.814625</td>
      <td>3.179250</td>
      <td>0.394810</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.824800</td>
      <td>12.951600</td>
      <td>17.927400</td>
      <td>2.449500</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Shape of dataset
dataset.shape
```




    (1372, 5)




```python
# Class frequency of target variable 
dataset.Class.value_counts()
```




    0    762
    1    610
    Name: Class, dtype: int64




```python
## Your Observations 

# We have 1372 examples , so have to be careful with splitting the dataset as we dont want to lose too much info from training set
# Classes in the output variable are ALMOST balanced , with negative class having a slightly higher frequency

```

## Step 3: Create Features and Labels, Training and Test Data

So now we need to create our feature set X and labels y. 
- Create X and y by selecting the appropriate columns from the dataset
- Create a 80/20 split on the dataset for training/testing. Use random_state=10 for reproducibility


```python
# Create features and labels
X = dataset.drop('Class', axis=1)  
y = dataset['Class']  
```


```python
# Perform an 80/20 split
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 10)  
```

## Step 4: Train the Classifier and Make PRedictions
- Create an instance of decision tree classifier with random_state=10 for reproducibility
- Fit the traiing data to the model 
- USe the trained model to make predictions with test data


```python
# Train a DT classifier
classifier = DecisionTreeClassifier(random_state=10)  
classifier.fit(X_train, y_train)  
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=10,
                splitter='best')




```python
# Make predictions for test data
y_pred = classifier.predict(X_test)  
```

## Step 5: Check Predictive Performance

We can now use different evaluation measures to check the predictive performance of the classifier. 
- Check the accuracy , AUC and create a confusion matrix 
- Interpret the results 


```python
# Calculate Accuracy 
acc = accuracy_score(y_test,y_pred) * 100
print("Accuracy is :{0}".format(acc))

# Check the AUC for predictions
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("\nAUC is :{0}".format(round(roc_auc,2)))

# Create and print a confusion matrix 
print('\nConfusion Matrix')
print('----------------')
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
```

    Accuracy is :97.81818181818181
    
    AUC is :0.98
    
    Confusion Matrix
    ----------------





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>149</td>
      <td>3</td>
      <td>152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>120</td>
      <td>123</td>
    </tr>
    <tr>
      <th>All</th>
      <td>152</td>
      <td>123</td>
      <td>275</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Your observations here 

# The classifier seems to perform well with a high accuracy 97% and AUC of 0.98
# We see a small number (3) for false positive and false negative errors. 
```

## Step 6: Visualize the tree 

- Use graphviz to visualize the tree 
- Interpret the results 


```python
# Visualize the tree trained from complete dataset
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```




![png](index_files/index_21_0.png)




```python
## Your observations here

# We see a deep tree with 8 layers
# Deep trees trying to classify every single example could lead to overfitting in most cases
```

## Bonus: Re-grow the Tree Using Entropy 

SO in the above example, we used all default settings for decision tree classifier. The default impurity crietrion in scikit learn is the Gini impurity. We can change it back to entropy by passing in `criterion='entropy'` argument to the classifier in the training phase. 
- Repeat the above tasks for training, evaluation and visualization using Entropy measure. (
- Compare and interpret the results 


```python
# Train a DT classifier
classifier2 = DecisionTreeClassifier(random_state=10, criterion='entropy')  
classifier2.fit(X_train, y_train)  
# Make predictions for test data
y_pred = classifier2.predict(X_test) 
# Calculate Accuracy 
acc = accuracy_score(y_test,y_pred) * 100
print("Accuracy is :{0}".format(acc))
# Check the AUC for predictions
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("\nAUC is :{0}".format(round(roc_auc,2)))
# Create and print a confusion matrix 
print('\nConfusion Matrix')
print('----------------')
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

# Visualize the tree trained from complete dataset
dot_data = StringIO()
export_graphviz(classifier2, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

    Accuracy is :99.63636363636364
    
    AUC is :1.0
    
    Confusion Matrix
    ----------------
    Predicted    0    1  All
    True                    
    0          151    1  152
    1            0  123  123
    All        151  124  275





![png](index_files/index_25_1.png)




```python
# Your observations here 

# With entropy we see a much lower accuracy and AUC than earlier
# Only one false positive value , leading to accuracy of 99%
# The tree shape changes considerably with change in impurity criteria 
```

## Level up - Optional 

- We discussed earlier that decision trees are very sensitive towards outliers. Try to identify and remove/fix any possible outliers in the dataset. 
- Check the distributions of the data. Is there any room for normalization/scaling of data ? Apply these techiques and see if it improves upon accuracy score. 

## Summary 

In this lesson, we looked at growing a decision tree for banknote authentication dataset which is composed of extracted continuous features from photographic data. We looked at different stages of the experiment including data acquisition, training, prediction and evaluation. We also looked at growing trees using entropy vs. gini impurity criteria. In following lessons, we shall look at some more such pre-train tuning techniques for ensuring an optimal classifier for learning and prediction.  
