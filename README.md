# Data Science Challenge

This repository contains my answer to a Data Science challenge from a Fintech company in Brazil.
   
------------

### Challenge Description

Your goal is to predict the *default probability*. The "default" variable in the training dataset indicates whether the client defaulted or not. We are going to calculate the log loss based on the true value of the clients in the "puzzle_test_dataset" (not provided) and on your probabilities.
___________
### Input

* Training dataset with multiple features related to customers buying behavior.
    * Training dataset contains label **default** (1 if cliente has defaulted, 0 otherwise).
    
* Test dataset without labels


### Expected Output

* Customer ID and Prediction over *default*

--------------

# Solution
Check the Jupyter Notebook attached.
### Explanation
* This puzzle is well suited to be formulated as a supervised learning problem, due to the dataset characteristics (labeled data).

* We used ensemble models of decision trees which are good supervised learning classification algorithms for predicting probabilities over binary classes (logistic regression could also be used).

* The models used were:
    * XGBoost Trees (http://xgboost.readthedocs.io/en/latest/model.html)
    * Random Forests (https://en.wikipedia.org/wiki/Random_forest)

* The first model requires more complex parameter tuning. The second is simpler but it is still an ensemble model and had similar performance.

* XGBoost Trees was implemented using Scikit-Learn wrapper(http://xgboost.readthedocs.io/en/latest/python/python_api.html). 

* Random Forest was implemented using Scikit-Learn library itself (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

* We used cross-validation to evaluate the model and LogLoss as scoring function as per instructions.

* The parameters can be further optimized using GridSearch (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). We haven't used due to time limitation.

* The final predictions were based on the XGBoost Tree model only. The choice was only based on the fact that it had better LogLoss performance.  
* The feature selection was based on my own assumptions about the meaning of the variables. Some feature engineering was implemented over last\_payment and end\_last\_loan variables as per attached code.

* Code in Python (version 2.7)

* Python libraries used: Numpy, Pandas, Scikit-learn, XGboost, Matplotlib

* Instructions in the code (Jupyter-Notebook format)



