# Airbnb New User Bookings
Where will a new guest book their first travel experience? You can find the link [here](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings).

## Business problem
The goal is to help Airbnb predict which country a new user will make his or her first booking.
By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

This type of classification problem is applicable for many internet/service companies that are looking to drive new users to their website and improve personalization of content to users.

## Data Structure
Data for this project mainly includes a users and sessions dataset.
* **Users** - demographic information about the user along with information about when and how the individual first signed up to be an Airbnb member.
* **Sessions** - web sessions logs for users. Fields include user actions, session time, and device type for the session

## Documentation

* `eda.ipynb` Jupyter Notebook that contains exploratory data analysis [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/eda.ipynb)
* `Airbnb 2.26.18 - Data Cleansing & Feature Creation.ipynb` Data Cleansing & Feature Creation [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Airbnb%202.26.18%20-%20Data%20Cleansing%20%26%20Feature%20Creation.ipynb)
* `Logistic.ipynb` Jupyter Notebook that contains logistic regression model building. [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Logistic.ipynb)
* `Model-SVM.ipynb` Jupyter Notebook that contains SVM model building. [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Model-SVM.ipynb)
* `Model-GBM.ipynb` Jupyter Notebook that contains boosted tree model building. [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Model-GBM.ipynb)
* `Model-RF.ipynb` Jupyter Notebook that contains random forest model building. [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Model-RF.ipynb)
* `Model-NN.ipynb` Jupyter Notebook that contains neural network model building. [[nbviewer]](http://nbviewer.jupyter.org/github/bkennedy04/msia420_airbnb_prediction/blob/master/src/Model-NN.ipynb)



## Directory Structure

```
project
|   README.md
|   .gitignore
|
|__ data: contains the data downloaded from kaggle
|
|__ script: Source code used for this project
|
|__ scr: notebook related files
|
|__ submission: a folder contains the submission file generated from python script
```

