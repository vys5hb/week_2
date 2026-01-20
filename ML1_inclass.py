"""
ML1 In-Class
.py file
"""
# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3021/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.py#scrollTo=9723a7ee)

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3001/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.ipynb#scrollTo=9723a7ee)
# %%

# %%
# import packages
# from turtle import color
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# set the dimension on the images
import plotly.io as pio
pio.templates.default = "plotly_dark" # set dark theme

# %%
iris = data('iris')
iris.head()

# %%
# What mental models can we see from these data sets?
# There are 5 features in the iris dataset. 4 quantitative and 1 categorical.
# What data science questions can we ask?
# The species is the target variable, it's a categorical variable. We can use the 4 quantitative variables
# to predict the species (target variable)

# %%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this,
# we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent
# dataframes.

X = iris.drop(['Species'], axis=1).values  # features
y = iris['Species'].values  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for now

# %%
# Now, we use the scikitlearn k-NN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

# %%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

# %%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a
# classification model point these out?
# Just looking at the mean of the 3 species and how they differ between the 4 qualitative 
# variables. You can see that 3 groups can be created based off these variables,
# which will allow us to classify the species from the variables
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

# %%
patterns['Sepal.Width'].describe()

# %%
patterns['Petal.Length'].describe()

# %%
patterns['Petal.Width'].describe()

# %%
# scatter plot using plotly
fig = px.scatter_3d(iris, x='Sepal.Length', y='Sepal.Width', z='Petal.Length',
                 color='Species', title='Iris Sepal Dimensions')
fig.show()
# %%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example!
# We go over kNN models at length later in the course; you do not need to
# understand exactly what the model is doing quite yet.
# For now, ask yourself:

# 1. What is the purpose of data splitting?
# It allows us to separate our training data from our testing data to allow us to generalize to new data for predictions.
# If we don't split our data, our model will overfit the training data and might not generalize to new data as well.

# 2. What can we learn from data testing/validation?
# We can learn how well our model will generalize to new data for predictions
# 3. How do we know if a model is working?
# If we get good accuracy scores that are similar among our training, validation, and testing data.
# If our training score is much better than our validation/testing score, then our model is overfit.
# 4. How could we find the model error?
# You can look at the number of wrong predictions our model made versus the correct classification for each flower

# If you want, try changing the size of the test data
# or the number of n_neighbors and see what changes!
