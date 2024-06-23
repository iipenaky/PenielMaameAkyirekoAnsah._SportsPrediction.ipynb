#!/usr/bin/env python
# coding: utf-8

# ## Question1: Data preparation & feature extraction process

# Imports for extraction

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


training = pd.read_csv("C:/Users/Ansah/OneDrive - Ashesi University/Documents/Ashesi/Sem4/AI/Lab02_peniel.ansah/male_players (legacy).csv")


# In[3]:


training.head()


# In[4]:


training.describe()


# In[5]:


#Removing all ids, all fifa versions and urls as they are irrelevant to the overall
for i in training.columns.tolist():
    if '_url' in i:
        training.drop(i, axis = 1, inplace= True)
    if '_id' in i:
        training.drop(i, axis = 1, inplace = True)
    if 'fifa' in i:
        training.drop(i, axis = 1, inplace = True)


# In[6]:


#Remove columns withe 30% or more null values too.
L= []
L_less = []
for i in training.columns:
    if((training[i].isnull().sum())<(0.3*(training.shape[0]))):
        L.append(i)
    else:
        L_less.append(i)
training = training[L]


# In[7]:


# More columns to remove:
#Take out positions from the dataframe because judging the players by how good they perform at positions they perform well at can lead to biased desicions.
# 'short_name' : There is a long name which can act as an identifier
# 'club_position': The player's position in the club is less informative than performance metrics.
# 'club_jersey_number': Jersey numbers are not indicative of player skill or performance.
# 'club_joined': The date the player joined the club is less relevant compared to their performance metrics.
# 'club_contract_valid_until': Contract length is less directly tied to skill.
# 'real_face': Whether the player's face is realistic in the game is irrelevant to performance.
# Goalkeeping specific attributes are dropped since we are focusing on general player attributes:
# 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'

columns_to_drop = ['ls',
 'st',
 'rs',
 'lw',
 'lf',
 'cf',
 'rf',
 'rw',
 'lam',
 'cam',
 'ram',
 'lm',
 'lcm',
 'cm',
 'rcm',
 'rm',
 'lwb',
 'ldm',
 'cdm',
 'rdm',
 'rwb',
 'lb',
 'lcb',
 'cb',
 'rcb',
 'rb',
 'gk',
 'long_name',
    'short_name','club_position', 'club_jersey_number', 'club_joined_date',
 'club_contract_valid_until_year', 'real_face', 
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
    'goalkeeping_positioning', 'goalkeeping_reflexes', 'club_name', 'league_name', 'dob', 'player_positions', 'nationality_name']
training.drop(columns = columns_to_drop, axis = 1, inplace = True)


# In[8]:


# separate the categorical columns from the numeric columns
categorical_players = training.select_dtypes(exclude='number')
numerical_players = training.select_dtypes(include='number')


# In[9]:


training.select_dtypes(include = 'number')


# In[10]:


categorical_players.info()


# No need to impute the categories since they are all not null

# In[11]:


sc = SimpleImputer(strategy = 'mean')
scaled = sc.fit_transform(numerical_players)
numerical_players = pd.DataFrame(scaled, columns = numerical_players.columns)


# In[12]:


numerical_players.isnull().sum()


# Use a boxplot to know the realtionahip between Categorical and Quantitative variables

# In[13]:


for col in categorical_players.columns.tolist():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y=numerical_players['overall'], data = training)
    plt.title(f'Box plot of numerical vs {col}')
    plt.show()


# No categories would beused since they all have less importance, and the one that changed a bit was the body weight. 
# I wouldn't use the body weight category in my model because, out of the ten categories available, only the unique body type has a significant impact on the model's training. The other body weight categories don't provide useful information and might even add noise to the data, making the model less accurate and harder to interpret. By excluding this category, I can simplify the model and focus on the features that truly matter, leading to better performance and easier analysis.

# In[14]:


new_players = numerical_players


# In[15]:


new_players


# In[16]:


# y = new_players.loc[:,'overall']

# #From the dataset, from 123813 is the same as the testing dataset.
# y_temp = y[:123813]

# X = new_players.drop('overall', axis = 1)

# #From the dataset, from 123813 is the same as the testing dataset.
# X_temp = X.iloc[:123813]


# In[17]:


# from sklearn.preprocessing import StandardScaler
# #This scales the independent variables
# scale = StandardScaler()
# scaled = scale.fit_transform(X_temp)

# #This puts the indpendent variables in a dataframe
# X_temp = pd.DataFrame(scaled, columns = X_temp.columns)


# ## Question2: Feature Engineering

# In[18]:


# # Combine x_temp and y_temp into a single DataFrame
# combined_df = X_temp.copy()
# combined_df['overall'] = y_temp

# Calculate the correlation matrix
corr_matrix = new_players.corr()
correlations = corr_matrix["overall"].sort_values(ascending=False)

# Plot the sorted correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
plt.title('Correlations with Overall')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()


# In[19]:


numeric_chosen = correlations[abs(correlations.values) >= 0.5]


# In[20]:


# Get the columns of numeric features that correlates well with overall
chosen_columns = []
for variable, correlation in numeric_chosen.items():
    if variable != 'overall':
        chosen_columns.append(variable)

# Create a new DataFrame with these chosen columns
X = new_players[chosen_columns]
y = new_players['overall']


# In[21]:


X


# In[22]:


y


# In[23]:


from sklearn.preprocessing import StandardScaler
#This scales the independent variables
scale = StandardScaler()
scaled = scale.fit_transform(X)

#This puts the independent variables in a data frame
X = pd.DataFrame(scaled, columns = X.columns)


# #From the dataset, from 123813 is the same as the testing dataset.
y_temp = y.iloc[:123813]


# #From the dataset, from 123813 is the same as the testing dataset.
X_temp = X.iloc[:123813]


# In[24]:


y_temp


# In[25]:


#this is to split the training and testing data
from sklearn.model_selection import train_test_split
#After splitting we must randomise the data.  The size there determis the size of the test set. That is 20%. We randomize the data to avoid being biased.
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_temp,y_temp,test_size=0.2,random_state=42)


# ## Question3:Training Models

# In[26]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
kf = KFold(n_splits=3)


# In[27]:


# RandomForest Regressor
rf = RandomForestRegressor( random_state=42)
rf.fit(Xtrain, Ytrain)
# Predict the test part
y_pred_rf = rf.predict(Xtest)
rf_scores = cross_val_score(rf, Xtrain, Ytrain, cv=3, scoring='neg_mean_squared_error')
print(f"Random Forest - Cross-Validation MSE: {-rf_scores.mean()}")


# In[28]:


gb = GradientBoostingRegressor(random_state=42)
gb.fit(Xtrain, Ytrain)

# Predict the test set
y_pred_gb = gb.predict(Xtest)
gb_tscores = cross_val_score(gb, Xtrain, Ytrain, cv=3, scoring='neg_mean_squared_error')
print(f"GradientBoosting - Cross-Validation MSE: {-gb_tscores.mean()}")


# In[29]:


model = XGBRegressor(random_state = 42)
scores = cross_val_score(model, Xtrain, Ytrain, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
model.fit(Xtrain,Ytrain)
y_pred = model.predict(Xtest)
print(f"XGBoosting - Cross-Validation MSE: {-scores.mean()}")


# ## Question4: Evaluation

# In[30]:


print(f"""Random Forest - Test Set
Mean Absolute Error = {mean_absolute_error(Ytest,y_pred_rf )},
Mean Squared Error = {mean_squared_error(Ytest,y_pred_rf)}
Root Mean Squared Error = {np.sqrt(mean_absolute_error(Ytest,y_pred_rf))},
R2 score = {r2_score(Ytest,y_pred_rf)}"""
)

print(f"""GradientBoosting - Test Set:
Mean Absolute Error = {mean_absolute_error(Ytest,y_pred_gb )},
Mean Squared Error = {mean_squared_error(y_pred_gb, Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred_gb, Ytest))},
R2 score = {r2_score(y_pred_gb, Ytest)}"""
)

print(f"""XGBoost - Test Set
Mean Absolute Error = {mean_absolute_error(Ytest,y_pred )},
        Mean Squared Error = {mean_squared_error(Ytest,y_pred)}
        Root Mean Squared Error = {np.sqrt(mean_absolute_error(Ytest,y_pred))},
        R2 score = {r2_score(Ytest,y_pred)}"""
)


# Based on the evaluation metrics, the Random Forest model has the lowest MAE, MSE, and RMSE on the test set, indicating that it has the best predictive accuracy among the three models. Additionally, it has the highest R² score, suggesting that it explains the most variance in the target variable

# #### Hyperparameter Tuning

# In[31]:


from sklearn.model_selection import RandomizedSearchCV


# In[32]:


param_distributions = {
    'n_estimators': [50, 60],        # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],      # Number of features to consider at every split
    'max_depth': [None, 10, 20],       # Maximum number of levels in the tree
    'min_samples_split': [2, 5],               # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2],                 # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                     # Method of selecting samples for training each tree
}


# In[33]:


random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=10, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
#Train the model again
random_search.fit(Xtrain, Ytrain)

print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {-random_search.best_score_}")
print (f"Best estimator: {random_search.best_estimator_}")

# Evaluate on test set again
best_model = random_search.best_estimator_

#predictions
rf_pred = best_model.predict(Xtest)


# In[34]:


rf_test_mae = mean_absolute_error(Ytest, rf_pred)
rf_test_rmse = np.sqrt(mean_squared_error(Ytest, rf_pred))
rf_test_rscore = r2_score(Ytest, rf_pred)

print("RandomForest Test MAE:", rf_test_mae)
print("RandomForest Test RMSE:", rf_test_rmse)
print("RandomForest Test R2 score:", rf_test_rscore)


# ## Question5: Test With New Dataset

# In[35]:


players_22 = pd.read_csv("C:/Users/Ansah/OneDrive - Ashesi University/Documents/Ashesi/Sem4/AI/Lab02_peniel.ansah/players_22-1.csv")


# In[36]:


#Get the features used to train the model.
test_players = players_22[X_temp.columns.tolist()]

#Fill the null values in the numerical columns with the mean values
sc = SimpleImputer(strategy = 'mean')
imputed_players = sc.fit_transform(test_players)
test_players = pd.DataFrame(imputed_players, columns = test_players.columns)

#Scale the features.
scale = StandardScaler()
scaled = scale.fit_transform(test_players)
test_players = pd.DataFrame(scaled, columns = test_players.columns)


# In[37]:


#Get the target
target = players_22.loc[:,'overall']


# In[38]:


# Make predictions using the test dataset
test_predictions = best_model.predict(test_players)


# In[39]:


# Evaluate predictions using MAE, RMSE, and R² score
rf_mae_new = mean_absolute_error(target,test_predictions)
rf_rmse_new = np.sqrt(mean_squared_error(target,test_predictions))
rf_rscore_new = r2_score(target,test_predictions)


print("Random Forest Model Metrics on New Data:")
print(f"MAE: {rf_mae_new}")
print(f"RMSE: {rf_rmse_new}")
print(f"R² Score: {rf_rscore_new}")


# ## Saving the model

# In[40]:


import pickle as pkl


# In[41]:


file_path = 'C:\\Users\\Ansah\\OneDrive - Ashesi University\\Documents\\Ashesi\\Sem4\\AI\\Lab02_peniel.ansah\\' + best_model.__class__.__name__ + '.pkl'

# Pickle dump the model to the specified file
with open(file_path, 'wb') as file:
    pkl.dump(best_model, file)


# In[42]:


# Constructing the file path dynamically
file_path = 'C:\\Users\\Ansah\\OneDrive - Ashesi University\\Documents\\Ashesi\\Sem4\\AI\\Lab02_peniel.ansah\\' + scale.__class__.__name__ + '.pkl'

# Pickle dump the object 'scale' to the specified file
with open(file_path, 'wb') as file:
    pkl.dump(scale, file)


# In[43]:


import sklearn
print(sklearn.__version__)

