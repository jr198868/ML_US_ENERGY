from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
import sys
import csv

from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm

from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import pearsonr
import time



def dataclean(target_data):
    data = pd.read_csv(target_data)

    data_list = data.values.tolist()

    for i in range(len(data_list)):
        data_list[i] = data_list[i][1:]
        
    return data_list


def Regressionmodel(target_data, model):
    data = pd.read_csv(target_data)
    
    # extract the independent variables (X) and dependent variable (y)
    X = data[['COAL, Thousand Short Tons', 'NATURALGAS, Billion Cubic Feet', 'ELECTRICITY, Million Kilowatthours',
              'PETRO_INDUSTRIAL, Thousand Barrels per Day', 'PETRO_RESIDENTIAL, Thousand Barrels per Day',
              'PETRO_COMMERCIAL, Thousand Barrels per Day', 'PETRO_TRANSPORTATION, Thousand Barrels per Day',
              'PETRO_ELECTRICPOWER, Thousand Barrels per Day']]

    y_lca = data['CO2_ based on LCA calculation (Million Metric Tons)']
    y_ml = data['CO2, Million Metric Tons']

    # Shuffle data
    X, y_ml = shuffle(X, y_ml, random_state = 42)

    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y_ml, test_size=0.3, random_state=42)
    
    regressor = model

    # Fit the model to the data
    regressor.fit(X_train_ml, y_train_ml)


    # Make predictions on the training data
    y_pred_train_ml = regressor.predict(X_train_ml)

    # Calculate the R-squared score for the training data
    r2_train = r2_score(y_train_ml, y_pred_train_ml)
    print("R² (Training):", r2_train)

    # Use the model to make predictions on the test data
    y_pred_ml = regressor.predict(X_test_ml)
    X_test_ml_list = X_test_ml.values.tolist()



    X_test_ml_y_pred_ml = zip(X_test_ml_list, y_pred_ml)
    X_test_ml_y_pred_ml_list = []

    for i in X_test_ml_y_pred_ml:
        item = i[0] + [i[-1]]
        X_test_ml_y_pred_ml_list.append(item)
    
    return X_test_ml_y_pred_ml_list




def is_sublist(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    
    # Check if list1 is longer than list2
    if len1 > len2:
        return False
    
    # Iterate through list2 using sliding window
    for i in range(len2 - len1 + 1):
        if set(list2[i:i+len1]) == set(list1):
            return True
    
    return False


def findlcaml(X_test_ml_y_pred_ml_list, data_list):
    result = []
    for i in X_test_ml_y_pred_ml_list:
        for j in data_list:
            if is_sublist(i[:-1], j):
                item = j + [i[-1]]
                result.append(item) 
    return result
    

    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("How to use: python3 ML_LCA_STATISTICS.py [target .csv file] regressor model" )
    
    else: 
        start_time = time.time()
        # Code to measure the execution time of

        target_data = sys.argv[1]
        regressor = sys.argv[2]
        
        if regressor == 'LR':
            model = LinearRegression()
        if regressor == 'RF':
            model = RandomForestRegressor(n_estimators=1000, random_state=42) # 1000 Trees
        if regressor == 'DT':
            model = DecisionTreeRegressor()
        if regressor == 'GB':
            model = xgb.XGBRegressor()
        if regressor == 'KNN':
            model = KNeighborsRegressor(n_neighbors=3) # k = 3
        if regressor == 'SVR':
            model = svm.SVR()
        
        
        data_list = dataclean(target_data)
        X_test_ml_y_pred_ml_list = Regressionmodel(target_data, model)
        result = findlcaml(X_test_ml_y_pred_ml_list, data_list)
        

        final_result = []
        for i in result:
            final_result.append(i[-3:])
            
            
        # Sample data
        lca_ttest = []
        ml_ttest = []
        pred_ttest = []

        for i in final_result:
            lca_ttest.append(i[0])

        for i in final_result:
            ml_ttest.append(i[1])

        for i in final_result:
            pred_ttest.append(i[2])

        # Perform t-test
        t_statistic_lca_ml, p_value_lca_ml = ttest_ind(lca_ttest, ml_ttest)
        t_statistic_pred_ml, p_value_pred_ml = ttest_ind(pred_ttest, ml_ttest)

        # Print the results
        print("T-Statistic for lca and ml:", t_statistic_lca_ml)
        print("P-Value for lca and ml:", p_value_lca_ml)


        # Print the results
        print("T-Statistic for pred and ml:", t_statistic_pred_ml)
        print("P-Value for pred and ml:", p_value_pred_ml)

        print('############################################')

        # Perform ANOVA
        f_statistic, p_value = f_oneway(lca_ttest, ml_ttest, pred_ttest)

        # Print the results
        print("F-Statistic:", f_statistic)
        print("P-Value:", p_value)

        print('############################################')

        # Calculate Pearson correlation coefficient and p-value
        corr_coeff_lca_ml, p_value_lca_ml = pearsonr(lca_ttest, ml_ttest)
        corr_coeff_pred_ml, p_value_pred_ml = pearsonr(pred_ttest, ml_ttest)

        # Print the results
        print("Pearson Correlation Coefficient lca vs ml:", corr_coeff_lca_ml)
        print("P-Value lca vs ml:", p_value_lca_ml)
        print("Pearson Correlation Coefficient pred vs ml:", corr_coeff_pred_ml)
        print("P-Value pred vs ml:", p_value_pred_ml)
    
            
        final_result.insert(0, ['CO2_ based on LCA calculation (Million Metric Tons)', 'Actual CO2, Million Metric Tons', 'Predicted CO2, Million Metric Tons'])
        
        with open ('lca_actual_pred_{}.csv'.format(regressor), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(final_result)
        
        end_time = time.time()
        total_time = (end_time - start_time)/60
        print("Total execution time:", total_time, "min")
     