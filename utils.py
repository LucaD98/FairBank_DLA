import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import fairlearn.metrics as flm
import sklearn.metrics as sk

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from lime import lime_tabular
from IPython.display import display, HTML

def identify_missing_values(dataframe, columns):
   # Replace the '#NULL!' and '?' values in the columns with NaN (missing value)
    dataframe[columns] = dataframe[columns].replace(["#NULL!", " ?"], np.NaN)

    # Find columns with missing values
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    
    # Calculate the count of missing values for each column
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    
    # Calculate the ratio of missing values for each column as a percentage
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    
    # Create a DataFrame to display the missing values count and ratio
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    
    # Print the DataFrame showing missing values information
    print(missing_df)
    
    # Return the list of column names with missing values
    return variables_with_na
"""
    This function calculates and displays information about missing values in a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    columns (list): List of columns in the DataFrame.

    Returns:
    list: List of column names with missing values.
"""

def handle_missing_data(dataframe, columns, strategy, missing_values):
  
    # Fill missing values in each column based on the mode of groups defined by "mistriage" and "KTAS_expert"
    if strategy=='mean':
        for col in columns:
            # numeric values got the mean value, categoric the most common value
            try: dataframe[col] = dataframe[col].transform(lambda x: x.fillna(x.mean()))
            except: dataframe[col] = dataframe[col].transform(lambda x: x.fillna(x.mode()[0]))
    
    elif strategy=='mode':
        # grouping the data based on the unique combinations of values in the "mistriage" and "KTAS_expert" columns
        # create a separate group of rows for each unique combination of "mistriage" and "KTAS_expert" values
        grouped_data = dataframe.groupby(["hours-per-week", "income"])    
        for col in columns:
            dataframe[col] = grouped_data[col].transform(lambda x: x.fillna(x.mode()[0]))
    
    elif strategy =='delete': 
            for col in missing_values: 
                dataframe = dataframe.drop(columns=col) 
                columns.remove(col)
           
    else: print("Unknown strategy for handling missing values")
       
    return dataframe, columns
"""
    This function handles missing values in a Pandas DataFrame using various strategies.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    columns (list): List of columns to be processed.
    strategy (str): The strategy to handle missing values ('mean', 'mode', 'delete', 'estimate').
    missing_values (list): List of columns with missing values.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    list: List of columns with values, which are still there.
"""

def transformObjectToFloat(dataframe):

    # Convert commas to periods in columns with decimal comma separators
    dataframe.replace({',': '.'}, regex=True, inplace=True)

    # List of columns with 'object' data type containing numeric strings
    object_columns_with_numbers = dataframe.select_dtypes(include=['object']).apply(pd.to_numeric, errors='coerce').notna().all()

    # List of columns with numeric strings that need to be converted to 'float'
    columns_to_convert = object_columns_with_numbers[object_columns_with_numbers].index

    # Convert selected columns to 'float'
    dataframe[columns_to_convert] = dataframe[columns_to_convert].astype(float)
"""
    Convert numeric strings with decimal comma separators to float values in a Pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    None
"""

def show_distribution(dataframe, feature_names: str):
    if len(feature_names) == 1:
        sns.histplot(data=dataframe, x=feature_names[0])
    else:
        fig, axes = plt.subplots(nrows=len(feature_names), ncols=1, figsize=(12,len(feature_names)*4), dpi=300)
        for i, name in enumerate(feature_names):
            sns.histplot(data=dataframe, x=name, ax=axes[i])
"""
    Show the distribution of one or more features in a Pandas DataFrame using histograms.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    feature_names (str or list): Name(s) of the feature(s) to visualize.

    Returns:
    None
"""

def select_the_features(dataframe, strategy, alpha, n_features, target, chosen_sensitive_features):
    
    new_data = pd.DataFrame()
    transform_data = dataframe.drop(chosen_sensitive_features, axis=1)

    if strategy == 'None':
        new_data = transform_data
        
    elif strategy == 'Lasso':
        
        sel_ = SelectFromModel(Lasso(alpha=alpha, random_state=10))
        sel_.fit(transform_data.drop(target, axis=1), transform_data[target])

        for feature in sel_.get_feature_names_out():
            new_data[feature] = transform_data[feature]
        new_data[target] = transform_data[target]

    elif strategy == 'PCA':
          
        # Initialisieren Sie den PCA-Modell
        pca = PCA(n_components=n_features)

        # Führen Sie PCA auf Ihren Daten durch
        new_data = pd.DataFrame(pca.fit_transform(dataframe.drop(target, axis=1)))

        #  erklärte Varianz durch jede Hauptkomponente
        explained_variance_ratio = pca.explained_variance_ratio_
        new_data[target] = dataframe[target]

 
    new_data = pd.concat((dataframe[chosen_sensitive_features], new_data), axis = 1)
    return new_data
"""
    Selects features from a DataFrame using strategy 'Lasso regression' or 'PCA'.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing features and target variable.
    strategy (str): The strategy for feature selection ('Lasso' or 'PCA').
    alpha (float): Alpha value for Lasso regularization.
    n_features (int): Number of components for PCA.
    target (str): Name of the target feature

    Returns:
    pd.DataFrame: DataFrame with selected features based on the chosen strategy.
"""

def OneHotEncoder(dataframe, columns):
    # Check if there are any categorical columns with 'object' data type and select them
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    categorical_columns_with_values = [col for col in categorical_columns if dataframe[col].nunique() > 1]

    # Apply one-hot encoding to categorical columns with categorical values
    if categorical_columns_with_values:
        encoded_categorical = pd.get_dummies(dataframe[categorical_columns_with_values], dtype=int)
        dataframe = pd.concat([dataframe.drop(categorical_columns_with_values, axis=1), encoded_categorical], axis=1)

    # Remove the changed columns from list columns
    for col in categorical_columns_with_values:
        if col in dataframe.columns:
            columns.remove(col)

    return dataframe, columns
"""
    Apply one-hot encoding to categorical columns in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing categorical columns.
    columns (list): List of columns in the DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded categorical columns.
    list: Updated list of columns after one-hot encoding.
"""


def feature_importance(model, X_train, X_test, chosen_sensitive_features, class_names):
    plt.figure(figsize=(12, 6))
    name = ""

    if isinstance(model, RidgeClassifier):
        name = "Linear Regression"
        feature_importance = model.coef_[0]
        plt.bar(range(X_train.shape[1]), feature_importance)

    if isinstance(model, DecisionTreeClassifier) or isinstance(model, RandomForestClassifier):
        feature_importance = model.feature_importances_
        plt.bar(range(X_train.shape[1]), feature_importance)
        if isinstance(model, DecisionTreeClassifier): 
            name = "Decision Tree"

        if isinstance(model, RandomForestClassifier): 
            name = "Random Forest"

    elif isinstance(model, MLPClassifier):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # instantiate LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=X_train.columns,
                class_names=class_names,
                mode='classification'
            )

            # explain a random instance
            exp = explainer.explain_instance(
                data_row=X_test.iloc[np.random.randint(0,len(X_test))],
                predict_fn=model.predict_proba
                ) 

            # Erzeuge das HTML-Dokument mit einem begrenzten weiß gefärbten Bereich
            html_str = exp.as_html()
            html_str = html_str.replace('<body>', '<body><style>.lime {background-color: white;padding:10px;margin-right:-20px}</style>')

            # Füge einen Titel hinzu
            title = '<h1>Feature Importance for MLPClassifier</h1>'
            html_str = html_str.replace('<body>', f'<body>{title}')

            # Zeige das HTML-Dokument im Notebook an
            display(HTML(html_str))
        return
    
    plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title(f"Feature Importance for {name}")
    plt.show()
"""
    Calculate and visualize feature importance for various machine learning models.

    This function calculates and visualizes feature importance using different methods based on the type of machine learning model provided.

    Parameters:
    model: The trained machine learning model (e.g., DecisionTreeClassifier, RandomForestRegressor, LinearRegression, etc.).
    X_train (DataFrame): The training data features.
    Y_train (Series): The training data labels or target values.
    X_test (DataFrame): The test data features.
    Y_test (Series): The test data labels or target values.

    Returns:
    None
"""

def categorize_age(age):
    if age < 1:
        if 0 <= age <= 0.18:
            return "18-30"
        elif 0.18 < age <= 0.58:
            return "30-60"
        else:
            return "60-99"
    else:
        if 10<= age <= 30:
            return "18-30"
        elif 30 < age <= 60:
            return "30-60"
        else:
            return "60-99"
"""categorizes the age in three classes"""

def remove_sensitive_features(dataset, remove, not_removed, features, categorizer="age"):
    for item in remove:
        if item in not_removed:
            not_removed.remove(item)

    decoded_sensitive_features = None

    if categorizer in not_removed:
        dataset["Age_Category"] = dataset[categorizer].map(categorize_age)
        
        # Copy the desired columns into decoded_sensitive_features
        decoded_sensitive_features = dataset.loc[:, not_removed].copy()

        # Add the "Age_Category" column from "dataset" to decoded_sensitive_features
        decoded_sensitive_features[categorizer] = dataset["Age_Category"]

        # Remove the "Age_Category" column from the original "dataset"
        dataset.drop("Age_Category", axis=1, inplace=True)
    else:
        decoded_sensitive_features = dataset.loc[:, not_removed].copy()

    if 'sex' in not_removed:
        decoded_sensitive_features['sex'] = decoded_sensitive_features['sex'].replace({0.0: ' Female', 1.0: ' Male'})
    if 'race' in not_removed:
        decoded_sensitive_features['race'] = decoded_sensitive_features['race'].replace({0: ' White', 1: ' Other'})
 
    removable = []
    for feature in remove:
        if feature in features:
            features.remove(feature)
            removable.append(feature)
            
    dataset.drop(removable, axis=1, inplace=True)

    return dataset, decoded_sensitive_features
"""removes sensitive features from dataset and creates decoded_sensitive_features for fairness metrics"""