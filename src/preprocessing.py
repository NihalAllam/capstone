import numpy as np
from sklearn.impute import SimpleImputer

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

def replace_invalid_zeros(df):
    df_copy = df.copy()
    df_copy[INVALID_ZERO_COLUMNS] = df_copy[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return df_copy

def drop_missing_rows(df):
    return df.dropna()

def mean_impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

def median_impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

def mode_impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed
