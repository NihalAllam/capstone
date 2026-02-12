from sklearn.impute import KNNImputer

def knn_impute_train_test(X_train, X_test, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed
