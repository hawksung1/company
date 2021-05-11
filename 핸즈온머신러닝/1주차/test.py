import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import preprocessing


def data_load(data_path):
    data = pd.read_csv(data_path)
    return data


def data_norm_preprocessing(data, feature_columns):
    data = data.dropna()
    sc = preprocessing.StandardScaler()
    data[feature_columns] = sc.fit_transform(data[feature_columns])
    return data, feature_columns


def data_pca_preprocessing(data, feature_columns):
    data = data.dropna()
    sc = preprocessing.StandardScaler()
    data[feature_columns] = sc.fit_transform(data[feature_columns])
    pca = PCA(n_components=0.8)
    data[feature_columns[:-1]] = pca.fit_transform(data[feature_columns])
    return data, feature_columns[:-1]


def data_lle_preprocessing(data, feature_columns):
    data = data.dropna()
    sc = preprocessing.StandardScaler()
    data[feature_columns] = sc.fit_transform(data[feature_columns])
    lle = LocallyLinearEmbedding(n_components=4)
    data[feature_columns[:-1]] = lle.fit_transform(data[feature_columns])
    return data, feature_columns[:-1]


def data_split(data, classification_feature_columns, label):
    X = data[classification_feature_columns]
    Y = data[label]
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    return X, Y, x_train, x_test, y_train, y_test


def _svc(X, Y, x_train, x_test, y_train, y_test, alpha=""):
    result = {}
    param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.01, 1, 100]}
    gs = GridSearchCV(SVC(), param_grid, return_train_score=True)
    gs_fitted = gs.fit(x_train, y_train)
    score = gs.score(x_test, y_test)
    cv_score = cross_validate(gs_fitted, X, Y)
    result["SVC"+alpha] = {
        "model": gs_fitted,
        "score": score,
        "cv_score": cv_score
    }
    return result


def _svr(X, Y, x_train, x_test, y_train, y_test, alpha=""):
    result = {}
    param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.01, 1, 100]}
    gs = GridSearchCV(SVR(), param_grid, return_train_score=True)
    gs_fitted = gs.fit(x_train, y_train)
    score = gs.score(x_test, y_test)
    cv_score = cross_validate(gs_fitted, X, Y)
    result["SVR"+alpha] = {
        "model": gs_fitted,
        "score": score,
        "cv_score": cv_score
    }
    return result

def run_supervised():
    data_path = 'https://grantmlong.com/data/titanic.csv'
    classification_label_column = "Embarked"
    feature_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch"]
    regression_label_column = "Fare"

    data = data_load(data_path)
    data_norm_preprocessed, feature_columns = data_norm_preprocessing(data, feature_columns)
    data_pca_preprocessed, pca_feature_columns = data_pca_preprocessing(data, feature_columns)
    data_lle_preprocessed, lle_feature_columns = data_lle_preprocessing(data, feature_columns)

    result = {}
    # classification
    X, Y, x_train, x_test, y_train, y_test = data_split(data_norm_preprocessed, feature_columns,
                                                        classification_label_column)
    result.update(_svc(X, Y, x_train, x_test, y_train, y_test))
    X, Y, x_train, x_test, y_train, y_test = data_split(data_pca_preprocessed, pca_feature_columns,
                                                        classification_label_column)
    result.update(_svc(X, Y, x_train, x_test, y_train, y_test, alpha="_pca"))
    X, Y, x_train, x_test, y_train, y_test = data_split(data_lle_preprocessed, lle_feature_columns,
                                                        classification_label_column)
    result.update(_svc(X, Y, x_train, x_test, y_train, y_test, alpha="_lle"))

    # regression
    X, Y, x_train, x_test, y_train, y_test = data_split(data_norm_preprocessed, feature_columns,
                                                        regression_label_column)
    result.update(_svr(X, Y, x_train, x_test, y_train, y_test))
    X, Y, x_train, x_test, y_train, y_test = data_split(data_pca_preprocessed, pca_feature_columns,
                                                        regression_label_column)
    result.update(_svr(X, Y, x_train, x_test, y_train, y_test, alpha="_pca"))
    X, Y, x_train, x_test, y_train, y_test = data_split(data_lle_preprocessed, lle_feature_columns,
                                                        regression_label_column)
    result.update(_svr(X, Y, x_train, x_test, y_train, y_test, alpha="_lle"))

    return result


if __name__ == '__main__':
    result = run_supervised()
    print(result)