from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def descisionTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=20, min_samples_leaf=20)
    clf.fit(X_train, y_train)

    predication = clf.predict(X_test)
    config_matrix = confusion_matrix(y_test, predication)

    acc_score = accuracy_score(y_test, predication)

    return config_matrix, acc_score


def NB(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    predication = clf.predict(X_test)
    config_matrix = confusion_matrix(y_test, predication)

    acc_score = accuracy_score(y_test, predication)

    return config_matrix, acc_score


def RForrest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    predication = clf.predict(X_test)
    config_matrix = confusion_matrix(y_test, predication)

    acc_score = accuracy_score(y_test, predication)

    return config_matrix, acc_score


def LogRegression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=1, solver='lbfgs', max_iter=10000)
    clf.fit(X_train, y_train)

    predication = clf.predict(X_test)
    config_matrix = confusion_matrix(y_test, predication)

    acc_score = accuracy_score(y_test, predication)

    return config_matrix, acc_score


def SVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    predication = clf.predict(X_test)
    config_matrix = confusion_matrix(y_test, predication)

    acc_score = accuracy_score(y_test, predication)

    return config_matrix, acc_score

def transform(dataset):
    le = LabelEncoder()
    le.fit_transform(dataset)

    return le.transform(dataset)

def getPrediction(alg, X_train, X_test, y_train, y_test):
    if(alg == 'Decision tree'):
        return descisionTree(X_train, X_test, y_train, y_test)
    if(alg == 'Naive Bayes'):
        return NB(X_train, X_test, y_train, y_test)
    if(alg == 'Random forest'):
        return RForrest(X_train, X_test, y_train, y_test)
    if(alg == 'Logistic regression'):
        return LogRegression(X_train, X_test, y_train, y_test)
    if(alg == 'SVM'):
        return SVM(X_train, X_test, y_train, y_test)