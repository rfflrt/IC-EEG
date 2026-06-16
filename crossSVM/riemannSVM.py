import matplotlib.pyplot as plt
import numpy as np
import pyriemann as rmn
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def SVMxDawn_regular(train_path, test_path, classvec, metric = "riemann", estimator = "lwf", xDAWN_filters = 5):
    X_train = np.load(train_path, allow_pickle=True) # signals
    X_test  = np.load(test_path, allow_pickle=True)
    y_train = np.array([])           # labels
    part = len(X_train) // len(classvec)
    for i in range(len(classvec)):
        y_train = np.concatenate([y_train, i * np.ones(part, dtype=np.int64)])

    y_test = np.array([])
    part = len(X_test) // len(classvec)
    for i in range(len(classvec)):
        y_test = np.concatenate([y_test, i * np.ones(part, dtype=np.int64)])
    
    cov = rmn.estimation.XdawnCovariances(nfilter=xDAWN_filters, estimator=estimator)
    X_train_cov = cov.fit_transform(X_train, y_train)
    X_test_cov = cov.transform(X_test)

    # Classifier
    classifier = sklearn.svm.LinearSVC()
    tangent = rmn.tangentspace.TangentSpace(metric=metric)
    X_tang_train = tangent.fit_transform(X_train_cov)
    X_tang_test = tangent.transform(X_test_cov)
    classifier.fit(X_tang_train, y_train)
    y_pred = classifier.predict(X_tang_test)

    corr = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            corr += 1

    y_pred_train = classifier.predict(X_tang_train)
    corr_train = 0
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == y_train[i]:
            corr_train += 1

    print("Train Acc " + str(corr_train/len(y_pred_train)))
    print("Test Acc " + str(corr/len(y_pred)))

    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()