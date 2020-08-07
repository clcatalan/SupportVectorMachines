import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def printAll():
    print("-------PREDICTIONS-------")
    for i in range(len(y_pred)):
        actual = ''
        predicted = ''
        if(y_test[i] == 1 or y_pred[i] == 1):
            actual = classes[0]
            predicted = classes[0]
        elif(y_test[i] == 0 or y_pred[i] == 0):
            actual = classes[1]
            predicted = classes[1]

        print(f"[{i+1}]: Actual: {actual} | Predicted: {predicted}")


cancer = datasets.load_breast_cancer()


x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
classes = ['malignant', 'benign']

#if you use polynomial, it's gonna take a while to train, because it involves a lot more math (x^6, etc)
clf = svm.SVC(kernel="linear", C=2) #C = soft margin
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)

print("-------RESULTS-------")
print("|")
print("|")
print("|")
print(f"Model Accurracy: {acc}")
print("|")
print("|")
printAll()







