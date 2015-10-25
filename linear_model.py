import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn import preprocessing

# load data iris
iris = datasets.load_iris()

# iris list, target list / actual instance class
X_iris, y_iris = iris.data, iris.target

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# X is s new list consist of two very first columns which is Sepal length and width
# y is the class
X, y = X_iris[:,:2], y_iris

'''
train_test_split: split data into two list; train and test.
    - test_size = how much test data will be used.
    On the sample we set it to 0.25 which mean we will use 25% data as testing data and the res 75% as train data
    - random_state = how random you want to generate train and test data
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=33)

# Standarize the data to scale. Create a standard scale
scaler = StandardScaler().fit(X_train)

# apply the standard scale into data train
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print(X_test, X_test)

# make a plot color list
colors = ['red', 'greenyellow', 'blue']

'''
now let try to plot / visualize the data
    add each data into a
'''
for i in range(len(colors)):
    # add colum 1 to serial x
    xs = X_train[:, 0][y_train == i]
    # add colum 1 to serial y
    ys = X_train[:, 1][y_train == i]
    # add it to scatter plot
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal lenght')
plt.ylabel('Sepal width')
# plt.show()

clf = SGDClassifier()
clf.fit(X_train, y_train)
# print(clf.coef_)
# print(clf.intercept_)

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1,3)
fig.set_size_inches(10, 6)

# for i in [0,1,2]:
#     axes[i].set_aspect('equal')
#     axes[i].set_title('Class' + str(i) + ' versus the rest')
#     axes[i].set_xlabel('Sepal length')
#     axes[i].set_ylabel('Sepal width')
#     axes[i].set_xlim(x_min, x_max)
#     axes[i].set_ylim(y_min, y_max)
#     plt.sca(axes[i])
#     plt.scatter(X_train[:, 0], X_train[:,1], c=y_train, cmap=plt.cm.prism)
#     ys = (-clf.intercept_[i] - xs * clf.coef_[i,0]) / clf.coef_[i, 1]
#     plt.plot(xs, ys, hold=True)

# plt.show()
#


# # print(iris.head(), iris.head())
# # print(clf.predict(scaler.transform([[4.7, 3.1]])))
#
#
from sklearn import metrics
y_train_pred = clf.predict(X_train)
# print metrics.accuracy_score(y_train, y_train_pred)
#
y_pred = clf.predict(X_test)
# print metrics.accuracy_score(y_test, y_pred)
#
# print metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
# print metrics.confusion_matrix(y_test, y_pred)

# print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
# print(metrics.confusion_matrix(y_test, y_pred))

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline

clf = Pipeline([('scaler', StandardScaler()), ('linear_model', SGDClassifier())])
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)

scores = cross_val_score(clf, X, y, cv=cv)

# print(score)

from scipy.stats import sem
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

print(mean_score(scores))
