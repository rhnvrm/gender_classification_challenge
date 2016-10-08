from sklearn import tree, svm, naive_bayes
from sklearn.metrics import accuracy_score


# CHALLENGE - create 3 more classifiers...
# 1 tree.DecisionTreeClassifier()
# 2 svm.SVC()
# 3 naive_bayes.GaussianNB()

classifiers = [tree.DecisionTreeClassifier(), svm.SVC(),
               naive_bayes.GaussianNB()]

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], ]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female']

x_pred = [[159, 55, 37], [171, 75, 42], [181, 85, 43]]
y_pred = ['female', 'male', 'male']

# CHALLENGE - ...and train them on our data
for clf in classifiers:
    clf = clf.fit(X, Y)
    predictions = clf.predict(x_pred)
    # CHALLENGE compare their reusults and print the best one!
    print(clf.__class__.__name__, accuracy_score(y_pred, predictions))
