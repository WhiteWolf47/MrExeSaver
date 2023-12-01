import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
import joblib
import pickle
import os
# Create a folder named "model" if it doesn't exist
model_folder = "models"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# for dataset_1 -> sep=',', for dataset_2 -> sep='|'
dataset = pd.read_csv('./datasets/dataset_1.csv', sep=',', low_memory=False)

dataset.head()
# dataset.describe()
# dataset.groupby(dataset['legitimate']).size()

# data preprocessing
X = dataset.drop(['ID', 'md5', 'legitimate'], axis=1).values
y = dataset['legitimate'].values

# Feature selection using ExtraTreesClassifier
extratrees = ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

features = [dataset.columns[2 + i] for i in np.argsort(extratrees.feature_importances_)[::-1][:nbfeatures]]

# List of classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=33),
    'DecisionTree': DecisionTreeClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'SGD': SGDClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'GaussianNB': GaussianNB()
}

# Train and evaluate each classifier
results = {}
for algo, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s Accuracy: %.2f%%" % (algo, score * 100))
    results[algo] = score
    # Save the model with its respective name
    model_filename = os.path.join(model_folder, f"{algo}_model.pkl")
    joblib.dump(clf, model_filename)
    #open('models/features.pkl', 'wb').write(pickle.dumps(features))
    print("%s Accuracy: %.2f%% - Model saved as %s" % (algo, score * 100, model_filename))

# Save the best model
'''best_classifier = max(results, key=results.get)
best_model = classifiers[best_classifier]
joblib.dump(best_model, "model/model.pkl")
open('model/features.pkl', 'wb').write(pickle.dumps(features))'''

# False Positives and Negatives for the best model
'''res = best_model.predict(X_new)
mt = confusion_matrix(y, res)
print("False positive rate : %.2f%%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
print('False negative rate : %.2f%%' % (mt[1][0] / float(sum(mt[1])) * 100))'''
