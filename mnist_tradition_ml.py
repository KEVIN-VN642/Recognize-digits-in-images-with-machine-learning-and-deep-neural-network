import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Reading data
x_train = 'train-images.idx3-ubyte'
x_train = idx2numpy.convert_from_file(x_train)
print('X_train shape: ',x_train.shape)

y_train = 'train-labels.idx1-ubyte'
y_train = idx2numpy.convert_from_file(y_train)
print('y_train shape: ',y_train.shape)

x_test = 't10k-images.idx3-ubyte'
x_test = idx2numpy.convert_from_file(x_test)
print('X_test shape: ',x_test.shape)

y_test = 't10k-labels.idx1-ubyte'
y_test = idx2numpy.convert_from_file(y_test)
print('y_test shape: ',y_test.shape)

# Checking any missing values
def check_nan():
    print('Nans in x_train: ', np.isnan(x_train).any())
    print('Nans in y_train: ', np.isnan(y_train).any())
    print('Nans in x_test: ', np.isnan(x_test).any())
    print('Nans in y_test: ', np.isnan(y_test).any())

check_nan()

def train_img():
    # visualize some images
    train_random_index = np.random.randint(low=0, high=len(y_train))
    plt.imshow(x_train[train_random_index])

def test_img():
    test_random_index = np.random.randint(low=0, high=len(y_test))
    plt.imshow(x_test[test_random_index])
    plt.show()

def category_count():
    # label distribution
    Label = pd.DataFrame({'Label':pd.Series(y_train)})
    Label['Label'].value_counts().sort_index().plot(kind='bar', rot=90)
    plt.show()

# CONVERT IMAGES TO VECTORS
X_train = x_train.reshape(x_train.shape[0],-1)
X_test = x_test.reshape(x_test.shape[0],-1)


def simple_model(model):
    mod = model.fit(X_train,y_train)
    y_preds = mod.predict(X_test)
    y_probs = mod.predict_proba(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_preds))
    print('Precision: ', precision_score(y_test, y_preds, average='macro'))
    print('Recall: ', recall_score(y_test, y_preds, average='macro'))
    print('AUC: ', roc_auc_score(y_test, y_probs, multi_class='ovr'))
    fig, ax = plt.subplots(figsize=(10,5))
    ConfusionMatrixDisplay.from_predictions(y_test,y_preds,ax=ax)
    plt.show()

#Randomforest model
#model = RandomForestClassifier()
#simple_model(model)
#Accuracy:  0.9689
#Precision:  0.9687704704460816
#Recall:  0.9686226118454933
#AUC:  0.9991660908266216

#LogisticRegression model
#reg = 0.1
#model = LogisticRegression(C=1/reg, max_iter=1000, multi_class='auto', solver='lbfgs')
#simple_model(model)
#Accuracy:  0.9207
#Precision:  0.9199026954473336
#Recall:  0.9194174452149149
#AUC:  0.9922201201548418

#KNN model
#model =KNeighborsClassifier(n_neighbors=13)
#simple_model(model)
#Accuracy:  0.9653
#Precision:  0.9662167148282264
#Recall:  0.9650899714739358
#AUC:  0.9968022579027804

def hyper_params():
    print('RandomForest Model')
    model = RandomForestClassifier(random_state=1)
    params = {
        'n_estimators': [200]
    }

    gridsearch = GridSearchCV(model, scoring='accuracy', cv=6, param_grid= params)
    gridsearch.fit(X_train, y_train)
    print('Best params:', gridsearch.best_params_)
    mod = gridsearch.best_estimator_
    y_preds = mod.predict(X_test)
    y_probs = mod.predict_proba(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_preds))
    print('Precision: ', precision_score(y_test, y_preds, average='macro'))
    print('Recall: ', recall_score(y_test, y_preds, average='macro'))
    print('AUC: ', roc_auc_score(y_test, y_probs, multi_class='ovr'))
    print('Classification Report:\n', classification_report(y_test, y_preds))
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_preds, ax=ax)
    plt.show()

#hyper_params() for KNN
#Best params: {'n_neighbors': 3}
#Accuracy:  0.9705
#Precision:  0.9709116052270813
#Recall:  0.9701144344783679
#AUC:  0.9929978319451124
#hyper_params()

hyper_params()
#Randomforest
#Best params: {'n_estimators': 200}
#Accuracy:  0.9711
#Precision:  0.9708725679950723
#Recall:  0.9708911749894025
#AUC:  0.9992349608879005

print('\n>>> END <<<')
