#Basic
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

#Visuals
import matplotlib.pyplot as plt
import seaborn as sns

#Text
import string
import re
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
words = set(nltk.corpus.words.words())

#SkLearn
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

#Other
from lime import lime_text
from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy.special import softmax


def modeling(classifier, params, x_train, y_train, x_val, y_val):
    '''
    Grid search hyperparameter tuning using training and validation sets
    
    Args:
        classifier: model to use
        params: search space for hyperparameter tuning
        x_train: training predictor variables
        y_train: training targets
        x_val: validation predictor variables
        y_val: validation targets
        
    Return:
        classifier name
        best parameters
        AUC
        runtime
    '''
    
    print(str(classifier), 'started')

    start = datetime.now()
    ## pipeline
    model = pipeline.Pipeline([("classifier", classifier)])
        
    grid_search = GridSearchCV(model, params,  n_jobs=-1, scoring='roc_auc', cv=2)
    grid_search.fit(x_train, y_train)
    
    if isinstance(classifier, LinearSVC):
        predicted_prob = grid_search.decision_function(x_val)
        predicted_prob = softmax(predicted_prob, axis=1)
    else:
        predicted_prob = grid_search.predict_proba(x_val)
        

    return [str(classifier), grid_search.best_params_,
            metrics.roc_auc_score(y_val, predicted_prob, multi_class="ovr"),
            datetime.now()-start,
           ]


def experiment(x_train, y_train, x_val, y_val, name:str):
    '''
    Train 5 models for given training features and sets
    
    Args:
        x_train: training predictor variables
        y_train: training targets
        x_val: validation predictor variables
        y_val: validation targets
        name: name of experiment
        
    Return:
        classifier name
        best parameters
        AUC
        runtime
    '''
    
    results = []
    start = datetime.now()

    #Naive Bayes
    results.append(modeling(MultinomialNB(), {'classifier__alpha': [.01, .1, 1]},
                              X_train, y_train, X_val, y_val))

    # Logistic Regression
    results.append(modeling(LogisticRegression(), {'classifier__C': np.logspace(-4, 4, 4)},
                           X_train, y_train, X_val, y_val))

    # Random Forest
    results.append(modeling(RandomForestClassifier(n_jobs=14), {'classifier__n_estimators': [100, 500, 1000], 'classifier__max_depth':[1,2,3]},
                           X_train, y_train, X_val, y_val))

    #SVM
    results.append(modeling(LinearSVC(), {},
                           X_train, y_train, X_val, y_val))

    print(name, datetime.now()-start)

    for result in results:
        result.append(name)
    
    return results


def test_results(X_train, y_train, X_test, y_test, model):
    '''
    Fit Model given training set, and output metrics on test set
    Args:
        X_train
        y
        
    Return:
        Accuracy
        Precision
        Recall
        Classification Report
        Confusion Matrix
        ROC Curves
        Precision Recall Curves
    '''
    
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    if isinstance(model, LinearSVC):
        predicted_prob = model.decision_function(X_test)
        predicted_prob = softmax(predicted_prob, axis=1)
    else:
        predicted_prob = model.predict_proba(X_test)
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, 
                                multi_class="ovr")
    print("Accuracy:",  round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5.5, 9.5, forward=True)
    
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
                               predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, 
                  label='{0} (area={1:0.2f})'.format(classes[i], 
                                  metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0,1], [0,1], color='black', lw=3, linestyle='-')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
              xlabel='False Positive Rate', 
              ylabel="True Positive Rate", 
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(False)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
                     y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, 
                   label='{0} (area={1:0.2f})'.format(classes[i], 
                                      metrics.auc(recall, precision))
                  )
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="lower left")
    ax[1].grid(False)
    plt.show()