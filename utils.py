## Library
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as py

from scipy.stats import kurtosis, skew
import statistics as stat

from sklearn import preprocessing  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

import shap

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

SEED = 7

# Define a function to plot categorical variable
def plot_discrete_variable(dataframe, nameOfFeature,rotation_degree):
   
    # Ensuring the input parameter exist in the dataset
    if nameOfFeature not in dataframe.columns:
        print(f"Error: {nameOfFeature} column not found in the dataframe.")
        return
    
    counts = dataframe[nameOfFeature].value_counts()
    plt.figure(figsize=(8, 6))
    ax = counts.plot(kind='bar')
    plt.title(f'Bar Plot of {nameOfFeature}')
    plt.xlabel(nameOfFeature)
    plt.ylabel('Count')
    plt.xticks(rotation=rotation_degree)
    plt.grid(axis='y')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')
    
    plt.show()
    
# Define a function to plot continuous variable
def plot_box_plot(dataframe, nameOfFeature):
    
    # Ensuring the input parameter exist in the dataset
    if nameOfFeature not in dataframe.columns:
        print(f"Error: {nameOfFeature} column not found in the dataframe.")
        return
    
    # Ploting the boxplot
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(dataframe[nameOfFeature], vert=True)
    plt.title(f'Box Plot of {nameOfFeature}')
    plt.xlabel(nameOfFeature)
    plt.grid(True)
    
    # Calculate quartiles and outliers
    min_val = dataframe[nameOfFeature].min()
    Q1 = dataframe[nameOfFeature].quantile(0.25)
    median = dataframe[nameOfFeature].median()
    Q3 = dataframe[nameOfFeature].quantile(0.75)
    max_val = dataframe[nameOfFeature].max()
    
    # Labelling the box plot with min, Q1, median, Q3 and max value
    plt.text(1.3,min_val, f'Min: {min_val}', verticalalignment='top', horizontalalignment='center', color='blue',fontweight='light')
    plt.text(1.3,Q1, f'Q1: {Q1}', verticalalignment='top', horizontalalignment='center', color='blue', fontweight='light')
    plt.text(1.3,median, f'Median: {median}', verticalalignment='top', horizontalalignment='center', color='green', fontweight='light')
    plt.text(1.3,Q3, f'Q3: {Q3}', verticalalignment='top', horizontalalignment='center', color='blue', fontweight='light')
    plt.text(1.3,max_val, f'Max: {max_val}', verticalalignment='top', horizontalalignment='center', color='blue', fontweight='light')
    
    # Provide a quick view of mean, variance, skew, and kurtosis analysis
    DescribeFloatSkewKurt(dataframe, nameOfFeature)
    
    plt.show()

# Define a function to return some statistics for the box plot variables
def DescribeFloatSkewKurt(df,target):
        """
            A fundamental task in many statistical analyses is to characterize
            the location and variability of a data set. A further
            characterization of the data includes skewness and kurtosis.
            Skewness is a measure of symmetry, or more precisely, the lack
            of symmetry. A distribution, or data set, is symmetric if it
            looks the same to the left and right of the center point.
            Kurtosis is a measure of whether the data are heavy-tailed
            or light-tailed relative to a normal distribution. That is,
            data sets with high kurtosis tend to have heavy tails, or
            outliers. Data sets with low kurtosis tend to have light
            tails, or lack of outliers. A uniform distribution would
            be the extreme case
        """
        print('-*-'*25)
        print("Mean    : ".format(target), np.mean(df[target]))
        print("Var     : ".format(target), np.var(df[target]))
        print("Skew    : ".format(target), skew(df[target]))
        print("Kurt    : ".format(target), kurtosis(df[target]))
        print('-*-'*25)
 
# Define a function to plot aggregated time frame variable    
def plot_time_frame(dataframe, nameOfFeature):
    
    dataframe1 = dataframe.copy()
    
    dataframe1['year'] = dataframe1[nameOfFeature].dt.year
    dataframe1['month'] = dataframe1[nameOfFeature].dt.month
    dataframe1['day'] = dataframe1[nameOfFeature].dt.day
    dataframe1['day_of_week'] = dataframe1[nameOfFeature].dt.day_of_week
    dataframe1['quarter'] = dataframe1[nameOfFeature].dt.quarter
    
    fig,ax = plt.subplots(1,5,figsize = (16,6))
    
    dataframe1.groupby(['year']).size().plot(kind='bar',ax=ax[0])
    dataframe1.groupby(['month']).size().plot(kind='bar',ax=ax[1])
    dataframe1.groupby(['day']).size().plot(kind='bar',ax=ax[2])
    dataframe1.groupby(['day_of_week']).size().plot(kind='bar',ax=ax[3])
    dataframe1.groupby(['quarter']).size().plot(kind='bar',ax=ax[4])
    
    
    ax[0].set_title('Year', fontsize = 12)
    ax[1].set_title('Month', fontsize = 12)
    ax[2].set_title('Day', fontsize = 12)
    ax[3].set_title('Day of Week', fontsize = 12)
    ax[4].set_title('Quarter', fontsize = 12)
    
    
    for i in range(5):
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='both',which = 'both', labelsize = 8)
    
    plt.show()

# Define a function to plot time series variable
def plot_time_series(dataframe,nameOfFeature):
    
    dataframe1 = dataframe.copy()
    
    dataframe1['YearMonth'] = dataframe1[nameOfFeature].dt.to_period('M')
    monthly_counts = dataframe1.groupby('YearMonth').size()
    
    plt.figure(figsize = (15,6))
    monthly_counts.plot(marker='o',linestyle='-',color ='skyblue')
    plt.title('Yearly Plot')
    plt.xlabel('Year')
    plt.ylabel('Number of Applications')
    plt.xticks(rotation = 0)
    plt.grid(True)
    plt.show()


# Define a function to plot the distribution of the dataset 
def plot_variable_distribution(dataframe, column_name):
    
    if column_name not in dataframe.columns:
        print(f"Error: {column_name} column not found in the dataframe.")
        return
    
    plt.figure(figsize=(8, 6))
    plt.hist(dataframe[column_name], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()
    
# Define a function to map the target variable (1: loan default; 0: loan no default)
def target_mapping(x):
    
    labels = {1: ['Settled Bankruptcy', 'Charged Off'],
            0: ['Paid Off Loan', 'Settlement Paid Off']} 
    for label, status in labels.items():
        if x in status: 
            return label

# Define a function to plot heatmap        
def HeatMap(df,x=True):
        correlations = df.corr(numeric_only = True)
        ## Create color map ranging between two colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = sns.heatmap(correlations, cmap=cmap, vmin = -1, vmax=1.0,square=True, 
                          fmt='.2f',linewidths=.5, annot=x, cbar_kws={"shrink": .75})
        fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
        fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
        plt.tight_layout()
        plt.show()

# Define a function for base model
def GetBasedModel():
    
    basedModels = []
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))
    basedModels.append(('NN'   , MLPClassifier(max_iter=1000, random_state=11)))
    return basedModels

# Define a function for base model
def GetBasedModel2():
    
    basedModels = []
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    return basedModels

def BasedLine2(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    roc_curves = {}
    
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s model: Mean (%f), Std (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        # Calculate TPR and FPR for ROC curve
        tprs = []
        fprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train, test in kfold.split(X_train, y_train):
            model.fit(X_train.iloc[train], y_train.iloc[train])
            y_scores = model.predict_proba(X_train.iloc[test])[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train.iloc[test], y_scores)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_curves[name] = (mean_fpr, np.mean(tprs, axis=0), auc(mean_fpr, np.mean(tprs, axis=0)))
    
     # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, _) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} ROC Curve (AUC = {roc_curves[name][2]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend()
    plt.show()

    return names, results, roc_curves     

class PlotBoxR(object):
    
    
    def __Trace(self,nameOfFeature,value): 
    
        trace = go.Box(
            y=value,
            name = nameOfFeature
        )
        return trace

    def PlotResult(self,names,results):
        
        data = []

        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))


        py.iplot(data)

# Define a function to store the results
def ScoreDataFrame(names,results,roc_result):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
        return float(prc.format(f_val))

    scores = []
    roc_scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    for item in roc_result:
        roc_scores.append(roc_result[item][2])
        
    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores,'ROC_Score':roc_scores})
    return scoreDataFrame


# Define a function for scaled model
def GetScaledModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()

    pipelines = []
    pipelines.append((nameOfScaler+' LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+' SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC(probability=True))])))
    pipelines.append((nameOfScaler+' AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfScaler+' GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+' RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])  ))
    
    return pipelines

# Define a function to plot accuracy
def plot_accuracy(data):
    
    models = data.iloc[:,0]
    bar_width = 0.25
    index = np.arange(len(models))
    
    baseline = data.iloc[:,1]
    standard = data.iloc[:,4]
    minmax = data.iloc[:,7]
    
    fig, ax = plt.subplots()
    
    bar1 = ax.bar(index, baseline, bar_width, label='Baseline Accuracy')
    bar2 = ax.bar(index + bar_width, standard, bar_width, label='Standardization')
    bar2 = ax.bar(index + 2*bar_width, minmax, bar_width, label='MinMax')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Accuracy Evaluation: Baseline, Standardization & MinMax')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Define a function to plot roc score
def plot_roc(data):
    
    models = data.iloc[:,0]
    bar_width = 0.25
    index = np.arange(len(models))
    
    baseline = data.iloc[:,2]
    standard = data.iloc[:,5]
    minmax = data.iloc[:,8]
    
    fig, ax = plt.subplots()
    
    bar1 = ax.bar(index, baseline, bar_width, label='Baseline Accuracy')
    bar2 = ax.bar(index + bar_width, standard, bar_width, label='Standardization')
    bar2 = ax.bar(index + 2*bar_width, minmax, bar_width, label='MinMax')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('ROC Evaluation: Baseline, Standardization & MinMax')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Define a function to plot variable importance
def plot_importance(xtrain,ytrain,data):
    
    clf = ExtraTreesClassifier(n_estimators=250,random_state=7)
    clf.fit(xtrain, ytrain)
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    filter_idx = sorted_idx[:20]
    pos = np.arange(filter_idx.shape[0]) + .5
    plt.subplot()
    plt.barh(pos, feature_importance[filter_idx], align='center')
    plt.yticks(pos, data.columns[filter_idx])#boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def SHAP_functoin(X_train,X_test,y_train,y_test):
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier()
    clf.fit(X_train_scaled, y_train)
    
    explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test)

    shap.summary_plot(shap_values[1], X_test)
    
