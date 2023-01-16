import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors._base

import scipy.stats as stats
import pyforest
import shap
import ranking
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet
from sklearn.metrics import plot_confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import make_scorer, precision_score, precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from collections import Counter
# from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from matplotlib import pyplot
from numpy import where
from missingpy import MissForest
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import BorderlineSMOTE


import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('max_colwidth',200)

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap


import numpy as np
import random
import pandas as pd
# from pandas.tools import plotting
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score


import xgboost as xgb
import lightgbm as  lgb
from xgboost.sklearn import XGBClassifier
# from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


import warnings
warnings.filterwarnings('ignore')
plt.rcParams['savefig.facecolor']='white'
df0 = pd.read_csv('missforest1.csv', sep=',', encoding= 'unicode_escape')
df = df0


df = pd.DataFrame(df0)


df['25-OH-Vitamin D'] = pd.to_numeric(df['25-OH-Vitamin D'],errors = 'coerce')
df['albumin'] = pd.to_numeric(df['albumin'],errors = 'coerce')
df['APTT'] = pd.to_numeric(df['APTT'],errors = 'coerce')
df['protein (total)'] = pd.to_numeric(df['protein (total)'],errors = 'coerce')
df['bilirubin (total)'] = pd.to_numeric(df['bilirubin (total)'],errors = 'coerce')
df['ultra-sensitive CRP'] = pd.to_numeric(df['ultra-sensitive CRP'],errors = 'coerce')
df['ferritin'] = pd.to_numeric(df['ferritin'],errors = 'coerce')
df['folic acid'] = pd.to_numeric(df['folic acid'],errors = 'coerce')
df['basophils %'] = pd.to_numeric(df['basophils %'],errors = 'coerce')
df['basophils #'] = pd.to_numeric(df['basophils #'],errors = 'coerce')
df['eosinophils %'] = pd.to_numeric(df['eosinophils %'],errors = 'coerce')
df['eosinophils #'] = pd.to_numeric(df['eosinophils #'],errors = 'coerce')
df['erythroblasts %'] = pd.to_numeric(df['erythroblasts %'],errors = 'coerce')
df['erythroblasts #'] = pd.to_numeric(df['erythroblasts #'],errors = 'coerce')
df['erythrocytes %'] = pd.to_numeric(df['erythrocytes %'],errors = 'coerce')
df['HTC %'] = pd.to_numeric(df['HTC %'],errors = 'coerce')
df['hemoglobin level'] = pd.to_numeric(df['hemoglobin level'],errors = 'coerce')
df['leukocytes %'] = pd.to_numeric(df['leukocytes %'],errors = 'coerce')
df['lymphocytes %'] = pd.to_numeric(df['lymphocytes %'],errors = 'coerce')
df['lymphocytes #'] = pd.to_numeric(df['lymphocytes #'],errors = 'coerce')
df['MCH'] = pd.to_numeric(df['MCH'],errors = 'coerce')
df['MCHC'] = pd.to_numeric(df['MCHC'],errors = 'coerce')
df['MCV'] = pd.to_numeric(df['MCV'],errors = 'coerce')
df['monocytes %'] = pd.to_numeric(df['monocytes %'],errors = 'coerce')
df['monocytes #'] = pd.to_numeric(df['monocytes #'],errors = 'coerce')
df['MPV'] = pd.to_numeric(df['MPV'],errors = 'coerce')
df['neutrophils %'] = pd.to_numeric(df['neutrophils %'],errors = 'coerce')
df['neutrophils #'] = pd.to_numeric(df['neutrophils #'],errors = 'coerce')
df['immature granulocytes #'] = pd.to_numeric(df['immature granulocytes #'],errors = 'coerce')
df['P-LCR'] = pd.to_numeric(df['P-LCR'],errors = 'coerce')
df['PCT'] = pd.to_numeric(df['PCT'],errors = 'coerce')
df['PDW'] = pd.to_numeric(df['PDW'],errors = 'coerce')
df['RDW-SD'] = pd.to_numeric(df['RDW-SD'],errors = 'coerce')
df['RDW-CV'] = pd.to_numeric(df['RDW-CV'],errors = 'coerce')
df['PT/INR'] = pd.to_numeric(df['PT/INR'],errors = 'coerce')
df['potassium'] = pd.to_numeric(df['potassium'],errors = 'coerce')
X = df.drop(["Mayo score", "gender"], axis=1)
y = df["Mayo score"]
import seaborn as sns
sns.countplot(df['Mayo score'])
#plt.savefig("podzial_0123.png")
# Define SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

counter = Counter(y)
print(counter)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)

scaler = StandardScaler() 


scaler.fit(x_train)

x_train = scaler.transform(x_train)
# scale the test dataset
x_test = scaler.transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
dt_model = LogisticRegression()
dt_model.fit(x_train,y_train)
dt_predict = dt_model.predict(x_test)

print('Decision Tree - ',accuracy_score(dt_predict,y_test))
explainer = shap.Explainer(dt_model, x_train)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("linear4.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 2])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 2])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 2])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 2])


plt.savefig("regres_podzial.png")

shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns, class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','2','3'], show=False)
plt.savefig("linear4.png")
# KNN
knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)

explainer = shap.KernelExplainer(knn.predict_proba, x_train)
shap_values = explainer.shap_values(x_test, nsamples=180)
shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("knn.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.05])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.05])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.05])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.05])


plt.savefig("knn_podzial.png")
# Gradient boosting
gb = sklearn.ensemble.GradientBoostingClassifier()
gb.fit(x_train, y_train)
explainer = shap.KernelExplainer(gb.predict_proba, x_train)
shap_values = explainer.shap_values(x_test, nsamples=150)
shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("gradientboosting.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.11])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.11])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.11])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.11])


plt.savefig("gradientboosting_podzial.png")



# SVM
svm = sklearn.svm.SVC()
svm.fit(x_train, y_train)
explainer = shap.KernelExplainer(svm.predict_proba, x_train, link="logit")

shap_values = explainer.shap_values(x_test, nsamples=100)


shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("svm1.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.2])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.2])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.2])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.2])


plt.savefig("svm_podzial.png")
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_predict = rfc.predict(x_test)
print('Accuracy score:',accuracy_score(y_test, rfc_predict))
explainer = shap.Explainer(rfc, x_train)
check_additivity = False
shap_values = explainer.shap_values(x_test, check_additivity=check_additivity)
# shap.summary_plot(shap_values, X, max_display=10, plot_type="bar", class_inds='original')

shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("random_forest.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.04])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.04])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.04])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.04])


plt.savefig("random_forest_podzial.png")

# Decision Tree
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_predict = dt_model.predict(x_test)

print('Decision Tree - ',accuracy_score(dt_predict,y_test))
explainer = shap.TreeExplainer(dt_model, x_train)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns, class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'],class_inds='original', show=False)
plt.savefig("decision_tree.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.15])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.15])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.15])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.15])


plt.savefig("decision_tree_podzial.png")

# Extra Tree
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
etc_model = ExtraTreesClassifier()
etc_model.fit(x_train,y_train)
etc_predict = etc_model.predict(x_test)

print('Extra Tree Classifier - ',accuracy_score(etc_predict,y_test))
explainer = shap.TreeExplainer(etc_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X.values, max_display=10, plot_type="bar", feature_names = X.columns,class_inds='original', class_names=['Mayo score: 0', 'Mayo score: 1','Mayo score: 2','Mayo score: 3'], show=False)
plt.savefig("extraTree.png")
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)
ax1.title.set_text('Mayo score: 0')
shap.summary_plot(shap_values[0], X,max_display=10,  plot_type='bar', show=False)
plt.subplots_adjust(wspace = 8)
ax1.set_xlabel(r'SHAP values', fontsize=11)
ax1.set_xlim([0, 0.03])

ax2 = fig.add_subplot(222)
ax2.title.set_text('Mayo score: 1')
shap.summary_plot(shap_values[1], X, max_display=10, plot_type='bar', show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)
ax2.set_xlim([0, 0.03])

ax3 = fig.add_subplot(223)
ax3.title.set_text('Mayo score: 2')
shap.summary_plot(shap_values[2], X, max_display=10, plot_type='bar', show=False)
ax3.set_xlabel(r'SHAP values', fontsize=11)
ax3.set_xlim([0, 0.03])

ax4 = fig.add_subplot(224)
ax4.title.set_text('Mayo score: 3')
shap.summary_plot(shap_values[3], X, max_display=10, plot_type='bar', show=False)
ax4.set_xlabel(r'SHAP values', fontsize=11)
ax4.set_xlim([0, 0.03])


plt.savefig("extraTree_podział.png")
