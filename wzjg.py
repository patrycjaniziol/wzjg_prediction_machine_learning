from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import random
import warnings
from sklearn.metrics import confusion_matrix
from pprint import pprint
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from missingpy import MissForest
from numpy import where
from matplotlib import pyplot
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from xgboost import plot_importance
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, precision_score, precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.metrics import plot_confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder, RobustScaler
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

warnings.filterwarnings('ignore')
warnings.warn("this will not show")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option('max_colwidth', 200)


plt.rcParams['savefig.facecolor'] = 'white'
# from pandas.tools import plotting
# % matplotlib inline


# auxiliary function


def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


warnings.filterwarnings('ignore')

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

# Define SMOTE
# oversample
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
