import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df[df.gender != 'Other']
df['gender'].unique()
df['ever_married'].unique()
df['work_type'].unique()
df['smoking_status'].value_counts()
df.drop(columns = 'smoking_status', inplace=True)

df.isnull().sum()

sns.distplot(df['bmi'], hist = True, kde =True)

sns.distplot(df['avg_glucose_level'], hist = True, kde =True)

imputer = KNNImputer(n_neighbors=2)
df['bmi']= imputer.fit_transform(df['bmi'].values.reshape(-1,1))

df.drop(columns = 'id', inplace=True)

df_new = pd.get_dummies(df,drop_first=True)
df_new

df_new.info()

df_new['stroke'].value_counts()

X, y = df_new.drop(columns = 'stroke'),df_new['stroke']
X_cat = X[['gender_Male','ever_married_Yes','work_type_Never_worked','work_type_Private','work_type_Self-employed', 'work_type_children', 'Residence_type_Urban']]
chi_scores = chi2(X_cat,y)
chi = pd.Series(chi_scores[1], index =X_cat.columns)
chi.plot.bar()

chi

X = df_new.drop(columns = ['gender_Male','work_type_Never_worked','work_type_Private','Residence_type_Urban'])
X_num = X[[	'age',	'hypertension',	'heart_disease',	'avg_glucose_level',	'bmi']]
X_num

sns.heatmap( X_num.corr(), annot=True, cmap='coolwarm' )

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9, random_state=702, stratify = y)

X_train, y_train = SMOTE().fit_resample(X_train, y_train)

y_train.value_counts()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train.values.reshape(-1,1))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier

lr = LogisticRegression()
svc = SVC()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
adb = AdaBoostClassifier()
xgb = XGBClassifier()

lr.fit(X_train,y_train)
svc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
rfc.fit(X_train,y_train)
adb.fit(X_train,y_train)
knn.fit(X_train,y_train)
xgb.fit(X_train,y_train)
#cb.fit(X_train,y_train)

X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test.values.reshape(-1,1))

lr.score (X_test, y_test),svc.score (X_test, y_test),dtc.score (X_test, y_test),rfc.score (X_test, y_test),adb.score(X_test, y_test), knn.score(X_test, y_test), xgb.score(X_test, y_test)

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
y_pred = xgb.predict(X_test)
matrix = cm(y_test,y_pred)
sns.heatmap( matrix, annot=True, cmap='coolwarm' )

accuracy_score(y_test,y_pred)
