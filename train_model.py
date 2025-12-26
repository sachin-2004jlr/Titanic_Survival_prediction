import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

print("📥 Loading Data...")
# Ensure you have train.csv in the data folder!
try:
    df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    print("❌ ERROR: 'train.csv' not found in data/ folder.")
    exit()

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search: return title_search.group(1)
    return ""

df['Title'] = df['Name'].apply(get_title)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Deck']]
y = df['Survived']

num_features = ['Age', 'Fare', 'FamilySize']
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss'))])

print("🧠 Training XGBoost Model...")
model.fit(X, y)
joblib.dump(model, 'models/titanic_model.pkl')
print("✅ SUCCESS: Model saved to models/titanic_model.pkl")