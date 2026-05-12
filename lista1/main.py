import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

kolumny = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv('data/processed.cleveland.data', header=None, names=kolumny, na_values='?')
df = df.dropna()
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"Wczytano {len(df)} rekordów. Brakujące wartości: {df.isnull().sum().sum()}")
print(f"\nRozkład klas docelowych:\n{df['target'].value_counts().rename({0: 'Zdrowy (0)', 1: 'Chory (1)'})}")
df.head()




X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Zbiór treningowy: {X_train.shape[0]} pacjentów")
print(f"Zbiór testowy:    {X_test.shape[0]} pacjentów")

train_df = X_train.copy()
train_df['target'] = y_train



cechy_numeryczne = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

stats = train_df[cechy_numeryczne].agg(['mean', 'median', 'min', 'max', 'std']).T
stats.columns = ['Średnia', 'Mediana', 'Min', 'Max', 'Odch. std.']
stats = stats.round(2)

print("Statystyki opisowe cech numerycznych (zbiór treningowy):")
print(stats.to_string())



# Sprawdzenie unikalnych wartości i rozkładu klas
cechy_kategoryczne = ['cp', 'ca', 'exang', 'sex', 'thal']

print("Rozkład unikalnych wartości i proporcje klas (zbiór treningowy):")
for col in cechy_kategoryczne:
    print(f"\n--- {col} ---")
    tab = pd.crosstab(train_df[col], train_df['target'],
                      rownames=[col], colnames=['Diagnoza'])
    tab.columns = ['Zdrowy (0)', 'Chory (1)']
    tab['% Chorych'] = (tab['Chory (1)'] / (tab['Zdrowy (0)'] + tab['Chory (1)']) * 100).round(1)
    print(tab)




def predict_v1(row) -> int:
    return 1 if row['ca'] > 0 else 0

acc1 = accuracy_score(y_test, X_test.apply(predict_v1, axis=1))
print(f"Dokładność (1 reguła — tylko ca): {acc1:.2%}")



def predict_v2(row) -> int:
    # Reguła 1: Zablokowane naczynia
    if row['ca'] > 0.:
        return 1
    # Reguła 2: Ból wysiłkowy
    elif row['exang'] == 1:
        return 1
    # Reguła 3: Brak bólu
    elif row['cp'] == 4:
        return 1
    # Reguła 4: Starszy pacjent z niskim tętnem maks.
    elif row['age'] > 55 and row['thalach'] < 140:
        return 1
    # Reguła 5: Starszy mężczyzna z wysokim cholesterolem
    elif row['sex'] == 1.0 and row['age'] > 60 and row['chol'] > 260:
        return 1
    else:
        return 0

y_pred_v1 = X_test.apply(predict_v1, axis=1)
y_pred_v2 = X_test.apply(predict_v2, axis=1)

acc1 = accuracy_score(y_test, y_pred_v1)
acc2 = accuracy_score(y_test, y_pred_v2)

print("=" * 45)
print(f"  Dokładność (1 reguła):  {acc1:.2%}")
print(f"  Dokładność (5 reguł):   {acc2:.2%}")
print(f"  Poprawa:                {(acc2 - acc1)*100:+.2f} pp")
print("=" * 45)