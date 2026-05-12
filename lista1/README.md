# Metody systemowe i decyzyjne

## Opis projektu

Ręcznie zbudowany system decyzyjny (rule-based) do przewidywania choroby serca
na podstawie danych klinicznych pacjentów. Projekt obejmuje analizę eksploracyjną (EDA),
budowę i ocenę systemu jednoregułowego (Accuracy ~70%) oraz pięcioregułowego (Accuracy ~80%).

## Zbiór danych

**Heart Disease Dataset (Cleveland)** — UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/45/heart+disease

## Uruchomienie

```bash
# 1. Utwórz środowisko wirtualne (opcjonalnie, ale zalecane)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Zainstaluj zależności
pip install -r requirements.txt

# 3. Umieść dane w katalogu data/
mkdir data
cp processed.cleveland.data data/

# 4. Uruchom notebook lub main
jupyter notebook Raport.ipynb
py main.py
```

## Wyniki

| System | Liczba reguł | Accuracy (zbiór testowy) |
|--------|-------------|--------------------------|
| v1     | 1 reguła    | ~67%                     |
| v2     | 5 reguł     | ~71%                     |

## Reguły decyzyjne (v2)

1. `ca > 0` — zablokowane naczynia wieńcowe
2. `exang == 1` — ból wywołany wysiłkiem
3. `cp == 4` — brak bólu (paradoksalny predyktor)
4. `age > 55 AND thalach < 140` — starszy pacjent z niskim tętnem maks.
5. `sex == 1 AND age > 60 AND chol > 260` — starszy mężczyzna z wysokim cholesterolem
