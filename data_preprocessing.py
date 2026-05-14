import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

red = pd.read_csv(r"C:\Users\hoang\Downloads\wine+quality\winequality-red.csv", sep=";")
white = pd.read_csv(r"C:\Users\hoang\Downloads\wine+quality\winequality-white.csv", sep=";")

red["type"]   = 0  
white["type"] = 1  

combined = pd.concat([red, white], ignore_index=True)

X = combined.drop(columns=["quality", "type"]).values
y = combined["type"].values 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Remap nhãn về 0-based index (quality: 3..9 → 0..6)
class_names = ['Red Whine', 'White Wine']


print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} quality classes")
print(f"Quality levels: {list(class_names)}")
print(f"Red: {len(red)} | White: {len(white)}\n")
