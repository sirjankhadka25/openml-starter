from pathlib import Path
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,confusion_matrix

wine=load_wine()
X=wine.data
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)

)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
matrix=confusion_matrix(y_test,predictions)
print("Accuracy Scores:",accuracy)
print("Confusion Matrix:")

print(matrix)

Path("models").mkdir(exist_ok=True)
joblib.dump(model,"models/wine_model.joblib")
print("Model saved Successfully")

