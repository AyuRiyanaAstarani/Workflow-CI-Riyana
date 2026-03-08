import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Model")

mlflow.sklearn.autolog()

train = pd.read_csv("train_preprocessed.csv")
test = pd.read_csv("test_preprocessed.csv")

X_train = train.drop("Exited", axis=1)
y_train = train["Exited"]

X_test = test.drop("Exited", axis=1)
y_test = test["Exited"]


with mlflow.start_run(run_name="RandomForest_Model"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

  
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    print("Training selesai")
    print("Accuracy:", acc)