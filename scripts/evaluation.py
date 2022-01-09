import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if __name__ == "__main__":
    print('lendo o modelo')
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
        
    model = pickle.load(open("xgboost-model", "rb"))
    print('lendo base de teste')
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    #df.drop(columns=[0], inplace=True)
    x = df.iloc[:, 1:]
    X_test = xgboost.DMatrix(x.values)

    predictions = model.predict(X_test)
    print(predictions)
    predictions = np.array(predictions)
    y_pred = []
    for i in predictions:
        pred = np.where(i == max(i))[0][0]
        y_pred.append(pred)
    print('calculo acurácia')
    acuracia = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, predictions)
    #precisao = precision_score(y_test, predictions)
    #recall = recall_score(y_test, predictions)
    print(acuracia)
    report_dict = {
        "classification_wine_metrics": {
            "acuracia": {"value":acuracia,"standard_deviation": "NaN"}
        }
    }
    print('Salvando as métricas')
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
