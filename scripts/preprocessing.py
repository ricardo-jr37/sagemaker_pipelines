import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    print('lendo a base')
    #df = pd.read_csv(f"{base_dir}/input/wine.data", header=None)
    df = pd.read_csv(f"{base_dir}/input/raw.csv")
    #Separando os atributos e a target
    print(df.columns)
    print(df['0'].value_counts)
    print('separando x e y')
    y, x = df.iloc[:, 0], df.iloc[:, 1:]
    #Separando entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=0, test_size=0.2)
    #Criando pipeline para normalizar os dados
    print('Criando pipeline')
    pipe = Pipeline([('scaler', StandardScaler())])
    X_train_processed = pipe.fit_transform(X_train)
    X_test_processed = pipe.transform(X_test)
    y_train = y_train.to_numpy().reshape(len(y_train), 1)
    y_test = y_test.to_numpy().reshape(len(y_test), 1)
    print('Consolidando os dados de treino, teste e validação')
    #Consolidando os dados de treino, teste e validação
    df_train = pd.DataFrame(np.concatenate((y_train, X_train_processed), axis=1))
    df_test = np.concatenate((y_test, X_test_processed), axis=1)
    #Dividindo entre teste e validação
    print('Dividindo entre teste e validação')
    test, validation = np.split(df_test, 2)
    #Salvando os dados de treino, teste e validação
    print('Salvando os dados de treino, teste e validação')
    pd.DataFrame(df_train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
