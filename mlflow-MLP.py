import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler



# Cargamos el dataset
df = pd.read_excel("data/Dataset.xlsx")
df = df.dropna()

# Separamos columnas RON y MON
X = df.drop(columns=["RON", "MON"])
y = df[["RON", "MON"]]

# Split del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamos los datos:
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


experiment = mlflow.set_experiment("MLPRegressor:RON-MON")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo.
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo, previamente encontrados con la búsqueda de hiperparámetros
    solver = "lbfgs"
    max_iter = 1000
    learning_rate_init = 0.0001
    learning_rate = 'constant'
    hidden_layer_sizes = (200, 100, 50, 25)
    alpha = 0.1
    activation = 'relu'
    # Cree el modelo con los parámetros definidos y entrénelo
    model = MLPRegressor(solver=solver,
                         max_iter=max_iter,
                         learning_rate_init=learning_rate_init,
                         learning_rate=learning_rate,
                         hidden_layer_sizes=hidden_layer_sizes,
                         alpha=alpha,
                         activation=activation,
                         random_state=42)
    model.fit(X_train, y_train)
    # Realice predicciones de prueba
    y_pred_s = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(y_pred_s)

    # Registre los parámetros
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("learning_rate_init", learning_rate_init)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("hidden_layer_sizes", hidden_layer_sizes)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("activation", activation)

    # Registre el modelo
    mlflow.sklearn.log_model(model, "MLP Regressor")

    # Cree y registre la métrica de interés
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")