# Interpretación:
# Este gráfico muestra cómo la regresión logística estima la probabilidad de que un paciente tenga una enfermedad cardíaca
# en función de su edad. Los puntos morados son los datos reales (0 = no, 1 = sí), mientras que la curva negra representa
# cómo el modelo "cree" que cambia esa probabilidad con la edad.
# La forma sigmoide es típica de la regresión logística, ya que su salida está limitada entre 0 y 1.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Cargar el dataset
df = pd.read_csv(r'C:\Users\Usuario\Desktop\PC\UDLA\SEMESTRE 8\INTELIGENCIA ARTIFICIAL I\regresion_logistica\heart.csv')

# Seleccionar solo una variable predictora (por ejemplo: 'age') y la variable objetivo
X = df[['age']]
y = df['target']

# Entrenar modelo
modelo = LogisticRegression()
modelo.fit(X, y)

# Crear puntos para graficar la curva sigmoide
X_test = np.linspace(df['age'].min(), df['age'].max(), 300).reshape(-1, 1)
y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1

# Graficar
plt.figure(figsize=(8, 5))
plt.scatter(df['age'], y, color='purple', label='Datos reales', alpha=0.5)
plt.plot(X_test, y_prob, color='black', linewidth=3, label='Curva sigmoide')
plt.xlabel("Edad (X)")
plt.ylabel("Probabilidad de enfermedad (Y)")
plt.title("Regresión Logística - Probabilidad según Edad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
