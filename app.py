from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# Entrena el modelo si no existe
def entrenar_modelo():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    entrenar_modelo()

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        datos = [float(request.form[clave]) for clave in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]
        prediccion = model.predict([datos])[0]
        resultado = "POSITIVO (tiene diabetes)" if prediccion == 1 else "NEGATIVO (no tiene diabetes)"
    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
