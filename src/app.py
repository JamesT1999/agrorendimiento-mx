from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_rendimiento_agricola.pkl"

# Cargar el pipeline completo
model = joblib.load(MODEL_PATH)

# Columnas que el modelo espera recibir
FEATURE_COLUMNS = [
    "Anio",
    "Nomestado",
    "Nomddr",
    "Nomcader",
    "Nommunicipio",
    "Nomcicloproductivo",
    "Nommodalidad",
    "Nomunidad",
    "Nomcultivo",
    "Sembrada",
    "Siniestrada",
    "Preciomediorural",
]

NUMERIC_COLUMNS = [
    "Anio",
    "Sembrada",
    "Siniestrada",
    "Preciomediorural",
]


def get_category_options():
    """Obtiene las categorías aprendidas por el OneHotEncoder."""
    preprocessor = model.named_steps["preprocessor"]

    categorical_columns = None

    for name, transformer, columns in preprocessor.transformers_:
        if name == "categorical":
            categorical_columns = list(columns)
            encoder = transformer
            break

    if categorical_columns is None:
        raise ValueError("No se encontró el transformador categórico.")

    return {
        column: sorted(
            [str(value) for value in categories],
            key=str.lower
        )
        for column, categories in zip(
            categorical_columns,
            encoder.categories_
        )
    }


CATEGORY_OPTIONS = get_category_options()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()

            input_data = {
                "Anio": int(form_data["Anio"]),
                "Nomestado": form_data["Nomestado"],
                "Nomddr": form_data["Nomddr"],
                "Nomcader": form_data["Nomcader"],
                "Nommunicipio": form_data["Nommunicipio"],
                "Nomcicloproductivo": form_data["Nomcicloproductivo"],
                "Nommodalidad": form_data["Nommodalidad"],
                "Nomunidad": form_data["Nomunidad"],
                "Nomcultivo": form_data["Nomcultivo"],
                "Sembrada": float(form_data["Sembrada"]),
                "Siniestrada": float(form_data["Siniestrada"]),
                "Preciomediorural": float(form_data["Preciomediorural"]),
            }

            if input_data["Sembrada"] <= 0:
                raise ValueError(
                    "La superficie sembrada debe ser mayor que cero."
                )

            if input_data["Siniestrada"] < 0:
                raise ValueError(
                    "La superficie siniestrada no puede ser negativa."
                )

            if input_data["Siniestrada"] > input_data["Sembrada"]:
                raise ValueError(
                    "La superficie siniestrada no puede superar "
                    "la superficie sembrada."
                )

            if input_data["Preciomediorural"] < 0:
                raise ValueError(
                    "El precio medio rural no puede ser negativo."
                )

            input_df = pd.DataFrame(
                [input_data],
                columns=FEATURE_COLUMNS
            )

            prediction = float(model.predict(input_df)[0])

        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)

        except Exception:
            error = (
                "No fue posible realizar la predicción. "
                "Verifica los datos ingresados."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        categories=CATEGORY_OPTIONS,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True) 
