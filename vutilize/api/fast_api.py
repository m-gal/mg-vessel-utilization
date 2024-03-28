"""
    RESTful API with FastAPI package

    You can run the application server using
        >>> uvicorn app.main:app --reload
    Here app.main indicates you use main.py file inside the 'app' directory
    and :app indicates our FastAPI instance name.

    run from Terminal:
        ./vessel_utilize => [ Open in Integrated Terminal ]
            >>> uvicorn api.fast_api:app --reload
                where: uvicorn {folder}.{module}:app --reload

    Example Value for testing:
    {
    "dim_a": 154,
    "dim_b": 49,
    "dim_c": 3,
    "dim_d": 23,
    "draught": 9.7
    }

    Response body must be:
    {
    "teu_esimated": 561,
    "vu_esimated": 0.4656
    }

    @author: mikhail.galkin
"""

# %% Load libraries
import uvicorn
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Initializing a FastAPI App Instance
app = FastAPI()


# Define request body
class ais_data(BaseModel):
    dim_a: float
    dim_b: float
    dim_c: float
    dim_d: float
    draught: float


def load_model(model_folder):
    p = Path(__file__).absolute()
    project_dir = p.parent.parent.parent
    model_path = project_dir / "models" / "mlmodels" / model_folder / "model.pkl"
    print(f"\nLoad model...")
    print(f"{model_path}")
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


# Load models
model_teu = load_model("XGBRegressor-teu_estimated_9ffee2d3fa3d469ea7bfc9df7e5d317f")
model_vu = load_model("XGBRegressor-vu_estimated_0c1914e51e754b3893110f91279301f5")


# Defining a Simple GET Request
@app.get("/utilize/")
def get_root():
    return {"Welcome": "The Vessels' Utilization API"}


# Creating an Endpoint to recieve the data to make prediction on.
@app.post("/utilize/predict")
async def predict(data: ais_data):

    data_list = [
        {
            "ais_dim_a": data.dim_a,
            "ais_dim_b": data.dim_b,
            "ais_dim_c": data.dim_c,
            "ais_dim_d": data.dim_d,
            "draught_fact": data.draught,
        }
    ]
    df = pd.DataFrame.from_records(data_list)

    teu_pred = round(model_teu.predict(df)[0].tolist(), 0)
    vu_pred = round(model_vu.predict(df)[0].tolist(), 4)

    return {"teu_esimated": teu_pred, "vu_esimated": vu_pred}


if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)
