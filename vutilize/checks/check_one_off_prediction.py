"""
    Stub to do one-off prediction using the trained model

    @author: mikhail.galkin
"""
#%% Load libraries
import sys
import warnings

#%% Load project's stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

from vutilize.config import models_dir
from vutilize.utils import load_model
from vutilize.utils import make_example_df
from vutilize.utils import mape

#%% Define some functions ------------------------------------------------------
def predict_example_df(model, target):
    examples = make_example_df(target=target)
    for e in examples:
        print(e[0])
        #%% Make prediction
        prediction = model.predict(e[0])
        mare = mape(e[1], prediction)
        # Print out
        print(f"\nReal Value: {e[1]}")
        print(f"Predicted Value: {list(prediction)}")
        print(f"Mean Absolute Ratio Error: {mare}\n")


#%% Main function for main.py ==================================================
def main(model_name=None, target="vu_estimated"):
    """Checks a one-off prediction using the trained model.

    Args:
        * model_name (str): Folder's  name for separatly saved model.\
            Defaults to None.
    """
    print(f"--------------- START: Check an one-off prediction ----------------")
    warnings.filterwarnings("ignore")
    if model_name is None:
        print(f"No model's name...")
        pass
    else:
        # # For model have been saved aside:
        model_path = models_dir / "mlmodels" / model_name / "model.pkl"
        model = load_model(model_path)
        predict_example_df(model=model, target=target)
        print(f"!!! DONE: Check an one-off prediction !!!")


#%% Workflow ===================================================================
if __name__ == "__main__":
    main(
        model_name="XGBRegressor-teu_estimated_9ffee2d3fa3d469ea7bfc9df7e5d317f",
        target="teu_estimated",
    )

    main(
        model_name="XGBRegressor-vu_estimated_0c1914e51e754b3893110f91279301f5",
        target="vu_estimated",
    )

#%%
