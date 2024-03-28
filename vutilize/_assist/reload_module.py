"""
    Helps to reload project's module and get its inspections
    w\o reloading working space

    @author: mikhail.galkin
"""

#%% Import libs
import sys
import inspect
import importlib

sys.path.extend([".", "./.", "././.", "../..", "../../.."])
import vutilize

#%% CONFIG: Reload -------------------------------------------------------------
import vutilize.config

importlib.reload(vutilize.config)
from vutilize.config import project_dir

print(project_dir)

#%% UTILS: Reload --------------------------------------------------------------
import vutilize.utils

importlib.reload(vutilize.utils)
print(inspect.getsource(vutilize.utils.set_pd_options))

#%% CLASSES: Reload ------------------------------------------------------------
# import vessel_utilize.classes

# importlib.reload(vessel_utilize.classes)

#%% PLOTS: Reload --------------------------------------------------------------
import vutilize.plots

importlib.reload(vutilize.plots)
print(inspect.getsource(vutilize.plots.plot_residuals_errors))

#%% MODEL: RandomForestRegressor -----------------------------------------------
import vutilize.model._rf

importlib.reload(vutilize.model._rf)
print(inspect.getsource(vutilize.model._rf.model))
print(inspect.getsource(vutilize.model._rf.param_search))


#%% MODEL: XGBRegressor --------------------------------------------------------
import vutilize.model._xgb

importlib.reload(vutilize.model._xgb)
print(inspect.getsource(vutilize.model._xgb.model))
print(inspect.getsource(vutilize.model._xgb.param_search))

#%% shipdb_train_model_enrich -------------------------------------------------------=
import vutilize.data.shipdb_train_model_to_enrich

importlib.reload(vutilize.data.shipdb_train_model_to_enrich)
print(inspect.getsource(vutilize.data.shipdb_train_model_to_enrich.train_model))

#%%
