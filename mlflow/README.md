<img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" width="1000">

# ML Ops: Locally model tracking with Mlflow.

Use localhost (or your laptop) to log experiments, runs, models and its metadata
with Tracking Server\
and later with Model Registry to register models and deploy them to "production"
as a REST endpoint locally.

* The Storage of an MLflow Tracking Server will being kept in separeted directory **[./mlflow]**
__________________________________

## # Setup MLflow Tracking Server:
----------------------------------
__Each time when the project is opened newally__
(or at first time you start train models)\
you must to start MLflow Tracking Server locally:

1. Open in Integrated Terminal in the **[./mlflow]** and run Tracking Server:
    * `>mlflow server --backend-store-uri sqlite:///mlregistry.db
    --default-artifact-root smb://{PATH-TO-ROOT-PROJECT-FOLDER}/mlflow/mlruns`
    ###### *for example, you should change the absolute path like: `file://d:/fishtail/projects/ft-vessel-utilization/mlflow/mlruns`*

    or run the following:
    * `>mlflow_run_local_server.sh`

    or run the following:
    * `>mlflow run -e server .` which runs the _mlflow_run_local_server.sh_ file.

2. Go to local [MLflow UI](http://127.0.0.1:5000)\
or you can launch MLflow UI in the **[./mlflow]** folder the following
command:
    * `>mlflow ui --backend-store-uri sqlite:///mlregistry.db`

3. You can check how MLflow Tracking Server works with `../checks/check_mlflow_local_server.py`


## # Models training and logging:
----------------------------------
1. Train models with the following modules:
    * `../model/enrich_shipdb_model.py` : logs models, params, metrics & artiacts.
    * `../model/vu_train_model_ml.py` : logs models, params, & metrics.


2. Monitor models and its parameters, performance and metadata with
local [MLflow UI](http://127.0.0.1:5000).


## # Serving model with MLflow Models (directly from ./mlruns folder):
----------------------------------
Pick up a best model from experiments and real-time
[serve model](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#serving-the-model) locally through a REST API.

1. Using [MLflow UI](http://127.0.0.1:5000)
with comparing the Models performance metrics pick the best model:\
<img src="https://mlflow.org/docs/latest/_images/tutorial-compare.png" width="1000">

2. To deploy (serve) the model locally, open in Integrated Terminal
the `/root_project_folder` and run:
    * `>mlflow models serve -m <Full_Path_to_your_model> -p 8080`\
where <Full_Path_to_your_model> can be drawn from the model Artifacts
in MLflow UI or using Python API,\
or may be your own custom folder on local machine.
<img src="https://mlflow.org/docs/latest/_images/oss_registry_1_register.png" width="1000">

        Deploying model on the port not equal `5000` make you possible use MLflow UI at the same time.\
It is also possible use other free localhost ports like `5555`.
It gives you opportunity to deploy several models at one time.\
Do not forget to change port in `mlflow_predict_local_deployed.py` !\
If you deploy model at first time a new Conda environment like
_mlflow-8b02f3a...._ will be created.\
If deploying was successfull, you will get terminal output like this:
        >Serving on [http://kubernetes.docker.internal:8080](http://kubernetes.docker.internal:1234)


## # Registering, Staging and Serving model with MLflow Registry:
----------------------------------
Take productioned model from Model Registry and [deploy model](https://www.mlflow.org/docs/latest/models.html#deploy-mlflow-models) locally as a REST endpoint to a server launched by MLflow CLI.

1. Go to local [MLflow UI](http://127.0.0.1:5000)

2. From the MLflow Runs detail page, select a logged MLflow Model in the Artifacts
section.

3. Click the Register Model button and register with Model Registry as `@projectname_model`

4. Transit the registred model into `Production`
<img src="https://mlflow.org/docs/latest/_images/oss_registry_5_transition.png" width="1000">

5. You can register any models you wish and transit it into `Staging`

6. To deploy the productioned model from Model Registry, Open in Integrated
Terminal the `./` and run:
    * `>mlflow_serve_local_model.sh`


## # Making predictions with deployed model:
----------------------------------
From another terminal opened in in **[./mlflow]**\
send a POST request with JSON-serialized pandas DataFrames in the 'split' orientation:
* `>python mlflow_predict_local_deployed.py`\
*Also you can run this module with VS Code Interactive Python Window.*

----------------------------------
----------------------------------
### # Stop deployed model:
1. To stop deployed model kill the Terminal in which it was deployed ( Ctrl+C ).
### # Deleting MLflow Registry (sqlite db file) and mlruns artifact folder:
1. Stop all deployed models.
2. Open in Integrated Terminal in the **[./mlflow]**:
    * `>mlflow_remove_local_db.sh`
