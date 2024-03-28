"""
    LOAD raw data: "shipdb_export_04_2021.csv"
    and GIVE us a first glimpse
    and MAKE Exploratory Data Analysis w\o any changing original data
    and SAVE results of EDA to the [./project/reports].

    @author: mikhail.galkin
"""

# %% Load libraries
import sys
import winsound
import pandas as pd
import pandas_profiling as pp
import sweetviz as sv

from pathlib import Path
from IPython.display import display


# %% Load project stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
dir = Path("d:/fishtailS3/ls-aishub-inflated/shipdb")
# dir = Path("d:/fishtail/projects/ft-vessel-utilization/data/processed")

# Load project configu
from vutilize.config import reports_dir
from vutilize.config import pandas_profiling_dir
from vutilize.config import shipdb_cols_to_drop

# Load project utils
from vutilize.utils import set_pd_options
from vutilize.utils import reset_pd_options
from vutilize.utils import cols_reorder
from vutilize.utils import cols_get_na
from vutilize.utils import cols_get_mixed_dtypes
from vutilize.utils import cols_coerce_mixed_to_num
from vutilize.utils import df_get_glimpse


# %% LOCAL: First testing loading -----------------------------------------------
def load_test(file="shipdb_export_04_2021.csv"):
    """Pre-loading testing of load"""
    ## num of records in file (excludes header)
    n_len = sum(1 for line in open(dir / file)) - 1
    print(f"Original file {file} contain {n_len} rows...")

    df = pd.read_csv(
        dir / file,
        nrows=None,
        # index_col=0,
    )
    print(f"{file}: \n\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    display(df.sample(n=10, random_state=42).T)


# %% LOCAL: Custom functions ----------------------------------------------------
def print_versions():
    print(f" ")
    print(f"Pandas: {pd.__version__}")
    print(f"Pandas_Profiling: {pp.__version__}")
    print(f"SweetViz: {sv.__version__}")


def load_raw_data(file_to_load, nrows=None):
    """Load raw dataset from file
    Args:
        dir (pathlib.WindowsPath): Main project's directory
        file_to_load (str, optional): File name to load.
    Returns:
        Pandas DF: [description]
    """
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        nrows=nrows,
        # low_memory=False,
    )
    print(f"\n{file_to_load}: Loaded {len(df)} rows X {len(df.columns)} cols")
    return df


def get_container_ships(df):
    print(f"\nKeeps only data for container ships...")
    df = df.loc[df["my_vessel_type.0"] == "container_ship"]
    df = df.dropna(axis=1, how="all")
    print(f"Dataset has {len(df)} rows X {len(df.columns)} cols")
    return df


def save_info_to_xls(df, file_name="info.xlsx"):
    print(f"\nSave data info to xlsx...")
    file = reports_dir / file_name
    df_na = cols_get_na(df)
    pd.DataFrame(df.dtypes, columns=["dtypes"]).join(df_na).to_excel(file)


def cols_coerce_str_to_float(df, col="mcr_fuel_consumption"):
    print(f"\nCoerce string to float...")
    df[col] = df[col].str.extract(r"(\d+.\d+)").astype("float")
    return df


def print_random_entities(df, nrows=6):
    # View some random selected records
    print(f"\nRandom {nrows} rows:")
    display(df.sample(n=nrows).T)


# %% Profiling data and save report ot HTML
# ------------------------------------------------------------------------------
# Profiling Exploratary Data Analysis
# ------------------------------------------------------------------------------
def make_pandas_profiling_report(
    df,
    config="optimal",
    report_name=None,
    to_file=True,
):
    print(f"\nPandas profiling report start...")
    # You can choose config between: "config_default." \"config_minimal." \"config_optimal."
    config_file = pandas_profiling_dir / f"{config}.yaml"
    report_name = f"{report_name}_profile.html"
    # Make: Pandas Profile report
    pp_report = pp.ProfileReport(df, config_file=config_file)
    if to_file:
        pp_report.to_file(reports_dir / report_name)
    return pp_report


# %% Change config for SweetViz report
# ------------------------------------------------------------------------------
# Sweetviz Exploratary Data Analysis
# ------------------------------------------------------------------------------
def config_sweetviz():
    """feat_cfg:
    A FeatureConfig object representing features to be skipped,
    or to be forced a certain type in the analysis.
    The arguments can either be a single string or list of strings.
    Parameters are skip, force_cat, force_num and force_text.
    The "force_" arguments override the built-in type detection.
    They can be constructed as follows:
        feature_config = sv.FeatureConfig(skip="PassengerId", force_text=["Age"])
    """
    print(f"\nConfig SweetViz...")
    sv.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sv.FeatureConfig(skip=None, force_text=None)
    return feature_config


def make_sweetviz_analyze(
    df,
    report_name=None,
    target_feat=None,
):
    """The analyze() function can take multiple other arguments:
    analyze(source: Union[pd.DataFrame, Tuple[pd.DataFrame, str]],
        target_feat: str = None,
        feat_cfg: FeatureConfig = None,
        pairwise_analysis: str = 'auto')
    """
    print(f"\nSweetViz analysis report start...")
    report_name = f"{report_name}_analyze.html"
    feature_config = config_sweetviz()
    report = sv.analyze(
        source=df,
        target_feat=target_feat,
        feat_cfg=feature_config,
        pairwise_analysis="on",
    )
    report.show_html(reports_dir / report_name)
    return report


# %% Main function for main.py ==================================================
def main(
    file_to_load,
    report_name,
    pandas_profile_config="optimal_shipdb",
    perfom_eda=True,
    pandas_profiling=True,
    sweetviz_analyze=True,
):
    """Performs a Exploratary Data Analisys with Pnadas Profiling and SweetViz
    packages.

    Args:
        * perfom_eda (bool, optional): Toggle to perform a EDA.\
            Defaults to True.
        * pandas_profiling (bool, optional): Toggle to create Pandas_profiling report.\
            Defaults to True
        * sweetviz_analyze (bool, optional): Toggle to create Sweetviz report.\
            Defaults to True
    """
    if perfom_eda:
        cols_to_drop = shipdb_cols_to_drop
        cols_to_num = ["cargo_holds", "crude_capacity", "engine_power"]
        print(f"-------------- START: Exploratary Data Analisys --------------")
        print_versions()
        set_pd_options()
        df = load_raw_data(file_to_load=file_to_load)
        df = get_container_ships(df)
        df = df.drop(cols_to_drop, axis=1)
        df = cols_reorder(df)
        # * Tackle w/ mixed types
        _ = cols_get_mixed_dtypes(df)
        df = cols_coerce_mixed_to_num(df, cols_to_num)
        df = cols_coerce_str_to_float(df)
        df_get_glimpse(df)

        if pandas_profiling:
            make_pandas_profiling_report(
                df,
                config=pandas_profile_config,
                report_name=report_name,
                to_file=True,
            )

        if sweetviz_analyze:
            df["teu"] = df["teu"].fillna(-1)
            make_sweetviz_analyze(
                df,
                target_feat="teu",
                report_name=report_name,
            )

        reset_pd_options()
        print(f"!!! DONE: Exploratory data analysis !!!")
        winsound.Beep(frequency=3000, duration=300)
    else:
        print(f"-------------- SKIP: Exploratary Data Analisys --------------")


# %% Workflow ===================================================================
if __name__ == "__main__":
    main(
        file_to_load="shipdb_export_04_2021.csv",
        report_name="containerships_042021_raw",
    )

# %%
