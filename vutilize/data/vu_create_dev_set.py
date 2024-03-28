"""
    LOAD enriched shipdb data: "containershipdb_042021_enriched.csv"
    LOAD voyages' data w/ factual draught: "combined_vt.csv"
    LOAD merged MRV report 2018-19: "merged_mrv.csv"
    LOAD ship's classes from raw data: "shipdb_export_04_2021.csv""
    MERGE all data on IMO
    CALCULATE auxilary variables
    CALCULATE Vessels' utilization will be used as label
    SAVE data in [./project/data/processed].

    @author: mikhail.galkin
"""

# %% Load libraries
import sys
import warnings

import pandas as pd
import sweetviz as sv

from pathlib import Path
from sklearn.ensemble import IsolationForest
from dataprep.eda import create_report
from IPython.display import display


# %% Load project's stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
import vutilize

# Load custom classes and utils
from vutilize.config import data_processed_dir
from vutilize.config import reports_dir
from vutilize.utils import set_pd_options
from vutilize.utils import set_matlotlib_params
from vutilize.utils import reset_pd_options
from vutilize.utils import df_get_glimpse
from vutilize.utils import get_outliers_zscore
from vutilize.utils import get_outliers_quantile
from vutilize.utils import ridoff_outliers


# %% Report functions -----------------------------------------------------------
def make_sweetviz_analyze(df, name, target="draught_fact"):
    print(f"\nSweetViz analysis report start...")
    print(f"\tSweetViz: {sv.__version__}")
    print(f"Config SweetViz...")
    sv.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sv.FeatureConfig(skip=None, force_text=None)
    report_name = f"{name}.html"
    report = sv.analyze(
        source=df,
        target_feat=target,
        feat_cfg=feature_config,
        pairwise_analysis="on",
    )
    report.show_html(reports_dir / report_name)


def make_dataprep_report(
    df,
    title,
    name,
):
    # aux_vars = [
    #     "draught_fact",
    #     "max_draught",
    #     "deadweight",
    #     "displacement",
    #     "lightweight",
    #     "fc_tn_ml",
    #     "fc_tn_hr",
    #     "voyage_ml",
    #     "voyage_hr",
    #     "voyage_fuel_tn",
    #     "voyage_oil_tn",
    #     "ballast_water",
    #     "freshwater",
    #     "constant_tn",
    #     "payload_estimated",
    #     "teu_estimated",
    #     "vu_estimated",
    # ]
    # check_vars = [
    #     "1_dft_ratio",
    #     "2_1*dwt",
    #     "3_lwt",
    #     "4_2-lwt",
    #     "5_4-voyage_fuel",
    #     "6_5-voyage_oil",
    #     "7_6-ballast_water",
    #     "8_7-freshwater",
    #     "9_8-constant",
    # ]
    # aux_vars = aux_vars + check_vars
    report_name = name
    report = create_report(df, title)
    report.save(filename=report_name, to=reports_dir)
    # report.show_browser()


# %% Data -----------------------------------------------------------------------
def load_static_csv(dir, file_to_load):
    print(f"\nLoad data:\n\tDir: {dir}\n\tFile: {file_to_load}")
    usecols_static = [
        "imo",
        "ais_dim_a",
        "ais_dim_b",
        "ais_dim_c",
        "ais_dim_d",
        "ballast",
        "ballast_water",
        "breadth_moulded",
        "breadth_registered",
        "deadweight",
        "depth",
        "depth_moulded",
        "diesel_oil",
        "displacement",
        "draught",
        "eng_total_bhp",
        "eng_total_kw",
        "engine_power",
        "freeboard",
        "freshwater",
        "fuel",
        "fuel_oil",
        "gas_oil",
        "gross_tonnage",
        "length_overall",
        "light_displacement_tonnes",
        "lube_oil",
        "main_engine_kw",
        "main_engine_rpm",
        "max_draught",
        "max_speed",
        "net_tonnage",
        "teu",
        "year_of_built",
    ]
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        usecols=usecols_static,
    )
    # Get column w/o NaNs
    # cols_stat = list(df_stat.columns[~df_stat.isna().any()])
    print(f"\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    return df


def load_voyages_csv(dir, file_to_load):
    print(f"\nLoad data:\n\tDir: {dir}\n\tFile: {file_to_load}")
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
    )
    print(f"\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    df["voyage_hr"] = df["duration"] // 3600
    # Drop redundant columns
    cols_to_drop = ["duration", "start", "stop"]
    df = df.drop(cols_to_drop, axis=1)
    df.rename({"distance": "voyage_ml", "draught": "draught_fact"}, axis=1, inplace=True)
    return df


def load_mrv_csv(dir, file_to_load):
    print(f"\nLoad data:\n\tDir: {dir}\n\tFile: {file_to_load}")
    new_names = [
        "imo",
        "year",
        "total_fuel_tn",
        "total_time_sea_hr",
        "total_co2_tn",
        "total_dist_ml",
    ]
    usecols_mrv = [
        "IMO",
        "Reporting Period",
        "Total Fuel",
        "Time at Sea",
        "Total CO2",
        "dist_miles",
    ]
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        usecols=usecols_mrv,
    )
    print(f"\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    df.rename(dict(zip(usecols_mrv, new_names)), axis=1, inplace=True)
    # Grouping by IMO
    df = df.groupby(by="imo", as_index=False)[
        "total_fuel_tn",
        "total_time_sea_hr",
        "total_co2_tn",
        "total_dist_ml",
    ].sum()
    df["fc_tn_ml"] = df["total_fuel_tn"] / df["total_dist_ml"]
    df["fc_tn_hr"] = df["total_fuel_tn"] / df["total_time_sea_hr"]
    df["speed_ml_hr"] = df["total_dist_ml"] / df["total_time_sea_hr"]
    df["co2_tn_ml"] = df["total_co2_tn"] / df["total_dist_ml"]
    df["co2_tn_hr"] = df["total_co2_tn"] / df["total_time_sea_hr"]
    return df


def load_size_class(
    dir=Path("d:/fishtailS3/ls-aishub-inflated/shipdb"),
    file_to_load="shipdb_export_04_2021.csv",
):
    print(f"\nLoad size classes for ships:\n\tDir: {dir}\n\tFile: {file_to_load}")
    usecols_static = ["imo", "my_vessel_type.0", "size_class"]
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        usecols=usecols_static,
    )
    df = df[df["my_vessel_type.0"] == "container_ship"]
    df = df.dropna(axis=0)
    df = df.drop_duplicates(ignore_index=True)
    df = df.drop(["my_vessel_type.0"], axis=1)
    print(f"\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    return df


# %% Merge ----------------------------------------------------------------------
def merge_data(df_voyages, df_static, df_mrv):
    print(f"\nMerge w/ voyages dataset...")
    df_merged = pd.merge(
        df_voyages,
        df_static,
        how="inner",
        on="imo",
        suffixes=("_fact", "_design"),
    )
    # Account the num of rows will been deleted
    num_na = df_merged[df_merged.isna().any(axis=1)].shape[0]
    print(
        f"""Merged Voyages & Static data shape: {df_merged.shape}
    it will be deleted: #{num_na} rows w/ NAs
    """
    )
    df_merged = df_merged.dropna()
    print(f"Merge w/ MRV dataset...")
    cols_mrv = [
        "imo",
        "fc_tn_ml",
        "fc_tn_hr",
        "speed_ml_hr",
        "co2_tn_ml",
        "co2_tn_hr",
    ]
    df = pd.merge(df_merged, df_mrv[cols_mrv], how="inner", on="imo")
    print(f"Merged Voyages & Static & MRV data shape: {df.shape}")
    return df


def merge_size(df, df_size):
    print(f"\nMerge w/ size classes...")
    df = pd.merge(
        df,
        df_size,
        how="left",
        on="imo",
    )
    print(f"Merged data shape: {df.shape}")
    return df


# %% Outliers -------------------------------------------------------------------
def del_outliers_isolation_forest(df):
    print(f"\nGet outliers with Isolation Forest...")
    cols_cat = ["from", "to"]
    # Identify outliers in the dataset
    iso = IsolationForest(
        n_estimators=1000,
        contamination="auto",
        bootstrap=True,
        n_jobs=-1,
        verbose=1,
        random_state=rnd_state,
    )
    df_fit = df[df.columns.difference(cols_cat)]
    yhat = iso.fit_predict(df_fit)
    # Get mask for all rows that are not outliers
    no_outliers_mask = yhat != -1
    df_wo_outl = df[no_outliers_mask]
    print(f"\tThere was found out {sum(yhat==-1)} outliers...")
    print(f"\tIncome data had {len(df)} rows X {len(df.columns)} cols...")
    print(f"\tOutcome data has {len(df_wo_outl)} rows X {len(df_wo_outl.columns)} cols...")
    return df_wo_outl


# %% Sanity check ---------------------------------------------------------------
def check_draught_truth(df, draught_check, draught_fact="draught_fact"):
    # Calculate difference in percents
    diff = (df[draught_check] - df[draught_fact]) / df[draught_check] * 100
    diff.plot(kind="hist", title=f"%Diff {draught_check} vs. {draught_fact}", bins=50)
    # diff[diff<0].plot(kind="hist", title=f"{draught_check} < {draught_fact}", bins=50)
    print(
        f""" Mistakes:
        \t {draught_check} < {draught_fact}:
        \t #Cases: {len(diff[diff<0])}
        \t Min: {min(diff[diff<0])}
        \t Mean: {diff[diff<0].mean()}
        \t %Mean: {(diff[diff<0]/df.loc[diff<0, draught_fact]).mean() * 100}
    """
    )
    diff[diff < 0].plot(kind="hist", title=f"{draught_check} < {draught_fact}", bins=50)
    df.loc[
        df[draught_check] < df[draught_fact],
        [draught_check, draught_fact],
    ].plot.scatter(x=draught_check, y=draught_fact, title=f"{draught_check} < {draught_fact}")

    print(
        f""" Truth:
        \t {draught_check} >= {draught_fact}:
        \t #Cases: {len(diff[diff>=0])}
        \t Min: {min(diff[diff>=0])}
        \t Mean: {diff[diff>=0].mean()}
        \t %Mean: {(diff[diff>=0]/df.loc[diff>=0, draught_fact]).mean() * 100}
    """
    )
    df.loc[
        df[draught_check] >= df[draught_fact],
        [draught_check, draught_fact],
    ].plot.scatter(x=draught_check, y=draught_fact, title=f"{draught_check} >= {draught_fact}")


def make_sanity_cleanoff(df):
    """DWR = Deadweight/Displacement
    Typical values of DWR for a range of ship types are as follow:
        –Reefer 0.58-0.60
        –General Cargo 0.62-0.72
        –Ore Carrier 0.72-0.77
        –Bulk Carrier 0.78-0.84
        –Tanker 0.80-0.86
    In our RAW min(DWR)=0.5113, Q5%=0.6770, max(DWR)=0.7877
    """
    df = df[(df["voyage_ml"] > 10) & (df["voyage_ml"] < 70000)]
    df = df[(df["voyage_hr"] > 1) & (df["voyage_hr"] < 9000)]
    df = df[df["max_draught"] >= df["draught_fact"]]
    df = df[(df["displacement"] * 0.62) < df["deadweight"]]
    # df["dwr"] = df["deadweight"] / df["displacement"]
    return df


# %% Columns --------------------------------------------------------------------
def setup_size_class(df):
    print(f"\nSetup Size Classes...")

    def setup_feeder(y):
        if y <= 1000:
            return "Feeder 1000"
        elif y < 2000:
            return "Feeder 1999"
        elif y < 4000:
            return "Feeder 3999"
        else:
            return "Feeder 4000+"

    def setup_size(y):
        if y < 10000:
            return "TEU .9999"
        elif y < 15000:
            return "TEU 14999"
        elif y < 20000:
            return "TEU 19999"
        else:
            return "TEU 20000+"

    def setup_class(x, y):
        if x == "Small Feeder":
            return setup_feeder(y)
        elif x in ["Feedermax", "Feeder maximum 3999"]:
            return "Feeder 3999"
        elif x == "Feeder medium 1999":
            return "Feeder 1999"
        elif x == "Feeder small 1000":
            return "Feeder 1000"
        elif x == "Feeder":
            return setup_feeder(y)

        elif x == "a Below 1000 TEU":
            return "Feeder 1000"
        elif x in ["b 1000..1999 TEU", "1000T", "1100T", "1700T"]:
            return "Feeder 1999"
        elif x in ["c 2000..3999 TEU", "3500T", "3600T"]:
            return "Feeder 3999"
        elif x == "Handymax MR2":
            return setup_feeder(y)

        elif x == "Panamax":
            if y < 6000:
                return "Panamax .5999"
            elif y < 8000:
                return "Panamax .7999"
            elif y < 10000:
                return "Panamax .9999"
        elif x == "Panamax 9999":
            if y < 10000:
                return "Panamax .9999"
            else:
                return "Panamax 14999"

        elif x in [
            "Post Panamax",
            "Midsize post Panamax 11999",
            "Midsize post Panamax 13999",
            "Large post Panamax 15999",
        ]:
            if y < 10000:
                return "Post Panamax .9999"
            elif y < 14000:
                return "Post Panamax 13999"
            elif y < 16000:
                return "Post Panamax 15999"
            else:
                return "Post Panamax 16000+"

        elif x in [
            "13100T",
            "13400T",
            "13800T",
            "14000T",
            "18300T",
            "4300T",
            "5100T",
            "5400T",
            "6500T",
            "6600T",
            "6700T",
            "8000T",
            "8600T",
            "8800T",
            "9000T",
            "9200T",
            "9700T",
            "d 4000..5999 TEU",
            "e 6000..7999 TEU",
            "f 8000..9999 TEU",
            "g 10000..11999 TEU",
            "h 12000..13999 TEU",
            "i 14000..15999 TEU",
            "j 16000..17999 TEU",
            "k 18000..19999 TEU",
        ]:
            return setup_size(y)

        elif pd.isnull(x):
            return setup_size(y)
        else:
            return x

    df["size_class"] = df.apply(lambda x: setup_class(x["size_class"], x["teu"]), axis=1)
    display(df.groupby(by="size_class").size())
    return df


def calc_aux_vars(df):
    container_density = 14
    crew = 20
    df["lightweight"] = df["displacement"] - df["deadweight"]
    # +5% as reserve
    df["voyage_fuel_tn"] = df["fc_tn_ml"] * df["voyage_ml"]
    # experts estimations 5-6% from fuel's volume
    df["voyage_oil_tn"] = df["voyage_fuel_tn"] * 0.05
    # expert estimations 145 gallons per day per person
    # 1 US gallon of water = 3.79 kilograms.
    df["voyage_freshwater_tn"] = (crew * 3.8 * (df["voyage_hr"] // 24 + 1)) / 1000
    # 20 persons per 100kg + 50kg stuff for each + 10kg of provision per 1 day
    df["constant_tn"] = crew * (100 + 50 + 10 * (df["voyage_hr"] // 24 + 1)) / 1000
    # draught ratio
    df["draught_ratio"] = df.apply(
        lambda x: 1 if x.draught_fact <= 0 else (x.draught_fact / x.max_draught),
        axis=1,
    )
    # Apply Draught Survey formula
    df["payload_estimated"] = (
        (df["draught_ratio"] * df["deadweight"])
        - df["lightweight"]
        - df["voyage_fuel_tn"]
        - df["voyage_oil_tn"]
        - df["voyage_freshwater_tn"]
        - df["constant_tn"]
    )
    df["teu_estimated"] = df["payload_estimated"] / container_density
    df["vu_estimated"] = df["teu_estimated"] / df["teu"]
    return df


def calc_check_vars(df):
    df["1_dft_ratio"] = df["draught_fact"] / df["max_draught"]
    df["2_1*dwt"] = df["1_dft_ratio"] * df["deadweight"]
    df["3_lwt"] = df["displacement"] - df["deadweight"]
    df["4_2-lwt"] = df["2_1*dwt"] - df["3_lwt"]
    df["5_4-voyage_fuel"] = df["4_2-lwt"] - df["voyage_fuel_tn"]
    df["6_5-voyage_oil"] = df["5_4-voyage_fuel"] - df["voyage_oil_tn"]
    df["7_6-ballast_water"] = df["6_5-voyage_oil"] - df["ballast_water"]
    df["8_7-freshwater"] = df["7_6-ballast_water"] - df["freshwater"]
    df["9_8-constant"] = df["8_7-freshwater"] - df["constant_tn"]
    return df


def score_od_pairs(df):
    df["o_d"] = df["from"] + "-" + df["to"]
    df["vu_estimated_od_avg"] = df.groupby(by="o_d")["vu_estimated"].transform("mean")
    df["co2_tn_ml_od_avg"] = df.groupby(by="o_d")["co2_tn_ml"].transform("mean")
    return df


def cols_reorder(df):
    cols = df.columns.drop("imo").tolist()
    cols.sort()
    cols = ["imo"] + cols
    df = df[cols]
    return df


# %% Main =======================================================================
def main(
    file_static_data,
    file_voyages_data,
    file_mrv_data,
    dir_voyages_data,
    dir_mrv_data,
    file_to_save,
):
    """Creates development data set from enhriched containershipdb,\
        combined vessel tracker w/ distance,\
            merged (2018-2019) MRV\
                files.
    Args:
        file_static_data (str): name for csv enhriched containershipdb data file
        file_voyages_data (str): name for csv vessels tracker data file
        file_mrv_data (str): name for csv merged MRV data file
        dir_voyages_data (str): Absolute path to 'file_voyages_data'
        dir_mrv_data (str): Absolute path to 'file_mrv_data'
        file_to_save (str]): name for csv dev data file
    """
    print(f"---------- START: Create Development set --------------------------")
    set_pd_options()
    set_matlotlib_params()
    # * Data
    dir_voyages_data = Path(dir_voyages_data)
    dir_mrv_data = Path(dir_mrv_data)
    df_static = load_static_csv(dir=data_processed_dir, file_to_load=file_static_data)
    df_voyages = load_voyages_csv(dir=dir_voyages_data, file_to_load=file_voyages_data)
    df_mrv = load_mrv_csv(dir=dir_mrv_data, file_to_load=file_mrv_data)
    df = merge_data(df_voyages, df_static, df_mrv)
    df_size = load_size_class()
    df = merge_size(df, df_size)
    df = setup_size_class(df)
    df = make_sanity_cleanoff(df)
    df = calc_aux_vars(df)
    # * Cut off the tails
    df = df[(df["vu_estimated"] >= 0) & (df["vu_estimated"] <= 1)]
    df = df[df["vu_estimated"] > df["vu_estimated"].quantile(0.01)]
    df = score_od_pairs(df)
    df = cols_reorder(df)
    # df_get_glimpse(df)
    # df.groupby(by="size_class").agg({"vu_estimated": ["mean", "median", "min", "max"]})
    make_dataprep_report(
        df[df.columns.drop("imo")],
        title="ContainerShips VU development dataset",
        name="containerships_vu_devset-dataprep",
    )
    make_sweetviz_analyze(
        df[df.columns.drop("imo")],
        name="containerships_vu_devset-analyze",
        target="vu_estimated",
        # target="co2_tn_ml",
    )
    df.to_csv(data_processed_dir / file_to_save, header=True, index=False)
    reset_pd_options()
    print(f"!!! DONE !!!")


# %% RUN ========================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    rnd_state = 42
    main(
        file_static_data="containershipdb_042021_enriched.csv",
        file_voyages_data="combined_vt.csv",
        file_mrv_data="merged_mrv.csv",
        dir_voyages_data="d:/fishtailS3/ls-aishub-inflated/voyages_vt_p1_wd",
        dir_mrv_data="d:/fishtailS3/ls-aishub-inflated/eu_mrv_data",
        file_to_save="containershipdb_vu_devset.csv",
    )

# %% Aux cell
# file_static_data = "containershipdb_042021_enriched.csv"
# dir_voyages_data = Path("d:/fishtailS3/ls-aishub-inflated/_voyages_vt_p1_wd")
# file_voyages_data = "combined_vt.csv"
# dir_mrv_data = Path("d:/fishtailS3/ls-aishub-inflated/eu_mrv_data")
# file_mrv_data = "merged_mrv.csv"
# file_to_save = "containershipdb_vu_devset.csv"
# %%
# ^ TO DO:
# ^ [X] turn the "freshwater" to the formula from duration
# ^ [] turn the number of crew to the formula from deadweiht
