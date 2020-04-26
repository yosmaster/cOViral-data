import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta, datetime
import urllib.request
import yaml
import pickle
import io
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

pd.set_option("max_columns", 300)
pd.set_option("expand_frame_repr", False)

config = yaml.safe_load(open("./config.yml", "r"))


def prepare_data(df, win=15, day_proj=3, validation=2, var="hosp"):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    d = df.sort_values("jour")
    # time-window
    start = d.jour.min()
    end = d.jour.max()
    window = timedelta(days=win)
    while start + window + timedelta(days=day_proj) <= end:
        s = start
        e = start + window
        train_dep = pd.Series(d.dep.unique())
        for c in train_dep:
            if e <= end - timedelta(days=validation + day_proj):
                X_train.append(d[(d.dep == c) & (d.jour.between(s, e))][var].values)
                y_train.append(d[(d.dep == c) & (d.jour == e + timedelta(days=day_proj))][var].values)
            # for c in test_dep:
            else:
                X_test.append(d[(d.dep == c) & (d.jour.between(s, e))][var].values)
                y_test.append(d[(d.dep == c) & (d.jour == e + timedelta(days=day_proj))][var].values)
        start += timedelta(days=1)
    train = pd.DataFrame(X_train)
    train["target"] = pd.DataFrame(y_train)[0]
    test = pd.DataFrame(X_test)
    test["target"] = pd.DataFrame(y_test)[0]
    return train.dropna(), test.dropna()


def train_model(X_train, y_train, n_estimators=200):
    regr = RandomForestRegressor(n_estimators=n_estimators)
    regr.fit(X_train, y_train)
    return regr


def perf_model(regr, X_test, y_test):
    res = regr.predict(X_test)
    rmse = mean_squared_error(res, y_test)
    rmse_easy = mean_squared_error(X_test[X_test.columns[-1]], y_test)
    print(rmse, rmse_easy)
    (res - y_test).hist(bins=100)
    (res - X_test[X_test.columns[-1]]).hist(bins=100)


def shift(s):
    return s - s.shift(1)


def read_csv(path):
    text = urllib.request.urlopen(path).read()
    df = pd.read_csv(io.StringIO(text.decode("utf-8").replace('"', "")), sep=";")
    return df


def download_data():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = "https://storage.googleapis.com/coviral_bucket/donnees_hospitalieres/donnees-hospitalieres-covid19-{}-19h00.csv".format(
        yesterday)
    try:
        # text = urllib.request.urlopen(path).read()
        df = pd.read_csv(path, sep=";")  # io.StringIO(text.decode("utf-8").replace('"', "")), sep=";")
        return df
    except:
        print("No new file found for day {}".format(yesterday))
        return None


def get_data():
    d = download_data()
    if d is None:
        d = pd.read_csv("./donnees-hospitalieres-covid19.csv", sep=";")
    d.jour = pd.to_datetime(d.jour)
    df = d[d.sexe == 0][["dep", "jour", "hosp"]]
    df.hosp = df.hosp.astype(float)
    df["diff"] = df.dropna().groupby("dep")["hosp"].transform(shift)
    df["diff_norm"] = df["diff"] / df.dropna().groupby("dep")["hosp"].transform(max)
    return df


def update_model():
    df = get_data()
    train, test = prepare_data(df, win=config["window"], day_proj=config["day_proj"], validation=3, var=config["var"])
    regr = train_model(train.drop("target", axis=1), train["target"], n_estimators=200)
    perf_model(regr, test.drop("target", axis=1), test["target"])
    filename = config["model"]
    pickle.dump(regr, open(filename, 'wb'))
    return df


def get_feature_series(all_data, dateOfRun):
    date = datetime.strptime(dateOfRun, "%Y-%m-%d")
    dateOfRun_minus_one = date - timedelta(days=1)

    all_data["jour"] = pd.to_datetime(all_data['jour'], format='%Y-%m-%d')
    train = all_data[all_data.jour <= dateOfRun_minus_one].query("sexe != 'All'")
    y_train = all_data[all_data.jour == dateOfRun_minus_one].query("sexe != 'All'")
    y_train['y'] = y_train["hosp"] / y_train["dpt_pop"]

    y_train["id"] = y_train.apply(lambda r: r["sexe"] + "," + r["dpt"], axis=1)
    y_train = y_train.drop(["sexe", "dpt"], axis=1)
    y = y_train.set_index(["id"]).loc[:, "y"]

    timeseries = train.loc[:, ["dpt", "sexe", "jour", "rea", "rea_day", "rad", "rad_day", "dc", "dc_day"]].fillna(0)
    timeseries["id"] = timeseries.apply(lambda r: r["sexe"] + "," + r["dpt"], axis=1)
    timeseries = timeseries.drop(["dpt", "sexe"], axis=1)

    extracted_features = extract_features(timeseries, column_id="id", column_sort="jour")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)

    features_filtered = features_filtered.reset_index()
    features_filtered["sexe"] = features_filtered["id"].apply(lambda x: x.split(',')[0])
    features_filtered["dpt"] = features_filtered["id"].apply(lambda x: x.split(',')[1])
    features_filtered.drop(["id"], axis=1)
    features_filtered["jour"] = dateOfRun
    features_filtered["jour"] = pd.to_datetime(features_filtered['jour'], format='%Y-%m-%d')

    all_data = pd.merge(all_data[all_data.jour == dateOfRun].query("sexe != 'All'"), features_filtered,
                        on=["dpt", "sexe", "jour"])

    return (all_data)


def get_category_ages(cat_file, bins=v_bins, names=v_names):
    df_age = pd.read_csv(cat_file, sep=";")
    df_age = df_age.query('cl_age90 != 0')
    df_age["category"] = pd.cut(df_age.cl_age90, bins=bins, labels=names)
    df_age = df_age.groupby(['reg', "jour", "category"])['hosp', 'rea', 'rad', 'dc'].agg(sum).reset_index().rename(
        columns={"hosp": "reg_hosp", "rea": "reg_rea", "rad": "reg_rad", "dc": "reg_dc"})
    return df_age


def get_pv_catego(filelog):
    logements = pd.read_csv(filelog, sep=";")
    cols = ["codeReg", "codeDep"] + [col for col in logements.columns if
                                     ("popMonopAppart" in col) | ("popMen4pAppart" in col)]
    df = logements.loc[:, cols]
    df["1P-2P"] = df["popMonopAppart12p"] + df["popMen4pAppart12p"]
    df["3P-4P"] = df["popMonopAppart34p"] + df["popMen4pAppart34p"]
    df[">5P"] = df["popMonopAppart5pP"] + df["popMen4pAppart5pP"]
    df["totPopulation"] = df["popMonopAppart"] + df["popMen4pAppart"]
    df["1P-2P"] = df["1P-2P"] / df["totPopulation"]
    df["3P-4P"] = df["3P-4P"] / df["totPopulation"]
    df[">5P"] = df[">5P"] / df["totPopulation"]
    df = pd.melt(df.query("codeReg not in ['FRA', 'DOM', 'MET'] "), id_vars=['codeReg', 'codeDep'],
                 value_vars=['1P-2P', '3P-4P', '>5P'], value_name="pop", var_name="piece")
    return df


def preprocess_data():
    dfs = []
    root_path = "https://storage.googleapis.com/coviral_bucket"
    sources = root_path+"/Enrichissement_donnees_Covid-19_Sante_Publique_France_25_avril_2020/{}"
    for f in ["source_01.csv", "source_02.csv", "source_03.csv"]:
        dfs.append(pd.read_csv(sources.format(f)))
    # df_covid = pd.concat(dfs, axis=1)

    all_data = dfs[0][
        ["dpt", "dpt_pop_0_19", "dpt_pop_20_39", "dpt_pop_40_59", "dpt_pop_60_74", "dpt_pop_75_plus", "code_region",
         "latitude", "longitude", "dpt_pop", "reg_pop", "Superficie", "jour",
         "sexe", "hosp", "hosp_day", "rea", "rea_day", "rad", "rad_day", "dc", "dc_day"]]

    df_with_ts = get_feature_series(all_data,"2020-04-23")

    cat_age_file = root_path+"/donnees_hospitalieres/donnees-hospitalieres-classe-age-covid19-2020-04-23-19h00.csv"
    v_bins = [0, 19, 39, 59, 79, 99]
    v_names = ['[9-19[', '[19-39[', '[39-59[', '[59-79[', '>79']

    cat_ages = get_category_ages(cat_age_file)

    df_with_ages = pd.merge(dfs[0].query("sexe != 'All'"), cat_ages, left_on=["code_region", "jour"],
                            right_on=["reg", "jour"], how="inner")

    filelog = root_path+"/INSEE_conditions_menages/data_confinement_logements.csv"

    pv_cat = get_pv_catego(filelog)

    df_with_ages = pd.merge(df_with_ages, pv_cat, left_on=["dpt"], right_on=["codeDep"], how="inner")
    return df_with_ages


