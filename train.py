import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta, datetime
import urllib.request
import yaml
import pickle
import io


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


def download_data():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = "https://storage.googleapis.com/coviral_bucket/donnees_hospitalieres/donnees-hospitalieres-covid19-{}-19h00.csv".format(
        yesterday)
    try:
        text = urllib.request.urlopen(path).read()
        df = pd.read_csv(io.StringIO(text.decode("utf-8").replace('"', "")), sep=";")
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







