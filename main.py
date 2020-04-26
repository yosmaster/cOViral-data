from flask import escape, jsonify, make_response, Response
from datetime import datetime
import train
import pickle
import yaml
import pandas as pd
import json
import os
import urllib.request
import numpy as np

print("reading yaml")
config = yaml.safe_load(open("./config.yml", "r"))
print("config loaded")

print("Let's start")
last_date = datetime.now().date()
print("loading model")
model = pickle.load(open(config["model"], 'rb'))
print("getting data")
df = train.get_data()
print("init Ok")
result = None

mapping_reg_dep = pd.read_csv("mapping_region_dep.csv", dtype={"region": str, "dep": str})


def get_risks(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    print("lets update globals")
    update_globals()
    global df
    # if os.path.exists("result.json"):
    #    print("result exisits")
    #    with open("result.json", "r") as f:
    #        result = json.load(f)
    #    return jsonify(result)
    print('lets predict')
    res = get_predict(df)
    # with open("result.json", "w") as f:
    #    json.dump(res, f)
    return res


def get_individual_score(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'age' in request_json:
        age = str(request_json['age'])
        dep = str(request_json['dep']).zfill(2)
        sex = int(request_json['sex'])
        pav = str(request_json['pav'])
    elif request_args and 'age' in request_args:
        age = str(request_args['age'])
        dep = str(request_args['dep']).zfill(2)
        sex = int(request_args['sex'])
        pav = str(request_args['pav'])
    try:
        region = mapping_reg_dep[mapping_reg_dep.dep == dep]["region"].values[0]
    except:
        print("Region for dep {} was not found".format(dep))
        region = "Unknown"
    print({"age": age, "dep": dep, "sex": sex, "pav": pav, "region": region})


    try:
        train.preprocess_data()
        print("Preprocess data OK")
    except Exception as e:
        print(e)

    resp = make_response(jsonify({"score": 80}))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def download_json(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = str(request_json['name'])
    elif request_args and 'name' in request_args:
        name = str(request_args['name'])
    path = "https://storage.googleapis.com/coviral_bucket/donnees_maille_departement/{}".format(name)
    j = urllib.request.urlopen(path).read().decode("utf-8")
    resp = make_response(jsonify(j))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def update_globals():
    global df
    global model
    global last_date
    global result
    if datetime.now().date() != last_date:
        last_date = datetime.now().date()
        df = train.update_model()
        model = pickle.load(open(config["model"], 'rb'))


def get_predict(df):
    def get_vector(d):
        return pd.Series(d.sort_values("jour")["diff"].values[-config["window"] - 1:-1])

    def get_last(d):
        return d.sort_values("jour")["diff"].values[-1]

    def get_score(x):
        if x < 0:
            return int(-round(x / .3 * 20)) + 10
        elif x < 0.045:
            return int(round(20 + 60 * (0.045 - x) / 0.045))
        else:
            return int(round(80 + 20 * (0.5 + x)))

    print("0")
    real = df.dropna().groupby("dep")[["jour", "diff"]].apply(get_last)
    print("1")
    vectors = df.dropna().groupby("dep")[["jour", "diff"]].apply(get_vector)
    print("2")
    res = model.predict(vectors)
    print("3")
    vectors["predict"] = res
    print("4")
    vectors["real"] = real
    print("5")
    vectors["diff"] = vectors["real"] - vectors["predict"]
    print("6")
    vectors["max"] = df.groupby("dep")["hosp"].max()
    print("7")
    vectors["diff_norm"] = vectors["diff"] / vectors["max"]
    print("8")
    vectors["risk"] = vectors["diff_norm"].apply(get_score)
    print("9")
    resp = make_response(jsonify(vectors["risk"].to_dict()))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
