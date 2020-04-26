# cOViral-data


## Documentation <a name="documentation"></a>

Git repository for the data part of the **cOViral**'s projet for EUvsCovid Hackathon


## Repository Structure <a name="repository-structure"></a>

```
|_ __init__.py
|_ config.yaml
|_ main.py
|_ train.py
|_ requirements.txt
|_ donnees-hospitalieres-covid19.csv
|_ mapping_region_dep.csv
|_ model.pkl
```

- ```[requirements.txt](requirements.txt)``` Contains the list of all packages that need to be installed to make the code work well. <br/>
- ```[main.py](main.py)``` Contains the functions deployed as google cloud functions <br/>
- ```[train.py](train.py)``` Tools to retrain model to score risks of second wave of virus propagation <br/>
- ```[config.yaml](config.yaml)``` Configuration file <br/>
- ```[donnees-hospitalieres-covid19](donnees-hospitalieres-covid19.csv)``` Data on April 23rd, 2020 <br/>
- ```[mapping_region_dep](mapping_region_dep.csv)``` Mapping file between french regions and departments <br/>
- ```[model.pkl](model.pkl)``` Resulting model for predicting normal number of hospitalisation delta for a given a department <br/>


## Installation <a name="installation"></a>

First you need to install [gcloud CLI](https://cloud.google.com/sdk/docs)

```shell
gcloud functions deploy YOUR_FUNCTION --runtime python37 --trigger-http --allow-unauthenticated
```



## Configuration <a name="config"></a>

Parameters available :
- **window**: Number of days used for prediction of normal number of hospitalisation delta
- **day_proj**: Number of days to project the predicition
- **var**: Name of the variable to modelize (number of hospitalisation or delta)
- **model**: Path to dump the resulting model

