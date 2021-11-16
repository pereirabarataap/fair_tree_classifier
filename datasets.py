import openml
import numpy as np
import pandas as pd
from copy import deepcopy as copy
openml.config.apikey = "4ef25cfe971a3731fddbe4fb2f6d1d98"

def get_bank_age(show=True):
    #bank-marketing dataset
    did = 1461
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    attribute_names = np.array(range(X.shape[1])).astype(str)
    X.columns = attribute_names
    df = pd.concat((copy(X), copy(y)), axis=1)
    df = df.dropna()
    columns = df.columns.tolist()
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    y = (df["Class"]==pd.value_counts(y).keys()[0]).values.astype(int)
    s = (df["0"]>=65).values.astype(int)
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_adult_race(show=True):
    # adult
    did = 179
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    attribute_names = np.array(range(X.shape[1])).astype(str)
    X.columns = attribute_names
    df = pd.concat((copy(X), copy(y)), axis=1)
    df = df.dropna()
    df.columns = attribute_names.tolist() + ["class"]
    columns = df.columns.tolist()
    columns.remove("8")
    columns.remove("9")
    columns = ["8"] + columns
    df = df[columns]
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    s = (df["8"]=="White").values.astype(int).astype(int)
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    y = (df["class"]==pd.value_counts(y).keys()[0]).values.astype(int)
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_adult_gender(show=True):
    # adult
    did = 179
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    attribute_names = np.array(range(X.shape[1])).astype(str)
    X.columns = attribute_names
    df = pd.concat((copy(X), copy(y)), axis=1)
    df = df.dropna()
    df.columns = attribute_names.tolist() + ["class"]
    columns = df.columns.tolist()
    columns.remove("8")
    columns.remove("9")
    columns = ["9"] + columns
    df = df[columns]
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    s = (df["9"]=="Male").values.astype(int).astype(int)
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    y = (df["class"]==pd.value_counts(y).keys()[0]).values.astype(int)
    
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_adult_multiple_1(show=True):
    did = 179
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    attribute_names = np.array(range(X.shape[1])).astype(str)
    X.columns = attribute_names
    df = pd.concat((copy(X), copy(y)), axis=1)
    df = df.dropna()
    df.columns = attribute_names.tolist() + ["class"]
    columns = df.columns.tolist()
    columns.remove("8")
    columns.remove("9")
    columns = ["8", "9"] + columns
    df = df[columns]
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)

    s = np.concatenate(
        (
            (df["8"]=="White").values.astype(int).astype(int).reshape(-1,1),
            (df["9"]=="Male").values.astype(int).astype(int).reshape(-1,1)
        ),
        axis=1
    )
    features = columns[2:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    y = (df["class"]==pd.value_counts(y).keys()[0]).values.astype(int)
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        for s_column in range(s.shape[1]):
            print("s dist "+str(s_column)+":", pd.value_counts(s[:,s_column]).values.ravel()/len(s))
            
    return X_dummy, y, s

def get_adult_multiple_2(show=True):
    did = 179
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    categorical_indicator = np.array(categorical_indicator)
    attribute_names = np.array(range(X.shape[1])).astype(str)
    X.columns = attribute_names
    df = pd.concat((copy(X), copy(y)), axis=1)
    df = df.dropna()
    df.columns = attribute_names.tolist() + ["class"]
    columns = df.columns.tolist()
    columns.remove("8")
    columns.remove("9")
    columns = ["8", "9"] + columns
    df = df[columns]
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)

    s = []
    s_race = (df["8"]=="White").values.astype(int).astype(str).tolist()
    s_gender = (df["9"]=="Male").values.astype(int).astype(str).tolist()
    for i in range(len(df)):
        row = s_race[i] + s_gender[i]
        s.append(row)
    s = pd.get_dummies(s).values.astype(int) #order: BF, BM, WF, WM 
    features = columns[2:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    y = (df["class"]==pd.value_counts(df["class"]).keys()[0]).values.astype(int)
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        for s_column in range(s.shape[1]):
            print("s dist "+str(s_column)+":", pd.value_counts(s[:,s_column]).values.ravel()/len(s))

    return X_dummy, y, s

def get_recidivism_age(show=True):
    df = pd.read_csv("compas.csv")
    df = df.loc[df["race"].isin(["Caucasian", "African-American"])]
    df = df[
        [
            "age", 
            "decile_score", "priors_count", "c_charge_degree",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "two_year_recid",
        ]
    ]
    columns = df.columns.tolist()
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    
    df = df[columns].copy()
    df = df.dropna()
    y = (df["two_year_recid"]==1).values.astype(int)
    s = (df["age"]<=24).values.astype(int).astype(int)
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_recidivism_race(show=True):
    df = pd.read_csv("compas.csv")
    df = df.loc[df["race"].isin(["Caucasian", "African-American"])]
    df = df[
        [
            "race",
            "decile_score", "priors_count", "c_charge_degree",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "two_year_recid",
        ]
    ]
    columns = df.columns.tolist()
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    
    df = df[columns].copy()
    df = df.dropna()
    y = (df["two_year_recid"]==1).values.astype(int)
    s = (df["race"]=="Caucasian").values.astype(int).astype(int)
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_recidivism_gender(show=True):
    df = pd.read_csv("compas.csv")
    df = df.loc[df["race"].isin(["Caucasian", "African-American"])]
    df = df[
        [
            "sex",
            "decile_score", "priors_count", "c_charge_degree",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "two_year_recid",
        ]
    ]
    columns = df.columns.tolist()
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    
    df = df[columns].copy()
    df = df.dropna()
    y = (df["two_year_recid"]==1).values.astype(int)
    s = (df["sex"]=="Male").values.astype(int).astype(int)
    features = columns[1:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        print("s dist:", pd.value_counts(s).values.ravel()/len(s))
    return X_dummy, y, s

def get_recidivism_multiple_1(show=True):
    df = pd.read_csv("compas.csv")
    df = df.loc[df["race"].isin(["Caucasian", "African-American"])]
    df = df[
        [
            "age", "race", "sex",
            "decile_score", "priors_count", "c_charge_degree",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "two_year_recid",
        ]
    ]
    columns = df.columns.tolist()
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    
    df = df[columns].copy()
    df = df.dropna()
    y = (df["two_year_recid"]==1).values.astype(int)
    
    # 1, 1, 1 ---> young, white, male
    s_age = (df["age"]<=24).values.astype(int).astype(int).reshape(-1,1)
    s_race = (df["race"]=="Caucasian").values.astype(int).astype(int).reshape(-1,1)
    s_gender = (df["sex"]=="Male").values.astype(int).astype(int).reshape(-1,1)
    
    s = np.concatenate(
        (s_age, s_race, s_gender),
        axis=1
    )
    
    features = columns[3:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        for s_column in range(s.shape[1]):
            print("s dist "+str(s_column)+":", pd.value_counts(s[:,s_column]).values.ravel()/len(s))
            
    return X_dummy, y, s

def get_recidivism_multiple_2(show=True):
    df = pd.read_csv("compas.csv")
    df = df.loc[df["race"].isin(["Caucasian", "African-American"])]
    df = df[
        [
            "age", "race", "sex",
            "decile_score", "priors_count", "c_charge_degree",
            "juv_fel_count", "juv_misd_count", "juv_other_count",
            "two_year_recid",
        ]
    ]
    columns = df.columns.tolist()
    for column in columns:
        try:
            df[column] = df[column].astype(float)
        except:
            df[column] = df[column].astype(str)
    
    df = df[columns].copy()
    df = df.dropna()
    y = (df["two_year_recid"]==1).values.astype(int)
    
    s = []
    # 1, 1, 1 ---> young, white, male
    s_age = (df["age"]<=24).values.astype(int).astype(str).tolist()
    s_race = (df["race"]=="Caucasian").values.astype(int).astype(str).tolist()
    s_gender = (df["sex"]=="Male").values.astype(int).astype(str).tolist()
    
    for i in range(len(df)):
        row = s_age[i] + s_race[i] + s_gender[i]
        s.append(row)
    s = pd.get_dummies(s).values.astype(int)
    
    features = columns[3:-1]
    X_dummy = pd.get_dummies(df[features]).values # dummyfying (not really needed)
    X = df[features].values
    if show:
        display(df.head())
        print("X shape:", X.shape)
        print("X_dummy shape:", X_dummy.shape)
        print("y dist:", pd.value_counts(y).values.ravel()/len(y))
        for s_column in range(s.shape[1]):
            print("s dist "+str(s_column)+":", pd.value_counts(s[:,s_column]).values.ravel()/len(s))
            
    return X_dummy, y, s