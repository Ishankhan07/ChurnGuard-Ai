
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():

    df = pd.read_csv("data/Telecom_customer churn.csv")

    print(df.columns)
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df["Churn"].value_counts())

    # TotalCharges ko number mein convert karo
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Customer ID remove karo
    df.drop("customerID", axis=1, inplace=True)

    # Duplicate rows remove karo
    print("Duplicate rows:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)

    # Missing values handle karo
    df_num = df.select_dtypes(include="number").columns
    df_cat = df.select_dtypes(include="object").columns

    df[df_num] = df[df_num].fillna(df[df_num].median())

    for col in df_cat:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values remaining:", df.isnull().sum().sum())

    return df
