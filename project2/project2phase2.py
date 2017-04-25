#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import urllib.request


df = pd.read_json("yummly.json")

col_names = ['id', 'cuisine', 'ingredients']
df = df.reindex(columns=col_names)
#df = df.set_index('id')
#le = LabelEncoder()
#le.fit(df.cuisine.unique())
#df["cuisine"] = le.transform(df["cuisine"])

s = df["ingredients"]

dummies_df = pd.get_dummies(s.apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)
df = pd.concat([df, dummies_df], axis=1)
print(df.head())

