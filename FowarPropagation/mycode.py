import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


with open('info1.json') as f:
    js = json.loads(f.read())
df = pd.DataFrame(js)
df = pd.read_json('info1.json')

print(df)
