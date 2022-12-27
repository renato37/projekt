import pandas as pd
import json
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/renato37/projekt/main/default/texts.json"
response = urlopen(url)
texts = json.loads(response.read())
data_basic = pd.read_csv("https://raw.githubusercontent.com/renato37/projekt/main/basic.csv")
