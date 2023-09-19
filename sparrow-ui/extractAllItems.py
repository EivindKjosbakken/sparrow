"""
Given your labelled dataset, extracts all items and their price
This data can then be used to create synthetic receipt data
"""
import os
import json
import pickle

PATH_TO_DATA = "docs/json"

files = os.listdir(PATH_TO_DATA)
files.remove("key")

itemsAndPrice = [] #2d list, row is (item, price)
for filename in files:
	jsonObj = json.load(open(PATH_TO_DATA + "/" + filename))
	temp = None
	for item in jsonObj["words"]:
		if (temp is None):
			temp = item["value"]
		else:
			itemsAndPrice.append((temp, item["value"]))
			temp = None


print(len(itemsAndPrice))
with open('itemsAndPrice.pickle', 'wb') as f:
    pickle.dump(itemsAndPrice, f)


"""
#if you want to read:
import pandas as pd
df2 = pd.read_pickle('itemsAndPrice.pickle')
print(df2)
"""