"""

scipt to convert data to desired format for fine-tuning Donut model
Desired format is:
dataset_name
├── test
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
├── train
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
└── validation
    ├── metadata.jsonl
    ├── {image_path0}
    ├── {image_path1}
              .
              .

"""
import os
import json

from tqdm import tqdm
from PIL import Image
import jsonlines

DATA_FOLDER_NAME = "TEST_DATA"


#Following the example data from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb#scrollTo=ok4IudPaFhqi

#read data
labeledReceipts = os.listdir("./docs/json")
labeledReceipts.remove("key") #do not want the key folder



def getMenuListAndTotal(jsonObj):
    menuList = [] # list of objects : {"nm": "Bbk Bengil Nasi", "cnt": "", "price": "125,000"}   -> cnt is empty string always
    # print(jsonObj["words"])
    total = 0

    uniqueRowLabels = set() # set of unique rows
    for box in jsonObj["words"]:
        uniqueRowLabels.add(box["label"])
    if (len(uniqueRowLabels) == 0):
        return None, None
    maxRow = max(list(uniqueRowLabels)) #to prevent adding the total as a menu item
    
    validLineArr = [] #for storing all words and their row and category
    currTotPrice = 0
    for rowLabel in uniqueRowLabels:
        #find item with rowLabel
        items = []
        for box in jsonObj["words"]:
            if box["label"] == rowLabel:
                items.append([box, box["rect"]["x1"]])
        
        if (items[0][1] < items[1][1]): #product is further to the left -> smaller x1
            productBox, priceBox = items[0][0], items[1][0]
        else:
            productBox, priceBox = items[1][0], items[0][0]


        if (rowLabel == maxRow and float(priceBox["value"].replace(",", ".")) >= currTotPrice): #last row is the total, dont want to add that. Have extra check for the fact that it is the total with the tot price check!
            total = priceBox["value"]
        else:
            menuList.append({"nm": productBox["value"], "cnt": "", "price": priceBox["value"]})
        currTotPrice += float(priceBox["value"].replace(",","."))
        
        productBoxTemp = getTempFromBox(productBox, isProduct=True)
        priceBoxTemp = getTempFromBox(priceBox, isProduct=False)
        validLineArr.append(productBoxTemp)
        validLineArr.append(priceBoxTemp)


    return menuList, total, validLineArr

def getTempFromBox(box, isProduct):
    #if point1 is bot left, point2 is bot right ,point3 is top right, point4 is top left
    x4,x2,y4,y2 = box["rect"]["x1"], box["rect"]["x2"], box["rect"]["y1"], box["rect"]["y2"] #we have bot left and top right corner
    x1,y1 = x4, y2
    x3,y3 = x2, y4
    #swap 2 and 3
    x2,y2,x3,y3 = x3,y3,x2,y2
    #swap 1 and 4
    x1,y1,x4,y4 = x4,y4,x1,y1

    if (isProduct):
        category = "menu.nm"
    else:
        category = "menu.price"

    temp = {"words" : [{"quad": 
            {""
            "x2": x2,
            "y3": y3,
            "x3": x3,
            "y4": y4,
            "x1": x1,
            "y1": y1,
            "x4": x4,
            "y2": y2},
            "is_key": 0, #always 0
            "row_id": int(box["label"]),
            "text": box["value"]
            }],
            "category": category,
            "group_id": box["label"],
            "sub_group_id": 0, #always 0
            }
    return temp



def getFullImgPath(imageName):
    path = "./docs/images/" + imageName
    if (os.path.exists(path)):
        return path
    print("DID NOT FIND PICTURE WITH NAME: " + imageName)
    return None

def prepareData():
    """prepares all data"""
    dictArr = []
    for idx, receipt in tqdm(enumerate(labeledReceipts)):
        newJsonObject = prepareReceipt(receipt, index=idx)
        if (newJsonObject != None):
            dictArr.append(newJsonObject)
    return dictArr

def prepareReceipt(receipt, index):
    #read json file
    jsonObj = json.load(open("./docs/json/" + receipt))

    newJsonObj = {}

    width, height = jsonObj["meta"]["image_size"]["width"], jsonObj["meta"]["image_size"]["height"]
    if (width > height):
        print("IGNORING RECEIPT SINCE IT IS NOT IN PORTRAIT MODE")
        return None
    # newJsonObj["meta"] = jsonObj["meta"]  #NOTE not needed
    
    menuList, total, validLineArr = getMenuListAndTotal(jsonObj)
    if (menuList == None):
        return None
    newJsonObj["gt_parse"] = {"menu":menuList, "total": {"total_price": total}} #adding menu list and total

    #getting info for "valid_line" key
        
    newJsonObj["valid_line"] = validLineArr
    newJsonObj["roi"] = dict()
    newJsonObj["repeating_symbol"] = dict()
    newJsonObj["dontcare"] = dict()

    imageName = receipt.split(".")[0]+".jpg"
    imgPath = getFullImgPath(imageName)
    # img = Image.open(imgPath)
    fullDict = {"file_name":str(index)+".jpg", "ground_truth": json.dumps(newJsonObj)}
    
    return fullDict
        

dataset = prepareData()


valIdx = int(len(dataset) * 0.8)
testIdx = int(len(dataset) * 0.9)

#split dataset into train/validation/test
train_dataset = dataset[:valIdx]
val_dataset = dataset[valIdx:testIdx]
test_dataset = dataset[testIdx:]

#create dirs if it does not exist
if not os.path.isdir(DATA_FOLDER_NAME):
	os.mkdir(DATA_FOLDER_NAME)

for subfolder in ["train", "validation", "test"]:
    if not os.path.isdir(f"{DATA_FOLDER_NAME}/{subfolder}"):
        os.mkdir(f"{DATA_FOLDER_NAME}/{subfolder}") 

#train
with jsonlines.open(f"{DATA_FOLDER_NAME}/train/metadata.jsonl", "w") as writer:
    writer.write_all(train_dataset)
#push all images to folder
for data in tqdm(train_dataset):
    imgName = data["file_name"]
    imgPath = getFullImgPath(imgName)
    print(imgName)
    img = Image.open(imgPath)
    img.save(f"{DATA_FOLDER_NAME}/train/{imgName}")

#val
with jsonlines.open(f"{DATA_FOLDER_NAME}/validation/metadata.jsonl", "w") as writer:
    writer.write_all(val_dataset)
#push all images to folder
for data in tqdm(val_dataset):
    imgName = data["file_name"]
    imgPath = getFullImgPath(imgName)
    img = Image.open(imgPath)
    img.save(f"{DATA_FOLDER_NAME}/val/{imgName}")
#test
with jsonlines.open(f"{DATA_FOLDER_NAME}/test/metadata.jsonl", "w") as writer:
    writer.write_all(test_dataset)
#push all images to folder
for data in tqdm(test_dataset):
    imgName = data["file_name"]
    imgPath = getFullImgPath(imgName)
    if (imgPath is not None):
        img = Image.open(imgPath)
        img.save(f"{DATA_FOLDER_NAME}/test/{imgName}")
    
