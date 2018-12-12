import random
import pandas as pn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
# import data from body or subject
def loadcsv(filename):
    dataset = pn.read_csv(filename)
    return dataset.values

# split tarin and test
def splitDataset(dataset, splitRatio):
    data_split = separateByClass(dataset)
    trainSet = []
    testSet = []
    trainSize = 0
    for Class in data_split:
        copy = data_split[Class]
        trainSize += int(len(copy) * splitRatio)
        while len(trainSet) < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))
        testSet.extend(copy)
    return [trainSet, testSet]

# Seperate the Dataset by the Label 0(Not Spam)/1(Spam)
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def laplace_p0(numbers,k,Nw):
    laplace0 = 1-(sum(numbers)+k) / float(len(numbers)+k*Nw)
    return laplace0


def laplace_p1(numbers,k,Nw):
    laplace1 = (sum(numbers)+k) / float(len(numbers)+k*Nw)
    return laplace1


def summarize(dataset,k,Nw):
    summaries = [(laplace_p0(attribute,k,Nw), laplace_p1(attribute,k,Nw)) for attribute in zip(*dataset)]
    del summaries[-1]
    del summaries[0]
    return summaries
#Calculate the Probabilty of Each Word:
def caculate_P(dataset,k=1,Nw=2):
    instance_num = len(dataset)
    separate = separateByClass(dataset)
    P_Class = {}
    P_Att = {}
    for ClassValue in separate:
        instances = separate[ClassValue]
        P_Att[ClassValue] = summarize(instances,k,Nw)
        P_Class[ClassValue] = len(instances)/instance_num
    Model = Model_Separate((P_Class,P_Att))
    return Model

def Model_Separate(Raw_Model):
    P_Class, P_Att = Raw_Model
    Model = {}
    for Class in P_Class:
        Model[Class] = P_Class[Class],P_Att[Class]
    return Model


def calculate_hx(Model,instance):
    instance = instance[1:-1]
    probability = Model[0]
    for att in range(len(instance)):
        probability *= Model[1][att][instance[att]]
    return probability


def predict(Model,test):
    hx = {}
    prediction = []
    ground_true = []
    for instance in test:
        ground_true.append(instance[-1])
        for Class in Model:
            hx[Class] = calculate_hx(Model[Class],instance)
        debug = max(hx, key=hx.get)
        prediction.append(max(hx, key=hx.get))
    return prediction,ground_true

def data_for_skl(data):
    data_skl = []
    label_skl = []
    for instance in data:
        data_skl.append(instance[1:-1])
        label_skl.append(instance[-1])
    return data_skl,label_skl
def hm5(filename):
    dataset = loadcsv(filename)
    splitratio = 0.8
    fscore = []
    fscore_skl = []
    for k in range(100):
        train, test = splitDataset(dataset, splitratio)
        data_train,label_train = data_for_skl(train)
        data_test, label_test = data_for_skl(test)
        clf = MultinomialNB()
        clf.fit(data_train,label_train)
        pred_skl = clf.predict(data_test)
        Model = caculate_P(train,k=0)
        prediction,ground_true = predict(Model,test)
        fscore.append(metrics.f1_score(ground_true,prediction))
        fscore_skl.append(metrics.f1_score(label_test,pred_skl))
    return fscore_skl
filename_body = 'dbworld_bodies_stemmed.csv'
filename_subject = 'dbworld_subjects_stemmed.csv'

fscore = [hm5(filename_body),hm5(filename_subject)]
fig1, ax1 = plt.subplots()
ax1.set_title('Sklearn F-score Plot')
ax1.boxplot(fscore)
plt.show()

