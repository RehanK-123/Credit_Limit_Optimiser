from django.shortcuts import render
from django.http import HttpResponse
from . import models
import pickle 
import numpy as np
import pandas as pd 
import xgboost as xg
# Create your views here.

def render_home_page(request):
    return render(request, "index.html")

def render_classify_page(request):
    return render(request, "classify.html")

def render_classification(request):
    data = []
    for i in ["Credit_Limit", "Avg_Utilization_Ratio", "Pay_on_time", "Months_on_book", "Income_Category", "Dependent_count"]:
        if i == "Pay_on_time" or i == "Dependent_count" or i == "Months_on_book":
            data.append(int(request.POST.get(i)))
        else:
            data.append(float(request.POST.get(i)))
    model = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/cluster_model", 'rb'))
    # data = [float(i) for i in data]
    print(data)
    cluster = model.predict(np.array(data).reshape(1, -1))
    if cluster == [1]:
        cluster = "Low Risk"
    else:
        cluster = "High Risk"
    return render(request, "classification.html", {"cluster" : cluster})

def render_predict_page(request):
    return render(request, "home.html")

def render_prediction(request):
    data = []
    for i in ["Avg_Utilization_Ratio", "Pay_on_time", "Income_Category", "Months_on_book" ,"Education_Level", "Card_Category"]:
        if i == "Avg_Utilization_Ratio":
            data.append(np.log1p(np.array(float(request.POST[i]))).reshape(-1, 1)[0][0])
            continue

        if i == "Card_Category":
            encoder1 = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/card_cat", 'rb'))
            data.append(encoder1.transform(np.array(request.POST[i]).reshape(1, -1)).toarray()[0][0])
            continue

        if i == "Education_Level":
            encoder2 = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/ed_lvl", 'rb'))
            data.append(encoder2.transform(np.array(request.POST[i]).reshape(-1, 1))[0][0])
            continue
        
        if i == "Pay_on_time" or i == "Months_on_book":
            data.append(int(request.POST[i]))
            continue

        if i == "Income_Category":
            data.append(float(request.POST[i]))
            continue
        data.append(request.POST[i])
    
    # print(data)
    data.insert(2, data[2] * data[3])
    print(np.array(data).reshape(1, -1))
    encoder3 = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/feature_scaler", 'rb'))
    data = encoder3.transform(np.array(data).reshape(1, -1))
    model = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/predictor_model", 'rb'))
    scaler = pickle.load(open("/home/rehan-khan/Downloads/Credit_Limit_Optimiser-master/target_scaler", 'rb'))
    cred = scaler.inverse_transform(np.array(model.predict(data)).reshape(1, -1))
    return render(request, "prediction.html", {"credit_limit" : cred[0][0]})