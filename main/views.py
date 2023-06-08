from django.shortcuts import render
from joblib import load
from django.core.cache import cache
from django.views.decorators.cache import cache_page
import matplotlib.pyplot as plt
import io
import base64

"""

Diabetes is a chronic medical condition characterized by high levels of blood sugar or glucose. 
It occurs when the body either does not produce enough insulin or cannot effectively use the insulin it produces. 
Insulin is a hormone produced by the pancreas that helps regulate blood sugar levels and allows glucose to enter the body's 
cells to be used as energy.

18-24 bmi good

X:[[Female,80.0,0,1,never,25.19,6.6,140],    Y:[0, 1],
    [Female,80.0,0,1,never,25.19,6.6,140]]   
"""

# I have used bootstrap: https://getbootstrap.com/docs/5.3/getting-started/introduction
# An also sklearn: https://scikit-learn.org/stable/index.html
#Databse Used: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset


# Load models and transformers outside the view function
mlp = load('main/models/mpl_nureal_network.joblib')
knn = load('main/models/knn.joblib')
forest = load('main/models/RandomForestClassifier.joblib')
svc = load('main/models/svc.joblib')
scaler = load('main/models/scaler.joblib')
ct_gender = load('main/models/gender_ct.joblib')
ct_smoker = load('main/models/smoker_ct.joblib')


@cache_page(1200)
def form_view(request):
    if request.method == 'POST':
        age = request.POST.get('age')
        bmi = request.POST.get('bmi')
        hba1c = request.POST.get('hba1c')
        glucose = request.POST.get('glucose')
        gender = request.POST.get('gender')
        hypertension = request.POST.get('hypertension')
        heart_disease = request.POST.get('HeartDisease')
        smoking = request.POST.get('smoking')

        try:
            age = float(age)
            bmi = float(bmi)
            hba1c = float(hba1c)
            glucose = float(glucose)
            
            gender = gender.capitalize()

            if hypertension == 'on':
                hypertension = 1
            elif hypertension is None:
                hypertension = 0
            
            if heart_disease == 'on':
                heart_disease = 1
            elif heart_disease is None:
                heart_disease = 0
    
        except:
            pass
        
        precent = 0

        if hba1c > 6:
            precent += 60

        elif hba1c > 5.7:
            percent += 40
        
        if glucose > 210:
            percent += 25
        
        

        input_non_6 = [[gender, age+5, hypertension, heart_disease, smoking, bmi, hba1c, glucose]]

        averages = []
        ages = [age, age+1, age+2, age+3, age+4, age+5]

        encoded_6 = ct_gender.transform(input_non_6)


        final_6 = ct_smoker.transform(encoded_6)
        final_6 = scaler.transform(final_6)


        svc_pred = svc.predict_proba(final_6)
        svc_pred = svc_pred[:, 1]
        mlp_pred = mlp.predict_proba(final_6)
        mlp_pred = mlp_pred[:, 1]
        knn_pred = knn.predict_proba(final_6)
        knn_pred = knn_pred[:, 1]
        forest_pred = forest.predict_proba(final_6)
        forest_pred = forest_pred[:, 1]

        value = mlp_pred[0] + knn_pred[0] + forest_pred[0] + svc_pred[0]
        value = value / 4
        value = value * 100
        value = round(value, 2)


        final = value+precent

        if final > 100:
            final = 99.5
        
        averages.append(final)



        print(averages)
        print(ages)

        return render(request, 'results.html', {'prediction': averages, 'ages': ages})
    
    # If the request method is GET, render the form template
    return render(request, 'index.html')
