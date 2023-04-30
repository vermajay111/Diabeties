from django.shortcuts import render
from joblib import load
from django.core.cache import cache
from django.views.decorators.cache import cache_page


@cache_page(120)
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
            elif hypertension == None:
                 hypertension = 0
            
            if heart_disease == 'on':
                 heart_disease = 1
            elif heart_disease == None:
                heart_disease = 0
    
            
        except:
            pass

        mlp = load('main/models/mpl_nureal_network.joblib')
        knn = load('main/models/knn.joblib')
        forest = load('main/models/RandomForestClassifier.joblib')

        scaler = load('main/models/scaler.joblib')
        ct_gender = load('main/models/gender_ct.joblib')
        ct_smoker = load('main/models/smoker_ct.joblib')
        

        input_non = [[gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose]]
        encoded = ct_gender.transform(input_non)
        final = ct_smoker.transform(encoded)

        final = scaler.transform(final)


        mlp_pred = mlp.predict_proba(final)
        mlp_pred = mlp_pred[:, 1]
        knn_pred = knn.predict_proba(final)
        knn_pred = knn_pred[:, 1]
        forest_pred = forest.predict_proba(final)
        forest_pred = forest_pred[:, 1]

        value = mlp_pred[0] + knn_pred[0] + forest_pred[0]
        value = value / 3
        
        final = round(value, 3)
        if final == 1.0:
          final = 100

        return render(request, 'results.html', {'prediction': final})
    
    # If the request method is GET, render the form template
    return render(request, 'index.html')
