from django.shortcuts import render
import numpy as np
import joblib
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
import os
from django.conf import settings

# Chemin vers le dossier "models" dans votre application Django
model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'models')

# Chargement des modèles avec le chemin absolu
dt_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
nn_model = joblib.load(os.path.join(model_dir, 'neural_network_model.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))


def home(request):
    prediction = None
    if request.method == 'POST':
        # Récupération des données du formulaire
        try:
            age = float(request.POST['age'])
            sexe = int(request.POST['sexe'])  # 0 ou 1
            tour_taille = float(request.POST['tour_taille'])
            imc = float(request.POST['imc'])
            glycemie = float(request.POST['glycemie'])
            hdl = float(request.POST['hdl'])
            trigly = float(request.POST['trigly'])

            # Création tableau numpy (1 échantillon)
            X_input = np.array([[age, sexe, tour_taille, imc, glycemie, hdl, trigly]])

            # Normalisation (avec scaler chargé)
            X_scaled = scaler.transform(X_input)

            # Prédictions modèles
            y_pred_dt = dt_model.predict(X_scaled)
            y_pred_svm = svm_model.predict(X_scaled)
            y_pred_nn = nn_model.predict(X_scaled)

            # Vote majoritaire
            preds = np.array([y_pred_dt[0], y_pred_svm[0], y_pred_nn[0]])
            vote = mode(preds)[0][0]

            prediction = "Atteint du syndrome métabolique" if vote == 1 else "Non atteint"

        except Exception as e:
            prediction = f"Erreur lors de la prédiction : {e}"

    return render(request, 'predictor/home.html', {'prediction': prediction})

# Create your views here.
