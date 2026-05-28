# 🌾 TP IA Agriculture Burundi — Guide de démarrage

## 📦 Contenu du projet
```
Agriculture_IA/
├── app.py
├── agriculture_burundi.csv
└── modeles/
    ├── arbre_decision.pkl
    ├── foret_aleatoire.pkl
    ├── regression_logistique.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── num_features.pkl
    └── metrics.pkl
```

## 🚀 Lancer l'application web

```bash
pip install streamlit scikit-learn pandas numpy matplotlib joblib
streamlit run app.py
```
Ouvrir : http://localhost:8501

## 📓 Exécuter le notebook

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
jupyter notebook TP_Agriculture_Burundi.ipynb
```

## 🌐 Déployer sur Streamlit Cloud (gratuit)
1. Créer un compte sur https://streamlit.io/cloud
2. Pousser le code sur GitHub (dépôt public)
3. Connecter le dépôt → déploiement automatique

## 📊 Performances des modèles
| Modèle              | Accuracy | AUC    |
|---------------------|----------|--------|
| Arbre de Décision   | 89.87%   | 0.7621 |
| Forêt Aléatoire     | 93.35%   | 0.7450 |
| Régression Log.     | 93.67%   | 0.8303 |
