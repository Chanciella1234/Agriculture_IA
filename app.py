import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

# ── Chemin absolu vers le dossier modeles (même dossier que app.py) ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELES_DIR = os.path.join(BASE_DIR, 'modeles')

# ── Configuration page ──────────────────────────────────────
st.set_page_config(
    page_title="Prédiction des Récoltes — Burundi",
    page_icon="🌾",
    layout="wide",
)

# ── Chargement des modèles ───────────────────────────────────
@st.cache_resource
def load_models():
    dt      = joblib.load(os.path.join(MODELES_DIR, 'arbre_decision.pkl'))
    rf      = joblib.load(os.path.join(MODELES_DIR, 'foret_aleatoire.pkl'))
    lr      = joblib.load(os.path.join(MODELES_DIR, 'regression_logistique.pkl'))
    scaler  = joblib.load(os.path.join(MODELES_DIR, 'scaler.pkl'))
    feats   = joblib.load(os.path.join(MODELES_DIR, 'feature_names.pkl'))
    num_f   = joblib.load(os.path.join(MODELES_DIR, 'num_features.pkl'))
    metrics = joblib.load(os.path.join(MODELES_DIR, 'metrics.pkl'))
    return dt, rf, lr, scaler, feats, num_f, metrics

dt, rf, lr, scaler, feature_names, num_features, metrics = load_models()
MODELS = {
    'Arbre de Décision': dt,
    'Forêt Aléatoire': rf,
    'Régression Logistique': lr,
}

PROVINCES = [
    'Bujumbura Rural', 'Gitega', 'Ngozi', 'Muyinga', 'Kirundo',
    'Kayanza', 'Muramvya', 'Mwaro', 'Bubanza', 'Cibitoke',
    'Makamba', 'Rutana', 'Ruyigi', 'Cankuzo', 'Bururi'
]
CULTURES = ['Maïs', 'Haricot', 'Manioc', 'Patate douce', 'Sorgho', 'Bananier']

# ── En-tête ──────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a5c2a, #2ecc71); padding: 28px 30px;
     border-radius: 14px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);'>
  <h1 style='color:white; margin:0; font-size:2.1rem;'>🌾 Prédiction des Récoltes au Burundi</h1>
  <p style='color:#d4f5e0; margin:6px 0 0 0; font-size:1rem;'>
    IA Appliquée à l'Agriculture 
  </p>
</div>
""", unsafe_allow_html=True)

# ── Onglets ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📊 Performances des Modèles"])

# ════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Saisir les caractéristiques de la parcelle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Localisation & Culture**")
        province = st.selectbox("Province", PROVINCES, index=1)
        culture  = st.selectbox("Culture", CULTURES, index=0)
        saison   = st.radio("Saison", ["A (Mars–Juin)", "B (Sept–Déc)"])
        saison_val = "A" if saison.startswith("A") else "B"

    with col2:
        st.markdown("**🌡️ Conditions Climatiques**")
        altitude      = st.slider("Altitude (m)", 700, 2100, 1500, step=10)
        pluviometrie  = st.slider("Pluviométrie (mm)", 200, 1300, 850, step=10)
        temperature   = st.slider("Température moyenne (°C)", 15.0, 28.0, 20.0, step=0.1)

    with col3:
        st.markdown("**🌱 Caractéristiques de la Parcelle**")
        superficie    = st.slider("Superficie (ha)", 0.3, 5.0, 2.0, step=0.1)
        nb_menages    = st.slider("Nombre de ménages", 10, 200, 80)
        engrais       = st.toggle("🧪 Utilisation d'engrais", value=True)
        irrigation    = st.toggle("💧 Accès à l'irrigation", value=False)

    # ── Choix du modèle ─────────────────────────────────────
    st.markdown("---")
    model_choice = st.selectbox("🤖 Modèle à utiliser",
                                list(MODELS.keys()), index=1)

    # ── Bouton prédire ───────────────────────────────────────
    if st.button("🔮 Lancer la prédiction", use_container_width=True, type="primary"):
        # Construire le DataFrame
        data = {
            'province': [province],
            'culture':  [culture],
            'saison':   [saison_val],
            'altitude_m': [altitude],
            'pluviometrie_mm': [pluviometrie],
            'temperature_moy_C': [temperature],
            'superficie_ha': [superficie],
            'utilisation_engrais': [int(engrais)],
            'acces_irrigation': [int(irrigation)],
            'nb_menages': [nb_menages],
        }
        df_input = pd.DataFrame(data)
        df_enc   = pd.get_dummies(df_input, columns=['province', 'culture', 'saison'],
                                  drop_first=False)
        for col in feature_names:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[[c for c in df_enc.columns if c in feature_names]]
        df_enc = df_enc[feature_names]
        df_enc[num_features] = scaler.transform(df_enc[num_features])

        model   = MODELS[model_choice]
        pred    = model.predict(df_enc)[0]
        prob    = model.predict_proba(df_enc)[0]
        prob_1  = prob[1] * 100  # probabilité bonne récolte

        # ── Résultat principal ───────────────────────────────
        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div style='background:#d4edda; border:2px solid #28a745; border-radius:12px;
                 padding:24px; text-align:center; margin:10px 0;'>
              <h2 style='color:#155724; margin:0;'>✅ BONNE RÉCOLTE PRÉVUE</h2>
              <p style='color:#155724; font-size:1.3rem; margin:10px 0 0 0;'>
                Probabilité : <strong>{prob_1:.1f}%</strong>
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#f8d7da; border:2px solid #dc3545; border-radius:12px;
                 padding:24px; text-align:center; margin:10px 0;'>
              <h2 style='color:#721c24; margin:0;'>⚠️ MAUVAISE RÉCOLTE PRÉVUE</h2>
              <p style='color:#721c24; font-size:1.3rem; margin:10px 0 0 0;'>
                Probabilité de mauvaise récolte : <strong>{100-prob_1:.1f}%</strong>
              </p>
            </div>""", unsafe_allow_html=True)

        # ── Jauge probabilité ───────────────────────────────
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 1.3))
            color = '#28a745' if prob_1 >= 50 else '#dc3545'
            ax.barh([0], [prob_1], color=color, height=0.5, edgecolor='white')
            ax.barh([0], [100 - prob_1], left=[prob_1],
                    color='#f0f0f0', height=0.5, edgecolor='white')
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Probabilité de bonne récolte (%)")
            ax.axvline(50, color='gray', linestyle='--', linewidth=1)
            ax.text(prob_1 / 2, 0, f"{prob_1:.1f}%",
                    ha='center', va='center', color='white',
                    fontweight='bold', fontsize=12)
            ax.set_title(f"Confiance du modèle · {model_choice}", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Comparaison 3 modèles ────────────────────────────
        st.markdown("#### 🔄 Comparaison des 3 modèles")
        cols = st.columns(3)
        for i, (mname, mmodel) in enumerate(MODELS.items()):
            mp   = mmodel.predict(df_enc)[0]
            mprob = mmodel.predict_proba(df_enc)[0][1] * 100
            emoji = "✅" if mp == 1 else "❌"
            label = "Bonne récolte" if mp == 1 else "Mauvaise récolte"
            with cols[i]:
                border = "#28a745" if mp == 1 else "#dc3545"
                bg     = "#f0fff4" if mp == 1 else "#fff5f5"
                st.markdown(f"""
                <div style='border:2px solid {border}; border-radius:10px;
                     padding:16px; text-align:center; background:{bg};'>
                  <strong>{mname}</strong><br>
                  <span style='font-size:1.5rem;'>{emoji}</span><br>
                  {label}<br>
                  <small>{mprob:.1f}% bonne récolte</small>
                </div>""", unsafe_allow_html=True)

        # ── Recommandations ──────────────────────────────────
        if pred == 0 or prob_1 < 60:
            st.markdown("---")
            st.markdown("#### 💡 Recommandations agronomiques")
            recos = []
            if pluviometrie < 600:
                recos.append("🌧️ Pluviométrie très faible — envisager la collecte d'eau ou l'irrigation")
            if not engrais:
                recos.append("🧪 L'utilisation d'engrais peut améliorer le rendement")
            if not irrigation and pluviometrie < 700:
                recos.append("💧 Accès à l'irrigation recommandé pour ce niveau de pluie")
            if culture == 'Haricot' and pluviometrie < 650:
                recos.append("🌱 Le Haricot est sensible à la sécheresse — envisager Sorgho ou Manioc")
            if temperature > 24 and culture in ['Maïs', 'Haricot']:
                recos.append("🌡️ Température élevée pour cette culture — considérer des variétés thermotolérantes")
            if not recos:
                recos.append("⚠️ Conditions limites — surveiller l'évolution climatique et préparer un plan B")
            for r in recos:
                st.warning(r)

# ════════════════════════════════════════════════════════════
# ONGLET 2 — PERFORMANCES
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Performances des modèles sur le jeu de test (20%)")

    col1, col2, col3 = st.columns(3)
    for col, (mname, mmet) in zip([col1, col2, col3], metrics.items()):
        with col:
            f1_0 = mmet['report']['0']['f1-score']
            f1_1 = mmet['report']['1']['f1-score']
            st.markdown(f"""
            <div style='background:#f8f9fa; border-radius:12px; padding:20px;
                 border-left:5px solid #2ecc71; text-align:center;'>
              <h4 style='margin:0 0 10px 0;'>{mname}</h4>
              <div style='font-size:1.8rem; font-weight:bold; color:#2ecc71;'>
                {mmet['accuracy']*100:.2f}%
              </div>
              <small>Accuracy</small><br><br>
              <div>AUC-ROC : <strong>{mmet['auc']:.4f}</strong></div>
              <div>F1 (Bonne) : <strong>{f1_1:.3f}</strong></div>
              <div>F1 (Mauvaise) : <strong>{f1_0:.3f}</strong></div>
            </div>""", unsafe_allow_html=True)

    # Graphique comparatif
    st.markdown("---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mnames = list(metrics.keys())
    short  = ['Arbre (DT)', 'Forêt (RF)', 'Régression (LR)']
    accs   = [metrics[m]['accuracy'] * 100 for m in mnames]
    aucs   = [metrics[m]['auc'] for m in mnames]

    bars1 = axes[0].bar(short, accs, color=['#3498db','#27ae60','#e74c3c'],
                        edgecolor='white', width=0.5)
    axes[0].set_ylim(80, 100)
    axes[0].set_title('Accuracy (%)', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    for b, v in zip(bars1, accs):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                     f'{v:.2f}%', ha='center', fontweight='bold')

    bars2 = axes[1].bar(short, aucs, color=['#3498db','#27ae60','#e74c3c'],
                        edgecolor='white', width=0.5)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title('AUC-ROC', fontweight='bold')
    axes[1].set_ylabel('AUC')
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Aléatoire (0.5)')
    for b, v in zip(bars2, aucs):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    **💡 Interprétation :**
    La Forêt Aléatoire offre la meilleure accuracy globale (93.35%).
    La Régression Logistique a la meilleure AUC (0.83) — meilleure discrimination entre classes.
    L'AUC est la métrique à privilégier ici car le dataset est déséquilibré (93% bonnes récoltes).
    """)
