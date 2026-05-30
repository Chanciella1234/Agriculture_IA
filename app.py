import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELES_DIR = os.path.join(BASE_DIR, 'modeles')

st.set_page_config(
    page_title="AgriSmart Burundi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════
# CSS — Design Terracotta × Forêt · Unique & Professionnel
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --terra:    #C0622A;
  --terra-dk: #8B3E15;
  --terra-lt: #E8A882;
  --forest:   #1D4A2F;
  --forest-md:#2D6B45;
  --forest-lt:#4A9463;
  --cream:    #FAF6EF;
  --sand:     #EDE4D3;
  --sand-dk:  #D4C4A8;
  --charcoal: #1A1A18;
  --muted:    #6B6355;
  --white:    #FFFFFF;
  --success:  #2D8B4E;
  --danger:   #B03A2E;
  --gold:     #C9A84C;
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--cream);
  color: var(--charcoal);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── Hero Banner ── */
.hero {
  background: var(--forest);
  background-image:
    radial-gradient(ellipse at 80% 50%, rgba(192,98,42,0.18) 0%, transparent 60%),
    radial-gradient(ellipse at 10% 80%, rgba(74,148,99,0.15) 0%, transparent 50%);
  padding: 52px 60px 44px 60px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 320px; height: 320px;
  border-radius: 50%;
  border: 2px solid rgba(192,98,42,0.2);
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 40%;
  width: 200px; height: 200px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.06);
}
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--terra-lt);
  margin-bottom: 14px;
}
.hero h1 {
  font-family: 'Playfair Display', serif;
  font-size: 3rem;
  font-weight: 900;
  color: var(--white);
  margin: 0 0 10px 0;
  line-height: 1.1;
}
.hero h1 span { color: var(--terra-lt); }
.hero-sub {
  color: rgba(255,255,255,0.55);
  font-size: 0.95rem;
  font-weight: 300;
  max-width: 520px;
}
.hero-badges {
  display: flex; gap: 10px; margin-top: 24px; flex-wrap: wrap;
}
.badge {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  color: rgba(255,255,255,0.75);
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  padding: 5px 14px;
  border-radius: 100px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--forest);
  padding: 0 60px;
  gap: 0;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.85rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  color: rgba(255,255,255,0.45) !important;
  padding: 16px 28px;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
  color: var(--terra-lt) !important;
  border-bottom: 2px solid var(--terra-lt) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--cream);
  padding: 40px 60px;
}

/* ── Section titles ── */
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--forest);
  margin: 0 0 6px 0;
}
.section-sub {
  color: var(--muted);
  font-size: 0.88rem;
  margin-bottom: 32px;
}

/* ── Card panels ── */
.card {
  background: var(--white);
  border-radius: 16px;
  padding: 28px;
  border: 1px solid var(--sand-dk);
  box-shadow: 0 2px 12px rgba(29,74,47,0.06);
}
.card-label {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--terra);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.card-label::before {
  content: '';
  display: inline-block;
  width: 18px; height: 2px;
  background: var(--terra);
}

/* ── Streamlit inputs restyling ── */
.stSelectbox > div > div,
.stSlider > div,
.stRadio > div {
  background: var(--white) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--forest) !important;
  letter-spacing: 0.03em;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--terra) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
  color: var(--terra) !important;
}

/* ── Toggle ── */
.stToggle label { font-size: 0.82rem !important; }

/* ── Predict Button ── */
.stButton > button {
  background: var(--terra) !important;
  color: white !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 14px 32px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 14px rgba(192,98,42,0.35) !important;
}
.stButton > button:hover {
  background: var(--terra-dk) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(192,98,42,0.45) !important;
}

/* ── Result cards ── */
.result-good {
  background: linear-gradient(135deg, #1D4A2F 0%, #2D6B45 100%);
  border-radius: 20px;
  padding: 36px 40px;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(29,74,47,0.3);
}
.result-good::before {
  content: '✦';
  position: absolute; top: 16px; right: 24px;
  font-size: 2rem; color: rgba(255,255,255,0.08);
}
.result-bad {
  background: linear-gradient(135deg, #6B1A12 0%, #B03A2E 100%);
  border-radius: 20px;
  padding: 36px 40px;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(176,58,46,0.3);
}
.result-tag {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.5);
  margin-bottom: 10px;
}
.result-title {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 700;
  color: white;
  margin: 0 0 8px 0;
}
.result-prob {
  font-size: 3rem;
  font-weight: 700;
  color: white;
  font-family: 'DM Mono', monospace;
}
.result-prob-label {
  color: rgba(255,255,255,0.5);
  font-size: 0.8rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Model comparison cards ── */
.model-card {
  background: var(--white);
  border: 1.5px solid var(--sand-dk);
  border-radius: 14px;
  padding: 20px;
  text-align: center;
  transition: all 0.2s ease;
}
.model-card.good { border-color: var(--forest-lt); background: #f0faf3; }
.model-card.bad  { border-color: #e08070; background: #fdf5f4; }
.model-card-name {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.model-card-verdict {
  font-family: 'Playfair Display', serif;
  font-size: 1rem;
  font-weight: 700;
}
.model-card-verdict.good { color: var(--forest); }
.model-card-verdict.bad  { color: var(--danger); }
.model-card-pct {
  font-family: 'DM Mono', monospace;
  font-size: 1.4rem;
  font-weight: 500;
  margin-top: 4px;
}
.model-card-pct.good { color: var(--forest-md); }
.model-card-pct.bad  { color: var(--danger); }

/* ── Reco cards ── */
.reco-item {
  background: var(--white);
  border-left: 3px solid var(--gold);
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  margin-bottom: 10px;
  font-size: 0.88rem;
  color: var(--charcoal);
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Perf metric cards ── */
.perf-card {
  background: var(--white);
  border-radius: 18px;
  padding: 28px 24px;
  border: 1px solid var(--sand-dk);
  text-align: center;
  position: relative;
  overflow: hidden;
}
.perf-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--terra), var(--forest-lt));
}
.perf-card-name {
  font-family: 'Playfair Display', serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--forest);
  margin-bottom: 16px;
}
.perf-acc {
  font-family: 'DM Mono', monospace;
  font-size: 2.6rem;
  font-weight: 500;
  color: var(--terra);
  line-height: 1;
}
.perf-acc-label {
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 4px 0 18px 0;
}
.perf-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-top: 1px solid var(--sand);
  font-size: 0.82rem;
}
.perf-row-label { color: var(--muted); }
.perf-row-val {
  font-family: 'DM Mono', monospace;
  font-weight: 500;
  color: var(--charcoal);
}

/* ── Divider ── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--sand-dk), transparent);
  margin: 36px 0;
}

/* ── Info box ── */
.info-box {
  background: linear-gradient(135deg, #EEF7F2, #F5EEE6);
  border: 1px solid var(--sand-dk);
  border-radius: 12px;
  padding: 20px 24px;
  font-size: 0.88rem;
  color: var(--charcoal);
  line-height: 1.7;
}

/* ── Selectbox label ── */
.stSelectbox label, .stSlider label, .stRadio label {
  font-size: 0.82rem !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .hero { padding: 32px 24px 28px 24px; }
  .hero h1 { font-size: 2rem; }
  .stTabs [data-baseweb="tab-list"] { padding: 0 16px; }
  .stTabs [data-baseweb="tab-panel"] { padding: 24px 16px; }
}
</style>
""", unsafe_allow_html=True)


# ── Chargement modèles ───────────────────────────────────────
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
    'Bujumbura Rural','Gitega','Ngozi','Muyinga','Kirundo',
    'Kayanza','Muramvya','Mwaro','Bubanza','Cibitoke',
    'Makamba','Rutana','Ruyigi','Cankuzo','Bururi'
]
CULTURES = ['Maïs','Haricot','Manioc','Patate douce','Sorgho','Bananier']


# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Université Polytechnique de Gitega &nbsp;·&nbsp; IA Appliquée</div>
  <h1>AgriSmart<br><span>Burundi</span></h1>
  <p class="hero-sub">
    Prédiction intelligente des récoltes à partir de données climatiques
    et agronomiques — 15 provinces · 6 cultures · 2015–2023
  </p>
  <div class="hero-badges">
    <span class="badge">🌳 Forêt Aléatoire</span>
    <span class="badge">🌳 Arbre de Décision</span>
    <span class="badge">📈 Régression Logistique</span>
    <span class="badge">1 620 observations</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Onglets ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Prédiction", "📊  Performances des Modèles"])


# ════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<p class="section-title">Caractéristiques de la parcelle</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Renseignez les paramètres agronomiques et climatiques pour obtenir une prédiction.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1], gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-label">📍 Localisation & Culture</div>', unsafe_allow_html=True)
        province   = st.selectbox("Province", PROVINCES, index=1)
        culture    = st.selectbox("Culture", CULTURES, index=0)
        saison     = st.radio("Saison agricole", ["A — Mars à Juin", "B — Septembre à Décembre"])
        saison_val = "A" if saison.startswith("A") else "B"
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-label">🌡️ Conditions Climatiques</div>', unsafe_allow_html=True)
        altitude     = st.slider("Altitude (m)", 700, 2100, 1500, step=10)
        pluviometrie = st.slider("Pluviométrie (mm)", 200, 1300, 850, step=10)
        temperature  = st.slider("Température moyenne (°C)", 15.0, 28.0, 20.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><div class="card-label">🌱 Parcelle & Intrants</div>', unsafe_allow_html=True)
        superficie = st.slider("Superficie cultivée (ha)", 0.3, 5.0, 2.0, step=0.1)
        nb_menages = st.slider("Nombre de ménages", 10, 200, 80)
        st.markdown("<br>", unsafe_allow_html=True)
        engrais    = st.toggle("🧪 Utilisation d'engrais", value=True)
        irrigation = st.toggle("💧 Accès à l'irrigation", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Choix modèle + bouton
    mc1, mc2, mc3 = st.columns([2,1,1])
    with mc1:
        model_choice = st.selectbox(
            "🤖 Algorithme de prédiction",
            list(MODELS.keys()), index=1
        )
    with mc3:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⟶  Lancer la prédiction", use_container_width=True, type="primary")

    # ── Résultats ─────────────────────────────────────────
    if predict_btn:

        # Préparer l'entrée
        data = {
            'province': [province], 'culture': [culture], 'saison': [saison_val],
            'altitude_m': [altitude], 'pluviometrie_mm': [pluviometrie],
            'temperature_moy_C': [temperature], 'superficie_ha': [superficie],
            'utilisation_engrais': [int(engrais)], 'acces_irrigation': [int(irrigation)],
            'nb_menages': [nb_menages],
        }
        df_input = pd.DataFrame(data)
        df_enc   = pd.get_dummies(df_input, columns=['province','culture','saison'], drop_first=False)
        for col in feature_names:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[[c for c in df_enc.columns if c in feature_names]]
        df_enc = df_enc[feature_names]
        df_enc[num_features] = scaler.transform(df_enc[num_features])

        model  = MODELS[model_choice]
        pred   = model.predict(df_enc)[0]
        prob_1 = model.predict_proba(df_enc)[0][1] * 100

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Résultat principal
        r1, r2 = st.columns([1.3, 1], gap="large")

        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-good">
                  <div class="result-tag">Résultat de la prédiction</div>
                  <div class="result-title">✦ Bonne Récolte Prévue</div>
                  <div class="result-prob">{prob_1:.1f}%</div>
                  <div class="result-prob-label">probabilité de bonne récolte</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-bad">
                  <div class="result-tag">Résultat de la prédiction</div>
                  <div class="result-title">⚠ Mauvaise Récolte Prévue</div>
                  <div class="result-prob">{100-prob_1:.1f}%</div>
                  <div class="result-prob-label">probabilité de mauvaise récolte</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            # Jauge matplotlib sobre
            fig, ax = plt.subplots(figsize=(5, 2.4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            color_bar = '#2D6B45' if prob_1 >= 50 else '#B03A2E'
            ax.barh([0], [prob_1],     color=color_bar,  height=0.45, zorder=3)
            ax.barh([0], [100-prob_1], left=[prob_1], color='#EDE4D3', height=0.45, zorder=3)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%','25%','50%','75%','100%'],
                               fontsize=8, color='#6B6355',
                               fontfamily='monospace')
            ax.axvline(50, color='#C0622A', linestyle='--', linewidth=1.2, zorder=4)
            ax.text(prob_1/2 if prob_1 > 12 else prob_1+2, 0,
                    f'{prob_1:.0f}%', ha='center', va='center',
                    color='white' if prob_1 > 12 else color_bar,
                    fontweight='bold', fontsize=13, fontfamily='monospace', zorder=5)
            ax.set_title(f'Confiance · {model_choice}',
                         fontsize=9, color='#6B6355', pad=10, fontfamily='sans-serif')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close()

            # Paramètres saisis résumé
            st.markdown(f"""
            <div style='background:#FAF6EF; border:1px solid #EDE4D3; border-radius:12px;
                 padding:16px 18px; margin-top:14px; font-size:0.82rem;'>
              <div style='font-family:monospace; font-size:0.68rem; letter-spacing:0.1em;
                   text-transform:uppercase; color:#C0622A; margin-bottom:10px;'>
                — Paramètres saisis
              </div>
              <b>{province}</b> · {culture} · Saison {saison_val}<br>
              🏔 {altitude} m &nbsp;|&nbsp; 🌧 {pluviometrie} mm &nbsp;|&nbsp; 🌡 {temperature}°C<br>
              🌿 {superficie} ha &nbsp;|&nbsp; 👥 {nb_menages} ménages<br>
              {'✅ Engrais' if engrais else '❌ Sans engrais'} &nbsp;·&nbsp; {'✅ Irrigation' if irrigation else '❌ Pluvial'}
            </div>""", unsafe_allow_html=True)

        # Comparaison 3 modèles
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-label" style="margin-bottom:16px;">🔄 Comparaison des 3 algorithmes</div>', unsafe_allow_html=True)
        cm1, cm2, cm3 = st.columns(3, gap="medium")
        for col_m, (mname, mmodel) in zip([cm1, cm2, cm3], MODELS.items()):
            mp    = mmodel.predict(df_enc)[0]
            mprob = mmodel.predict_proba(df_enc)[0][1] * 100
            cls   = "good" if mp == 1 else "bad"
            verdict = "Bonne récolte" if mp == 1 else "Mauvaise récolte"
            icon    = "✦" if mp == 1 else "⚠"
            with col_m:
                st.markdown(f"""
                <div class="model-card {cls}">
                  <div class="model-card-name">{mname}</div>
                  <div class="model-card-pct {cls}">{mprob:.1f}%</div>
                  <div class="model-card-verdict {cls}">{icon} {verdict}</div>
                </div>""", unsafe_allow_html=True)

        # Recommandations
        recos = []
        if pluviometrie < 600:
            recos.append("🌧️ Pluviométrie très faible — envisager la collecte d'eau ou l'irrigation complémentaire")
        if not engrais:
            recos.append("🧪 L'utilisation d'engrais peut significativement améliorer le rendement")
        if not irrigation and pluviometrie < 700:
            recos.append("💧 Un accès à l'irrigation est recommandé pour ce niveau de pluviométrie")
        if culture == 'Haricot' and pluviometrie < 650:
            recos.append("🌱 Le Haricot est sensible à la sécheresse — envisager Sorgho ou Manioc comme alternative")
        if temperature > 24 and culture in ['Maïs', 'Haricot']:
            recos.append("🌡️ Température élevée pour cette culture — privilégier des variétés thermotolérantes")

        if recos or pred == 0:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-label" style="margin-bottom:16px;">💡 Recommandations agronomiques</div>', unsafe_allow_html=True)
            if not recos:
                recos.append("⚠️ Conditions limites — surveiller l'évolution et préparer un plan de contingence")
            for r in recos:
                st.markdown(f'<div class="reco-item">{r}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# ONGLET 2 — PERFORMANCES
# ════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<p class="section-title">Performances des modèles</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Évaluation sur le jeu de test (20% des données — 324 observations).</p>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3, gap="large")
    for col_p, (mname, mmet) in zip([pc1, pc2, pc3], metrics.items()):
        f1_0 = mmet['report']['0']['f1-score']
        f1_1 = mmet['report']['1']['f1-score']
        with col_p:
            st.markdown(f"""
            <div class="perf-card">
              <div class="perf-card-name">{mname}</div>
              <div class="perf-acc">{mmet['accuracy']*100:.2f}%</div>
              <div class="perf-acc-label">Accuracy globale</div>
              <div class="perf-row">
                <span class="perf-row-label">AUC-ROC</span>
                <span class="perf-row-val">{mmet['auc']:.4f}</span>
              </div>
              <div class="perf-row">
                <span class="perf-row-label">F1 — Bonne récolte</span>
                <span class="perf-row-val">{f1_1:.3f}</span>
              </div>
              <div class="perf-row">
                <span class="perf-row-label">F1 — Mauvaise récolte</span>
                <span class="perf-row-val">{f1_0:.3f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Graphique comparatif
    mnames = list(metrics.keys())
    short  = ['Arbre (DT)', 'Forêt (RF)', 'Régression (LR)']
    accs   = [metrics[m]['accuracy'] * 100 for m in mnames]
    aucs   = [metrics[m]['auc'] for m in mnames]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor('#FFFFFF')
    COLORS = ['#C0622A', '#1D4A2F', '#C9A84C']

    for ax in axes:
        ax.set_facecolor('#FAFAF8')
        for spine in ax.spines.values():
            spine.set_color('#EDE4D3')

    # Accuracy
    bars1 = axes[0].bar(short, accs, color=COLORS, width=0.5,
                        edgecolor='white', linewidth=1.5, zorder=3)
    axes[0].set_ylim(75, 100)
    axes[0].set_title('Accuracy (%)', fontsize=11, fontweight='bold',
                      color='#1A1A18', pad=14)
    axes[0].set_ylabel('Accuracy (%)', color='#6B6355', fontsize=9)
    axes[0].tick_params(colors='#6B6355')
    axes[0].yaxis.grid(True, color='#EDE4D3', linewidth=0.8, zorder=0)
    axes[0].set_axisbelow(True)
    for b, v in zip(bars1, accs):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                     f'{v:.2f}%', ha='center', fontsize=9,
                     fontweight='bold', color='#1A1A18',
                     fontfamily='monospace')

    # AUC
    bars2 = axes[1].bar(short, aucs, color=COLORS, width=0.5,
                        edgecolor='white', linewidth=1.5, zorder=3)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title('AUC-ROC', fontsize=11, fontweight='bold',
                      color='#1A1A18', pad=14)
    axes[1].set_ylabel('AUC', color='#6B6355', fontsize=9)
    axes[1].tick_params(colors='#6B6355')
    axes[1].yaxis.grid(True, color='#EDE4D3', linewidth=0.8, zorder=0)
    axes[1].set_axisbelow(True)
    axes[1].axhline(0.5, color='#C0622A', linestyle='--',
                    linewidth=1.2, alpha=0.6, zorder=4)
    axes[1].text(2.42, 0.505, 'Aléatoire', fontsize=7.5,
                 color='#C0622A', alpha=0.7, fontfamily='monospace')
    for b, v in zip(bars2, aucs):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                     f'{v:.4f}', ha='center', fontsize=9,
                     fontweight='bold', color='#1A1A18',
                     fontfamily='monospace')

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="info-box">
      <strong>💡 Interprétation :</strong><br>
      La <strong>Forêt Aléatoire</strong> offre la meilleure accuracy globale (93.35%).
      La <strong>Régression Logistique</strong> obtient la meilleure AUC-ROC (0.83) — elle discrimine mieux entre bonnes et mauvaises récoltes.
      L'AUC est la métrique prioritaire ici car le dataset est fortement déséquilibré (93% de bonnes récoltes) :
      une accuracy élevée peut être trompeuse si le modèle prédit toujours la classe majoritaire.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='text-align:center; font-family:monospace; font-size:0.7rem;
         color:#9B8E7E; letter-spacing:0.12em; padding-bottom:32px;'>
      AGRISMART BURUNDI &nbsp;·&nbsp; UNIVERSITÉ POLYTECHNIQUE DE GITEGA
      &nbsp;·&nbsp; BAC 4 GÉNIE LOGICIEL
    </div>
    """, unsafe_allow_html=True)