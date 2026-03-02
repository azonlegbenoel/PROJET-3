"""
app.py — ScreenISE · Plateforme de Recommandation de Films
Auteur : AZONLEGBE Noël Junior Azonsou
Projet : Machine Learning Avancé — ENEAM / ISE
Run: streamlit run app.py
"""

import os, pickle, time, hashlib, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# INDISPENSABLE — pickle doit retrouver la classe ALS
from als_model import ALS

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScreenISE — Recommandation de Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --gold:   #D4AF37; --gold2:  #F0CC5A;
  --dark:   #0A0A0F; --dark2:  #12121A;
  --dark3:  #1C1C28; --dark4:  #242436;
  --border: #2E2E45; --text:   #E8E8F0;
  --muted:  #888899; --teal:   #2DD4BF;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--dark) !important;
  font-family: 'DM Sans', sans-serif; color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--dark2) !important;
  border-right: 1px solid var(--border);
}
h1,h2,h3 { font-family:'Playfair Display',serif; color:var(--gold); }
h4,h5,h6 { font-family:'DM Sans',sans-serif; color:var(--text); }

.stButton > button {
  background: linear-gradient(135deg,var(--gold),var(--gold2)) !important;
  color: var(--dark) !important; font-family:'DM Sans',sans-serif;
  font-weight:600; border:none !important; border-radius:8px !important;
  padding:0.5rem 1.4rem !important; transition:all .2s ease; letter-spacing:.5px;
}
.stButton > button:hover {
  transform:translateY(-2px);
  box-shadow:0 6px 20px rgba(212,175,55,.35) !important;
}

[data-testid="stTextInput"] input, .stTextInput input {
  background:var(--dark3) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; border-radius:8px !important;
}

[data-testid="metric-container"] {
  background:var(--dark3); border:1px solid var(--border);
  border-radius:12px; padding:1rem;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color:var(--gold) !important; font-family:'Playfair Display',serif; font-size:2rem !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background:var(--dark2); border-radius:10px; gap:4px; padding:4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background:transparent !important; color:var(--muted) !important; border-radius:8px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background:var(--dark4) !important; color:var(--gold) !important;
}

hr { border-color:var(--border); }

.film-card {
  background:var(--dark3); border:1px solid var(--border); border-radius:14px;
  padding:1rem; margin-bottom:.5rem; transition:all .2s ease;
}
.film-card:hover { border-color:var(--gold); box-shadow:0 0 18px rgba(212,175,55,.12); }
.film-title  { font-weight:600; font-size:.95rem; color:var(--text); margin-bottom:.2rem; }
.film-genre  { font-size:.75rem; color:var(--muted); }
.film-score  { font-family:'Playfair Display',serif; color:var(--gold); font-size:1.1rem; }
.rec-badge   {
  background:linear-gradient(135deg,var(--gold),var(--gold2)); color:var(--dark);
  font-size:.65rem; font-weight:700; padding:2px 8px; border-radius:20px;
  letter-spacing:1px; text-transform:uppercase;
}
.star-rating { font-size:1.1rem; color:var(--gold); }
.section-title {
  font-family:'Playfair Display',serif; font-size:1.6rem; color:var(--gold);
  margin:1.2rem 0 .8rem; border-left:4px solid var(--gold); padding-left:.8rem;
}
.hero-banner {
  background:linear-gradient(135deg,#0A0A0F 0%,#1C1228 50%,#0A0A0F 100%);
  border:1px solid var(--border); border-radius:16px; padding:2rem; margin-bottom:1.5rem;
}
.badge-algo {
  display:inline-block; background:rgba(45,212,191,.15);
  border:1px solid var(--teal); color:var(--teal);
  font-size:.72rem; font-weight:600; padding:3px 10px; border-radius:20px;
  letter-spacing:1px; text-transform:uppercase; margin-right:6px;
}
.input-label {
  font-size:.8rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:1px; margin-bottom:.3rem;
}

/* LOGIN page */
[data-testid="stAppViewContainer"].login-mode .block-container {
  max-width:480px !important;
}

.block-container { padding-top:1.5rem !important; }
[data-testid="stHeader"] { background:transparent !important; }
footer { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "./models"
USERS_FILE = "./data/users_db.json"
PLATFORM   = "ScreenISE"
ISE_LOGO   = "https://ise-eneam.org/storage/logo-ise-eneam.png"
AUTHOR     = "AZONLEGBE Noël Junior Azonsou"

# ──────────────────────────────────────────────────────────────────────────────
# BASE UTILISATEURS (JSON local)
# ──────────────────────────────────────────────────────────────────────────────
def _hash(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def load_users() -> dict:
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(db: dict):
    os.makedirs("./data", exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def register_user(username: str, password: str, display_name: str):
    db    = load_users()
    uname = username.strip().lower()
    if not uname or not password:
        return False, "Identifiant et mot de passe requis."
    if len(password) < 4:
        return False, "Mot de passe trop court (min. 4 caractères)."
    if uname in db:
        return False, "Cet identifiant est déjà utilisé."
    db[uname] = {
        "password_hash": _hash(password),
        "display_name":  display_name.strip() or username.strip(),
        "created_at":    time.strftime("%Y-%m-%d %H:%M"),
    }
    save_users(db)
    return True, "Compte créé avec succès !"

def authenticate(username: str, password: str):
    db    = load_users()
    uname = username.strip().lower()
    if uname not in db:
        return False, "Identifiant introuvable."
    if db[uname]["password_hash"] != _hash(password):
        return False, "Mot de passe incorrect."
    return True, db[uname]["display_name"]

# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    required = ["als_model.pkl","movies.pkl","ratings.pkl","metrics.pkl","popular.pkl"]
    if any(not os.path.exists(os.path.join(MODEL_DIR, f)) for f in required):
        return None, None, None, None, None
    with open(os.path.join(MODEL_DIR,"als_model.pkl"),"rb") as f:
        model = pickle.load(f)
    movies  = pd.read_pickle(os.path.join(MODEL_DIR,"movies.pkl"))
    ratings = pd.read_pickle(os.path.join(MODEL_DIR,"ratings.pkl"))
    popular = pd.read_pickle(os.path.join(MODEL_DIR,"popular.pkl"))
    with open(os.path.join(MODEL_DIR,"metrics.pkl"),"rb") as f:
        metrics = pickle.load(f)
    if "genres" not in movies.columns:
        movies["genres"] = "Unknown"
    return model, movies, ratings, metrics, popular

# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────
def stars(rating: float) -> str:
    n = int(round(rating))
    return "★"*n + "☆"*(5-n)

def genre_badge(genres) -> str:
    if pd.isna(genres) or str(genres) in ("Unknown",""):
        return ""
    return " · ".join(str(genres).split("|")[:2])

def get_recommendations(model, movies_df, user_ratings: dict, top_k: int = 12):
    known = {model.item_map[mid]: r
             for mid, r in user_ratings.items() if mid in model.item_map}
    if not known:
        return None
    idx  = np.array(list(known.keys()))
    vals = np.array(list(known.values()), dtype=np.float64)
    V    = model.item_factors[idx]
    lI   = model.reg * np.eye(model.n_factors)
    try:
        uf = np.linalg.solve(V.T @ V + lI, V.T @ vals)
    except np.linalg.LinAlgError:
        return None
    scores = model.item_factors @ uf
    for s in set(idx):
        scores[s] = -np.inf
    rev = {v: k for k, v in model.item_map.items()}
    recs = []
    for ii in np.argsort(scores)[::-1][:top_k]:
        mid  = rev[ii]
        sc   = float(np.clip(scores[ii], 1.0, 5.0))
        row  = movies_df[movies_df["movieId"] == mid]
        if row.empty:
            continue
        recs.append({"movieId": mid, "title": row["title"].values[0],
                     "genres":  row["genres"].values[0] if "genres" in row.columns else "Unknown",
                     "pred_score": sc})
    return pd.DataFrame(recs)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
for k, v in [("logged_in", False), ("username",""), ("display_name",""),
              ("user_ratings",{}), ("recs",None), ("search_results",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# ████  PAGE DE CONNEXION
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    [data-testid="stSidebar"]          { display:none !important; }
    [data-testid="collapsedControl"]   { display:none !important; }
    .block-container { max-width:500px !important; margin:auto; padding-top:1.5rem !important; }
    </style>""", unsafe_allow_html=True)

    # ── En-tête logo + titre ──
    st.markdown(f"""
    <div style="text-align:center; padding:1.2rem 0 .5rem;">
      <img src="{ISE_LOGO}"
           style="height:72px; object-fit:contain; margin-bottom:.8rem; border-radius:8px;"
           onerror="this.style.display='none'"/>
      <div style="font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900;
                  color:#D4AF37; letter-spacing:3px; line-height:1.1;">
        {PLATFORM}
      </div>
      <div style="color:#888899; font-size:.78rem; letter-spacing:3px; text-transform:uppercase; margin-top:6px;">
        Recommandation de Films
      </div>
      <div style="margin-top:.5rem; color:#444455; font-size:.72rem;">
        ENEAM / ISE &nbsp;·&nbsp; Machine Learning Avancé
      </div>
    </div>
    <hr style="border-color:#2E2E45; margin:1rem 0 1.5rem;">
    """, unsafe_allow_html=True)

    # ── Onglets Connexion / Créer un compte ──
    tab_l, tab_r = st.tabs(["🔑  Connexion", "✨  Créer un compte"])

    # ── Connexion ──────────────────────────────
    with tab_l:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Identifiant</div>", unsafe_allow_html=True)
        li_user = st.text_input("id_l", placeholder="votre identifiant",
                                label_visibility="collapsed", key="li_user")

        st.markdown("<div class='input-label' style='margin-top:.7rem;'>Mot de passe</div>",
                    unsafe_allow_html=True)
        li_pass = st.text_input("pw_l", placeholder="••••••••", type="password",
                                label_visibility="collapsed", key="li_pass")

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        if st.button("Se connecter", use_container_width=True, key="btn_login"):
            if li_user and li_pass:
                ok, msg = authenticate(li_user, li_pass)
                if ok:
                    st.session_state.logged_in    = True
                    st.session_state.username     = li_user.strip().lower()
                    st.session_state.display_name = msg
                    st.success(f"Bienvenue, **{msg}** !")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Remplissez tous les champs.")

    # ── Créer un compte ────────────────────────
    with tab_r:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Prénom & Nom</div>", unsafe_allow_html=True)
        rg_name = st.text_input("nm_r", placeholder="Ex : Jean Dupont",
                                label_visibility="collapsed", key="rg_name")

        st.markdown("<div class='input-label' style='margin-top:.7rem;'>Identifiant</div>",
                    unsafe_allow_html=True)
        rg_user = st.text_input("id_r", placeholder="Choisissez un identifiant unique",
                                label_visibility="collapsed", key="rg_user")

        st.markdown("<div class='input-label' style='margin-top:.7rem;'>Mot de passe</div>",
                    unsafe_allow_html=True)
        rg_pass = st.text_input("pw_r", placeholder="Min. 4 caractères", type="password",
                                label_visibility="collapsed", key="rg_pass")

        st.markdown("<div class='input-label' style='margin-top:.7rem;'>Confirmer le mot de passe</div>",
                    unsafe_allow_html=True)
        rg_pass2 = st.text_input("pw2_r", placeholder="Répétez le mot de passe", type="password",
                                 label_visibility="collapsed", key="rg_pass2")

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            if rg_pass != rg_pass2:
                st.error("Les mots de passe ne correspondent pas.")
            else:
                ok, msg = register_user(rg_user, rg_pass, rg_name)
                if ok:
                    st.success(msg + " Connectez-vous maintenant dans l'onglet Connexion.")
                else:
                    st.error(msg)

    # Footer auteur
    st.markdown(f"""
    <hr style="border-color:#1E1E2E; margin:2rem 0 .8rem;">
    <div style="text-align:center; color:#444455; font-size:.72rem; line-height:1.7;">
      Développé par <strong style="color:#666677;">{AUTHOR}</strong><br>
      Machine Learning Avancé &nbsp;·&nbsp; ENEAM / ISE &nbsp;·&nbsp; 2025
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ████  APP PRINCIPALE (post-auth)
# ══════════════════════════════════════════════════════════════════════════════
model, movies, ratings_df, metrics, popular = load_artefacts()
data_ready = model is not None

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    initials = "".join(p[0].upper() for p in st.session_state.display_name.split()[:2]) or "U"

    st.markdown(f"""
    <div style="text-align:center; padding:.6rem 0 .4rem;">
      <img src="{ISE_LOGO}"
           style="height:52px; object-fit:contain; margin-bottom:.5rem;"
           onerror="this.style.display='none'"/>
      <div style="font-family:'Playfair Display',serif; font-size:1.65rem;
                  color:#D4AF37; font-weight:900; letter-spacing:2px;">
        🎬 {PLATFORM}
      </div>
      <div style="color:#888899; font-size:.72rem; letter-spacing:2px;
                  text-transform:uppercase; margin-top:3px;">
        Recommandation de Films
      </div>
      <div style="margin-top:.5rem; padding:.3rem .8rem; background:rgba(212,175,55,.08);
                  border-radius:6px; border:1px solid rgba(212,175,55,.25); display:inline-block;">
        <span style="font-size:.66rem; color:#888899;">ENEAM — ISE</span>
      </div>
    </div>
    <hr style="border-color:#2E2E45; margin:.7rem 0;">
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation",
                   ["🏠  Accueil","🔍  Rechercher","⭐  Mes Films","📊  Statistiques","🤖  Modèle ALS"],
                   label_visibility="collapsed")
    page = nav.split("  ")[1]

    st.markdown("<hr style='border-color:#2E2E45;'>", unsafe_allow_html=True)

    # Carte utilisateur
    st.markdown(f"""
    <div style="background:#1C1C28; border:1px solid #2E2E45; border-radius:12px;
                padding:.75rem 1rem; display:flex; align-items:center; gap:.75rem;">
      <div style="width:38px; height:38px; background:linear-gradient(135deg,#D4AF37,#F0CC5A);
                  border-radius:50%; display:flex; align-items:center; justify-content:center;
                  font-weight:700; font-size:.88rem; color:#0A0A0F; flex-shrink:0;">
        {initials}
      </div>
      <div>
        <div style="font-weight:600; font-size:.87rem; color:#E8E8F0; line-height:1.2;">
          {st.session_state.display_name}
        </div>
        <div style="font-size:.7rem; color:#888899;">@{st.session_state.username}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    n_rated = len(st.session_state.user_ratings)
    if n_rated > 0:
        st.markdown(f"""
        <div style="margin-top:.5rem; background:rgba(212,175,55,.08);
                    border:1px solid rgba(212,175,55,.2); border-radius:10px;
                    padding:.45rem 1rem; text-align:center;">
          <span style="color:#D4AF37; font-size:1.1rem; font-weight:700;">{n_rated}</span>
          <span style="color:#888899; font-size:.76rem; margin-left:4px;">film(s) noté(s)</span>
        </div>
        """, unsafe_allow_html=True)

    if not data_ready:
        st.warning("⚠️ Modèle non trouvé.\nLancez `python train_als.py`")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("🚪  Se déconnecter", use_container_width=True):
        for k in ["logged_in","username","display_name","user_ratings","recs","search_results"]:
            st.session_state[k] = False if k == "logged_in" else \
                                   {} if k == "user_ratings" else \
                                   "" if k in ("username","display_name") else None
        st.rerun()

    st.markdown(f"""
    <div style="margin-top:1rem; padding-top:.8rem; border-top:1px solid #2E2E45;
                font-size:.67rem; color:#3A3A55; text-align:center; line-height:1.7;">
      {AUTHOR}<br>ML Avancé · ENEAM/ISE · 2025
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ████  ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "Accueil":

    st.markdown(f"""
    <div class="hero-banner">
      <div style="display:flex; align-items:center; gap:1rem; margin-bottom:.5rem;">
        <span style="font-family:'Playfair Display',serif; font-size:2.4rem;
                     color:#D4AF37; font-weight:900;">{PLATFORM}</span>
        <span class="badge-algo">ALS Engine</span>
        <span class="badge-algo">MovieLens 1M</span>
      </div>
      <p style="color:#C0C0D0; font-size:1.05rem; max-width:600px; line-height:1.6;">
        Bonjour <strong style="color:#D4AF37;">{st.session_state.display_name}</strong> !
        Découvrez votre prochain film grâce à notre moteur de recommandation par
        <strong style="color:#D4AF37;">Factorisation de Matrice ALS</strong>.
        Notez un film — obtenez des suggestions personnalisées instantanément.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.recs is not None and not st.session_state.recs.empty:
        st.markdown('<div class="section-title">✨ Recommandé pour vous</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for idx, row in st.session_state.recs.iterrows():
            with cols[idx % 4]:
                sc = row["pred_score"]
                g  = genre_badge(row["genres"])
                st.markdown(f"""
                <div class="film-card">
                  <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div class="film-title">{row['title']}</div>
                    <span class="rec-badge">Rec</span>
                  </div>
                  <div class="film-genre">{g}</div>
                  <div style="margin-top:.5rem; display:flex; justify-content:space-between;">
                    <span class="star-rating">{stars(sc)}</span>
                    <span class="film-score">{sc:.1f}/5</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown('<div class="section-title">🔥 Films Tendances</div>', unsafe_allow_html=True)
    if data_ready:
        cols = st.columns(4)
        for idx, row in popular.head(16).iterrows():
            with cols[idx % 4]:
                sc = float(row["bayes_score"])
                g  = genre_badge(row.get("genres",""))
                st.markdown(f"""
                <div class="film-card">
                  <div class="film-title">{row['title']}</div>
                  <div class="film-genre">{g}</div>
                  <div style="margin-top:.5rem; display:flex; justify-content:space-between; align-items:center;">
                    <span class="star-rating">{stars(round(sc))}</span>
                    <span style="font-size:.75rem; color:#888899;">{int(row['count']):,} votes</span>
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Entraînez d'abord le modèle pour afficher les films populaires.")

# ══════════════════════════════════════════════════════════════════════════════
# ████  RECHERCHER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Rechercher":

    st.markdown('<div class="section-title">🔍 Rechercher un Film</div>', unsafe_allow_html=True)
    if not data_ready:
        st.error("Modèle non chargé. Exécutez `python train_als.py` d'abord.")
    else:
        query = st.text_input("Titre du film …",
                              placeholder="Ex: Matrix, Titanic, The Godfather …",
                              key="search_box")
        if query and len(query) >= 2:
            mask = movies["title"].str.lower().str.contains(query.lower(), na=False)
            st.session_state.search_results = movies[mask].head(20)
        elif not query:
            st.session_state.search_results = None

        results = st.session_state.search_results

        def _rate_row(row, key_suffix):
            mid   = int(row["movieId"])
            title = row["title"]
            g     = genre_badge(row.get("genres",""))
            ci, cs, cb = st.columns([4, 2, 1.5])
            with ci:
                badge = '<span class="rec-badge">✓ Noté</span>' \
                        if mid in st.session_state.user_ratings else ""
                st.markdown(f"""
                <div style="padding:.55rem 0;">
                  <div style="font-weight:600;color:#E8E8F0;">{title} {badge}</div>
                  <div style="font-size:.78rem;color:#888899;">{g}</div>
                </div>""", unsafe_allow_html=True)
            with cs:
                rv = st.select_slider("Note", options=[1,2,3,4,5],
                                      value=st.session_state.user_ratings.get(mid, 3),
                                      format_func=lambda x: "★"*x,
                                      key=f"sl_{key_suffix}_{mid}",
                                      label_visibility="collapsed")
            with cb:
                if st.button("Valider", key=f"bt_{key_suffix}_{mid}"):
                    st.session_state.user_ratings[mid] = rv
                    recs = get_recommendations(model, movies, st.session_state.user_ratings)
                    if recs is None:
                        recs = popular[["movieId","title"]].copy()
                        recs["genres"] = popular.get("genres","Unknown")
                        recs["pred_score"] = popular["bayes_score"].values[:len(recs)]
                    st.session_state.recs = recs
                    st.success(f"✓ '{title}' noté {rv}★")
                    time.sleep(0.35)
                    st.rerun()
            st.markdown("<div style='border-bottom:1px solid #2E2E45;'></div>",
                        unsafe_allow_html=True)

        if results is not None and len(results) == 0:
            st.warning("Aucun film trouvé.")
        elif results is not None:
            st.markdown(f"<div style='color:#888899;font-size:.85rem;margin-bottom:.8rem;'>"
                        f"{len(results)} résultat(s)</div>", unsafe_allow_html=True)
            for _, row in results.iterrows():
                _rate_row(row, "srch")
        else:
            st.markdown('<div class="section-title">🎲 Films Populaires à Découvrir</div>',
                        unsafe_allow_html=True)
            for _, row in popular.head(10).iterrows():
                _rate_row(row, "pop")

# ══════════════════════════════════════════════════════════════════════════════
# ████  MES FILMS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Mes Films":

    st.markdown('<div class="section-title">⭐ Mon Historique & Profil</div>', unsafe_allow_html=True)
    ur = st.session_state.user_ratings

    if not ur:
        st.info("Vous n'avez encore noté aucun film. Allez dans **Rechercher** pour commencer !")
    else:
        rows = []
        for mid, r in ur.items():
            mv = movies[movies["movieId"] == mid]
            rows.append({
                "movieId":     mid,
                "Titre":       mv["title"].values[0]  if not mv.empty else f"ID {mid}",
                "Genres":      mv["genres"].values[0] if (not mv.empty and "genres" in mv.columns) else "Unknown",
                "Votre Note":  r,
            })
        hist = pd.DataFrame(rows).sort_values("Votre Note", ascending=False)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Films Notés",  len(hist))
        c2.metric("Note Moyenne", f"{hist['Votre Note'].mean():.2f} ★")
        c3.metric("Note Max",     f"{hist['Votre Note'].max()} ★")
        c4.metric("Note Min",     f"{hist['Votre Note'].min()} ★")
        st.markdown("---")

        cc1, cc2 = st.columns(2)
        with cc1:
            fig1 = px.histogram(hist, x="Votre Note", nbins=5,
                                title="Distribution de vos Notes",
                                color_discrete_sequence=["#D4AF37"],
                                template="plotly_dark")
            fig1.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                               font=dict(color="#E8E8F0"), title_font_color="#D4AF37",
                               xaxis=dict(tickvals=[1,2,3,4,5],title="Note"),
                               yaxis=dict(title="Films"), showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with cc2:
            gc: dict = defaultdict(int)
            for g in hist["Genres"]:
                if not pd.isna(g) and g != "Unknown":
                    for gg in g.split("|"):
                        gc[gg.strip()] += 1
            if gc:
                gdf = pd.DataFrame(list(gc.items()),
                                   columns=["Genre","Nb"]).sort_values("Nb",ascending=False).head(8)
                fig2 = px.bar(gdf, x="Nb", y="Genre", orientation="h",
                              title="Vos Genres Préférés",
                              color="Nb", color_continuous_scale="Oranges",
                              template="plotly_dark")
                fig2.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                                   font=dict(color="#E8E8F0"), title_font_color="#D4AF37",
                                   coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📋 Détail de votre historique")

        # Reset index pour avoir des indices propres (0, 1, 2, ...)
        hist = hist.reset_index(drop=True)

        to_delete = None
        for i in range(len(hist)):
            mid        = int(hist.at[i, "movieId"])
            titre      = hist.at[i, "Titre"]
            genres_str = hist.at[i, "Genres"] if hist.at[i, "Genres"] != "Unknown" else "—"
            note_int   = int(hist.at[i, "Votre Note"])
            note_stars = "★" * note_int + "☆" * (5 - note_int)

            with st.container():
                ct, cg, cr, cd = st.columns([4, 3, 2, 1])
                ct.write(f"**{titre}**")
                cg.caption(genres_str)
                cr.write(note_stars)
                if cd.button("🗑", key=f"del_{mid}_{i}"):
                    to_delete = mid
            st.divider()

        if to_delete is not None:
            del st.session_state.user_ratings[to_delete]
            st.rerun()

        col_btn, _ = st.columns([2, 4])
        if col_btn.button("🗑️  Effacer tout l'historique"):
            st.session_state.user_ratings = {}
            st.session_state.recs = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ████  STATISTIQUES (sans répartition des genres)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Statistiques":

    st.markdown('<div class="section-title">📊 Statistiques du Dataset MovieLens 1M</div>',
                unsafe_allow_html=True)
    if not data_ready:
        st.error("Données non disponibles. Entraînez le modèle d'abord.")
    else:
        m = metrics
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Utilisateurs", f"{m['n_users']:,}")
        k2.metric("Films",        f"{m['n_items']:,}")
        k3.metric("Notations",    f"{m['n_ratings']:,}")
        k4.metric("Sparsité",     f"{m['sparsity']*100:.2f}%")
        st.markdown("---")

        # Ligne 1
        c1, c2 = st.columns(2)
        with c1:
            rc = ratings_df["rating"].value_counts().sort_index()
            fig = px.bar(x=rc.index, y=rc.values,
                         labels={"x":"Note","y":"Nombre de notations"},
                         title="Distribution Globale des Notes",
                         color=rc.values, color_continuous_scale="Oranges",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                              font=dict(color="#E8E8F0"), title_font_color="#D4AF37",
                              coloraxis_showscale=False, xaxis=dict(tickvals=[1,2,3,4,5]))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cpm = ratings_df.groupby("movieId").size()
            fig2 = px.histogram(x=cpm, nbins=50,
                                labels={"x":"Nombre de notations","y":"Films"},
                                title="Films selon leur Popularité",
                                color_discrete_sequence=["#D4AF37"],
                                template="plotly_dark", log_y=True)
            fig2.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                               font=dict(color="#E8E8F0"), title_font_color="#D4AF37")
            st.plotly_chart(fig2, use_container_width=True)

        # Ligne 2
        c3, c4 = st.columns(2)
        with c3:
            cpu = ratings_df.groupby("userId").size()
            fig3 = px.histogram(x=cpu, nbins=60,
                                labels={"x":"Notations par utilisateur","y":"Utilisateurs"},
                                title="Activité par Utilisateur",
                                color_discrete_sequence=["#2DD4BF"],
                                template="plotly_dark", log_y=True)
            fig3.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                               font=dict(color="#E8E8F0"), title_font_color="#D4AF37")
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            top10 = (ratings_df.groupby("movieId")
                               .agg(count=("rating","count"), mean=("rating","mean"))
                               .reset_index()
                               .sort_values("count", ascending=False)
                               .head(10)
                               .merge(movies[["movieId","title"]], on="movieId"))
            fig4 = px.bar(top10, x="count", y="title", orientation="h",
                          title="Top 10 Films les Plus Notés",
                          color="mean", color_continuous_scale="RdYlGn",
                          template="plotly_dark",
                          labels={"count":"Nb Notations","title":"","mean":"Note Moy"})
            fig4.update_layout(paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                               font=dict(color="#E8E8F0"), title_font_color="#D4AF37",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ████  MODÈLE ALS — performances uniquement
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Modèle ALS":

    st.markdown('<div class="section-title">🤖 Performances du Modèle ALS</div>',
                unsafe_allow_html=True)
    if not data_ready:
        st.error("Modèle non entraîné. Lancez `python train_als.py` d'abord.")
    else:
        m = metrics

        st.markdown("#### ⚙️ Hyperparamètres")
        p1,p2,p3 = st.columns(3)
        p1.metric("Facteurs latents (K)", m["n_factors"])
        p2.metric("Itérations ALS",       m["n_iter"])
        p3.metric("Régularisation λ",     m["reg"])

        st.markdown("---")
        st.markdown("#### 📈 Métriques de Performance")
        r1,r2,r3 = st.columns(3)
        r1.metric("RMSE Validation",      f"{m['val_rmse']:.4f}")
        r2.metric("Note Moyenne Dataset", f"{m['global_mean']:.2f} ★")
        r3.metric("Sparsité Matrice",     f"{m['sparsity']*100:.2f}%")

        st.markdown("---")
        if m.get("train_rmse"):
            st.markdown("#### 📉 Courbe d'Apprentissage")
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=list(range(1, len(m["train_rmse"])+1)), y=m["train_rmse"],
                mode="lines+markers", name="Train RMSE",
                line=dict(color="#D4AF37", width=2.5),
                marker=dict(size=6, color="#D4AF37"),
            ))
            fig_lc.add_hline(y=m["val_rmse"], line_dash="dash", line_color="#2DD4BF",
                             annotation_text=f"Val RMSE = {m['val_rmse']:.4f}",
                             annotation_font_color="#2DD4BF")
            fig_lc.update_layout(
                paper_bgcolor="#1C1C28", plot_bgcolor="#1C1C28",
                font=dict(color="#E8E8F0"), title_font_color="#D4AF37",
                xaxis=dict(title="Itération", gridcolor="#2E2E45"),
                yaxis=dict(title="RMSE",      gridcolor="#2E2E45"),
                template="plotly_dark",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

            st.markdown("#### 📋 RMSE par Itération")
            rmse_df = pd.DataFrame({
                "Itération": range(1, len(m["train_rmse"])+1),
                "Train RMSE": [f"{v:.4f}" for v in m["train_rmse"]],
            })
            st.dataframe(rmse_df.set_index("Itération").T, use_container_width=True)

    # Footer
    st.markdown(f"""
    <div style="text-align:center; padding:2rem 0 .5rem; color:#444455;
                font-size:.72rem; border-top:1px solid #2E2E45; margin-top:2rem;">
      <strong style="color:#666677;">{AUTHOR}</strong> ·
      Machine Learning Avancé · ENEAM / ISE · 2025
    </div>
    """, unsafe_allow_html=True)