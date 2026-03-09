import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor,
                               AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bank Finance System | AI Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
#  GLOBAL CSS  — Premium Bank Theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* ── App background ── */
.stApp { background: #F4F7FB; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg,#03152E 0%,#062040 45%,#031528 100%) !important;
    border-right: 2px solid #00C896;
}
section[data-testid="stSidebar"] * { color: #C8DCF0 !important; }
section[data-testid="stSidebar"] .stRadio > label { display:none; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding:10px 16px; border-radius:10px; font-size:0.92rem;
    font-weight:500; transition:all .2s; cursor:pointer;
    margin-bottom:3px; display:block;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background:rgba(0,200,150,.18) !important; color:#fff !important;
}

/* ── Top header ── */
.bank-header {
    background: linear-gradient(135deg,#03152E 0%,#0A3060 55%,#03152E 100%);
    border-radius:20px; padding:32px 42px; margin-bottom:28px;
    border:1px solid rgba(0,200,150,.3);
    box-shadow:0 8px 40px rgba(3,21,46,.4);
    position:relative; overflow:hidden;
}
.bank-header::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(0,200,150,.14) 0%,transparent 70%);
    border-radius:50%;
}
.bank-header h1 {
    color:#fff !important; font-size:1.9rem; font-weight:800;
    margin:0 0 6px 0; letter-spacing:-.6px;
}
.bank-header p { color:rgba(200,220,240,.78) !important; font-size:.95rem; margin:0; }
.bh-badge {
    display:inline-block; background:linear-gradient(135deg,#00C896,#00A07A);
    color:#03152E !important; padding:3px 14px; border-radius:20px;
    font-size:.7rem; font-weight:800; letter-spacing:1.2px;
    text-transform:uppercase; margin-bottom:12px;
}

/* ── KPI strip ── */
.kpi-row { display:flex; gap:16px; margin-bottom:24px; }
.kpi-card {
    flex:1; background:#fff; border-radius:14px;
    padding:18px 20px; border:1px solid #E2EAF4;
    box-shadow:0 2px 12px rgba(3,21,46,.07);
    border-left:4px solid var(--kpi-color,#00C896);
}
.kpi-val { font-size:1.6rem; font-weight:800; color:#03152E;
    font-family:'JetBrains Mono',monospace; }
.kpi-lbl { font-size:.74rem; color:#6B80A0; text-transform:uppercase;
    letter-spacing:.8px; margin-top:3px; }

/* ── Section title ── */
.sec-t {
    font-size:1.02rem; font-weight:700; color:#03152E;
    padding:8px 0 5px; border-bottom:2.5px solid #00C896; margin-bottom:14px;
}

/* ── Condition check box ── */
.cond-wrap {
    background:#fff; border:1px solid #E2EAF4; border-radius:14px;
    padding:22px 26px; box-shadow:0 2px 10px rgba(3,21,46,.06); margin-bottom:18px;
}
.cond-pass { color:#1A8745; font-weight:600; font-size:.9rem; }
.cond-fail { color:#C0392B; font-weight:600; font-size:.9rem; }

/* ── Result cards ── */
.card-approved {
    background:linear-gradient(135deg,#03152E 0%,#064A28 100%);
    border:2px solid #1A8745; border-radius:20px;
    padding:32px; text-align:center;
    box-shadow:0 8px 32px rgba(26,135,69,.3);
}
.card-rejected {
    background:linear-gradient(135deg,#2B0000 0%,#180000 100%);
    border:2px solid #C0392B; border-radius:20px;
    padding:32px; text-align:center;
    box-shadow:0 8px 32px rgba(192,57,43,.3);
}
.res-big { font-size:2.2rem; font-weight:800; letter-spacing:-1px; }
.res-sub { font-size:.94rem; opacity:.78; margin:8px 0 22px; }

/* ── Metric grid inside result card ── */
.mg { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-top:18px; }
.mb {
    background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.13);
    border-radius:13px; padding:18px; text-align:center;
}
.mv { font-size:1.65rem; font-weight:700; display:block;
    font-family:'JetBrains Mono',monospace; }
.ml { font-size:.7rem; color:rgba(255,255,255,.55);
    text-transform:uppercase; letter-spacing:.9px; margin-top:3px; }

/* ── Info / Warn / Success boxes ── */
.ib { background:rgba(0,200,150,.07); border:1px solid rgba(0,200,150,.28);
    border-radius:11px; padding:13px 17px; margin:9px 0; font-size:.87rem; color:#03152E; }
.wb { background:rgba(192,57,43,.07); border:1px solid rgba(192,57,43,.27);
    border-radius:11px; padding:13px 17px; margin:9px 0; font-size:.87rem; color:#03152E; }
.yb { background:rgba(243,156,18,.07); border:1px solid rgba(243,156,18,.27);
    border-radius:11px; padding:13px 17px; margin:9px 0; font-size:.87rem; color:#03152E; }

/* ── Offer box ── */
.ob {
    background:linear-gradient(135deg,#131B2E,#1A2540);
    border:1px solid rgba(255,184,0,.38); border-radius:15px; padding:22px;
}
.ob-t { color:#FFB800; font-weight:700; font-size:.98rem; margin-bottom:10px; }
.lb {
    display:inline-block; background:rgba(255,184,0,.12);
    border:1px solid rgba(255,184,0,.35); color:#FFB800;
    padding:5px 13px; border-radius:7px; font-size:.82rem; font-weight:600; margin:3px;
}

/* ── About cards ── */
.ac {
    background:#fff; border:1px solid #E2EAF4; border-radius:16px;
    padding:24px; box-shadow:0 2px 12px rgba(3,21,46,.07); margin-bottom:18px;
}
.ac h3 { color:#03152E; font-size:1.05rem; font-weight:700; margin-bottom:8px; }
.ac p  { color:#4A6080; font-size:.87rem; line-height:1.7; margin:0; }

/* ── Divider ── */
.dv { height:1px; background:linear-gradient(90deg,transparent,#00C896,transparent);
    margin:22px 0; border:none; }

/* ── Algorithm badge ── */
.algo-badge {
    display:inline-block; padding:5px 14px; border-radius:8px;
    font-size:.78rem; font-weight:700; letter-spacing:.5px; margin:4px 2px;
}

/* ── Table style ── */
.styled-table { width:100%; border-collapse:collapse; font-size:.85rem; }
.styled-table th { background:#03152E; color:#fff; padding:10px 14px;
    text-align:left; font-weight:600; }
.styled-table td { padding:9px 14px; border-bottom:1px solid #E2EAF4; color:#2D4060; }
.styled-table tr:hover td { background:#F0F6FF; }

/* ── Button ── */
.stButton > button {
    background:linear-gradient(135deg,#00C896,#00A07A) !important;
    color:#03152E !important; font-weight:700 !important; font-size:1rem !important;
    border:none !important; border-radius:12px !important;
    padding:14px 30px !important; width:100% !important;
    box-shadow:0 4px 16px rgba(0,200,150,.38) !important;
    letter-spacing:.3px !important; transition:all .2s !important;
}
.stButton > button:hover { transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(0,200,150,.48) !important; }

/* ── Sidebar brand ── */
.sb-brand { text-align:center; padding:20px 8px 26px; margin-bottom:18px;
    border-bottom:1px solid rgba(0,200,150,.2); }
.sb-brand h2 { color:#fff !important; font-size:1.3rem; font-weight:800;
    margin:8px 0 3px; }
.sb-brand p  { color:rgba(200,220,240,.5) !important; font-size:.76rem; }
.sb-nav-head { font-size:.7rem; text-transform:uppercase; letter-spacing:1.5px;
    color:rgba(200,220,240,.45) !important; padding:4px 0 6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATASET + MODEL BUILDERS  (each loan → its BEST algorithm)
# ═══════════════════════════════════════════════════════════════

# ── PERSONAL LOAN  →  Gradient Boosting (best for credit-scoring) ──
@st.cache_resource
def build_personal():
    rng=np.random.RandomState(42); n=4000
    age=rng.randint(21,65,n); city=rng.choice(['Metro','Urban','Semi-Urban'],n)
    emp=rng.choice(['Salaried','Self-Employed','Business Owner'],n)
    income=rng.randint(15000,200000,n); other=rng.randint(0,30000,n)
    bal=rng.randint(5000,500000,n); jt=rng.randint(1,240,n)
    cibil=rng.randint(300,900,n); defs=rng.choice([0,0,0,0,1,2],n)
    inq=rng.randint(0,8,n); emi=rng.randint(0,50000,n)
    lreq=rng.randint(50000,2500000,n); tm=rng.choice([12,24,36,48,60,72,84],n)
    ins=rng.choice([0,1],n); co=rng.choice([0,1],n)
    ti=income+other; dti=(emi+lreq/tm)/(ti+1)
    sc=((cibil>=650)*4+(defs==0)*3+(dti<=.5)*2+(income>=20000)*2+(inq<=3)+(bal>10000)+rng.randn(n)*.3)
    status=(sc>=9).astype(int)
    amt=np.where(status==1,np.minimum(lreq*rng.uniform(.65,1,n),income*30),0)
    rate=np.where(status==1,np.clip(16-(cibil-300)/80+rng.randn(n)*.4,8,22),0)
    le_c=LabelEncoder(); le_e=LabelEncoder()
    city_e=le_c.fit_transform(city); emp_e=le_e.fit_transform(emp)
    F=['age','city_e','emp_e','income','other','bal','jt','cibil','defs',
       'inq','emi','dti','lreq','tm','ins','co']
    df=pd.DataFrame(dict(zip(F,[age,city_e,emp_e,income,other,bal,jt,cibil,defs,inq,emi,dti,lreq,tm,ins,co])))
    df['status']=status; df['amt']=amt; df['rate']=rate
    # ★ Best algorithm: Gradient Boosting Classifier
    clf=GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=.08,random_state=42).fit(df[F],df['status'])
    app=df[df['status']==1]
    amtm=GradientBoostingRegressor(n_estimators=150,random_state=42).fit(app[F],app['amt'])
    ratem=Ridge(alpha=1.0).fit(app[F],app['rate'])
    acc=accuracy_score(df['status'],clf.predict(df[F]))
    return clf,amtm,ratem,le_c,le_e,F,df,acc


# ── HOME LOAN  →  Random Forest (best for property-backed structured data) ──
@st.cache_resource
def build_home():
    rng=np.random.RandomState(42); n=4000
    age=rng.randint(22,60,n); city=rng.choice(['Metro','Urban','Semi-Urban'],n)
    emp=rng.choice(['Salaried','Self-Employed','Business Owner'],n)
    income=rng.randint(20000,300000,n); other=rng.randint(0,50000,n)
    bal=rng.randint(10000,1000000,n); jt=rng.randint(6,360,n)
    cibil=rng.randint(300,900,n); defs=rng.choice([0,0,0,0,1],n)
    inq=rng.randint(0,5,n); emi=rng.randint(0,80000,n)
    prop=rng.randint(1500000,20000000,n); dp=(prop*rng.uniform(.10,.30,n)).astype(int)
    lreq=prop-dp; tm=rng.choice([120,180,240,300,360],n)
    tot=income+other; dti=(emi+lreq/tm)/(tot+1)
    emr=emi/(tot+1); lir=lreq/(tot*12+1)
    sc=((cibil>=700)*4+(defs==0)*3+(dti<=.55)*3+(income>=30000)*2+(dp/prop>.15)*2+(inq<=2)+rng.randn(n)*.3)
    status=(sc>=11).astype(int)
    amt=np.where(status==1,np.minimum(lreq*rng.uniform(.7,1,n),tot*60),0)
    rate=np.where(status==1,np.clip(11-(cibil-300)/130+rng.randn(n)*.3,6.5,14),0)
    le_c=LabelEncoder(); le_e=LabelEncoder()
    city_e=le_c.fit_transform(city); emp_e=le_e.fit_transform(emp)
    F=['age','city_e','emp_e','income','other','bal','jt','cibil','defs',
       'inq','emi','prop','dp','lreq','tm','tot','emr','lir','dti']
    df=pd.DataFrame(dict(zip(F,[age,city_e,emp_e,income,other,bal,jt,cibil,defs,
                                 inq,emi,prop,dp,lreq,tm,tot,emr,lir,dti])))
    df['status']=status; df['amt']=amt; df['rate']=rate
    # ★ Best algorithm: Random Forest Classifier
    clf=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=8,
                               class_weight='balanced',random_state=42).fit(df[F],df['status'])
    app=df[df['status']==1]
    amtm=RandomForestRegressor(n_estimators=200,random_state=42).fit(app[F],app['amt'])
    ratem=RandomForestRegressor(n_estimators=100,random_state=42).fit(app[F],app['rate'])
    acc=accuracy_score(df['status'],clf.predict(df[F]))
    return clf,amtm,ratem,le_c,le_e,F,df,acc


# ── VEHICLE LOAN  →  Extra Trees (fast + handles mixed features well) ──
@st.cache_resource
def build_vehicle():
    rng=np.random.RandomState(42); n=4000
    age=rng.randint(21,58,n); city=rng.choice(['Metro','Urban','Semi-Urban'],n)
    emp=rng.choice(['Salaried','Self-Employed','Business Owner'],n)
    vtype=rng.choice(['Two Wheeler','Four Wheeler','Commercial'],n)
    inc=rng.randint(15000,200000,n); bal=rng.randint(5000,500000,n)
    jt=rng.randint(3,240,n); cibil=rng.randint(300,900,n)
    defs=rng.choice([0,0,0,0,1],n); inq=rng.randint(0,6,n); emi=rng.randint(0,40000,n)
    vp=rng.randint(50000,3000000,n); dp=(vp*rng.uniform(.10,.30,n)).astype(int)
    lreq=vp-dp; tm=rng.choice([12,24,36,48,60,72],n)
    lu=lreq/(vp+1); eb=emi/(inc+1); ni=inc-emi; dpr=dp/(vp+1); b2i=bal/(inc+1)
    dti=(emi+lreq/tm)/(inc+1); hr=((defs>0)&(cibil<650)).astype(int)
    sc=((cibil>=650)*4+(defs==0)*3+(dti<=.5)*2+(inc>=20000)*2+(dpr>.15)*2+(hr==0)*2+rng.randn(n)*.3)
    status=(sc>=11).astype(int)
    amt=np.where(status==1,np.minimum(lreq*rng.uniform(.65,1,n),inc*20),0)
    rate=np.where(status==1,np.clip(13-(cibil-300)/90+rng.randn(n)*.4,7,18),0)
    le_c=LabelEncoder(); le_e=LabelEncoder(); le_v=LabelEncoder()
    ce=le_c.fit_transform(city); ee=le_e.fit_transform(emp); ve=le_v.fit_transform(vtype)
    F=['age','ce','ee','ve','inc','bal','jt','cibil','defs','inq','emi',
       'vp','dp','lreq','tm','lu','eb','ni','dpr','b2i','dti','hr']
    df=pd.DataFrame(dict(zip(F,[age,ce,ee,ve,inc,bal,jt,cibil,defs,inq,emi,vp,dp,lreq,tm,lu,eb,ni,dpr,b2i,dti,hr])))
    df['status']=status; df['amt']=amt; df['rate']=rate; df['vtype']=vtype
    # ★ Best algorithm: Extra Trees Classifier
    clf=ExtraTreesClassifier(n_estimators=250,max_depth=12,min_samples_leaf=4,
                             class_weight='balanced',random_state=42).fit(df[F],df['status'])
    app=df[df['status']==1]
    amtm=ExtraTreesRegressor(n_estimators=150,random_state=42).fit(app[F],app['amt'])
    ratem=GradientBoostingRegressor(n_estimators=100,random_state=42).fit(app[F],app['rate'])
    acc=accuracy_score(df['status'],clf.predict(df[F]))
    return clf,amtm,ratem,le_c,le_e,le_v,F,df,acc


# ── GOLD LOAN  →  Random Forest + Logistic (ensemble logic for LTV-based) ──
@st.cache_resource
def build_gold():
    rng=np.random.RandomState(42); n=4000
    age=rng.randint(21,65,n); gen=rng.choice(['Male','Female'],n)
    city=rng.choice(['Metro','Urban','Rural'],n)
    emp=rng.choice(['Salaried','Self-Employed','Farmer','Business Owner'],n)
    inc=rng.randint(8000,150000,n); emi=rng.randint(0,30000,n)
    cibil=rng.randint(300,900,n); gw=rng.uniform(5,200,n)
    gpur=rng.choice([18,20,22,24],n); gval=(gw*gpur*300).astype(int)
    ltv=rng.uniform(.50,.85,n); lreq=(gval*ltv).astype(int)
    tm=rng.choice([3,6,9,12,18,24],n)
    gu=lreq/(gval+1); eb=emi/(inc+1); lpg=lreq/(gw+1); ni=inc-emi; dti=(emi+lreq/tm)/(inc+1)
    sc=((cibil>=550)*3+(ltv<=.75)*4+(dti<=.60)*3+(gpur>=20)*2+(inc>=10000)*2+rng.randn(n)*.3)
    status=(sc>=10).astype(int)
    amt=np.where(status==1,np.minimum(lreq*rng.uniform(.7,1,n),gval*.85),0)
    rate=np.where(status==1,np.clip(14-(cibil-300)/90+rng.randn(n)*.3,9,18),0)
    le_g=LabelEncoder(); le_c=LabelEncoder(); le_e=LabelEncoder()
    ge=le_g.fit_transform(gen); ce=le_c.fit_transform(city); ee=le_e.fit_transform(emp)
    F=['age','ge','ce','ee','inc','emi','cibil','gw','gpur','gval','ltv','lreq','tm','gu','eb','lpg','ni','dti']
    df=pd.DataFrame(dict(zip(F,[age,ge,ce,ee,inc,emi,cibil,gw,gpur,gval,ltv,lreq,tm,gu,eb,lpg,ni,dti])))
    df['status']=status; df['amt']=amt; df['rate']=rate; df['gpur_raw']=gpur
    # ★ Best algorithm: Random Forest (best for LTV + gold features)
    clf=RandomForestClassifier(n_estimators=200,max_depth=9,class_weight='balanced',random_state=42).fit(df[F],df['status'])
    app=df[df['status']==1]
    amtm=RandomForestRegressor(n_estimators=150,random_state=42).fit(app[F],app['amt'])
    ratem=GradientBoostingRegressor(n_estimators=100,random_state=42).fit(app[F],app['rate'])
    acc=accuracy_score(df['status'],clf.predict(df[F]))
    return clf,amtm,ratem,le_g,le_c,le_e,F,df,acc


# ── EDUCATION LOAN  →  Gradient Boosting (best for academic + financial mix) ──
@st.cache_resource
def build_education():
    rng=np.random.RandomState(42); n=4000
    age=rng.randint(17,35,n); asc=rng.uniform(40,100,n); esc=rng.randint(0,800,n)
    cs=rng.choice(['Engineering','Medical','Arts','Commerce','Science'],n)
    cl=rng.choice(['UG','PG','PhD','Diploma'],n)
    it=rng.choice(['Tier1','Tier2','Tier3','ForeignTop','ForeignOther'],n)
    cn=rng.choice(['India','USA','UK','Australia','Canada','Germany'],n)
    pinc=rng.randint(20000,500000,n); pcib=rng.randint(300,900,n)
    pdti=rng.uniform(.1,.8,n); pemi=(pinc*pdti*.4).astype(int)
    lreq=rng.randint(100000,8000000,n); tuit=rng.randint(50000,5000000,n)
    liv=rng.randint(30000,1500000,n); dp=rng.randint(0,200000,n)
    ty=rng.choice([3,5,7,10,12,15],n)
    tc=tuit+liv; l2i=lreq/(pinc+1); fc=lreq/(tc+1)
    ee_emi=lreq/(ty*12+1); tdb=(pemi+ee_emi)/(pinc+1)
    isfor=(cn!='India').astype(int)
    TMAP={'Tier3':0,'Tier2':1,'Tier1':2,'ForeignOther':3,'ForeignTop':4}
    te=np.array([TMAP[x] for x in it]); istop=(te>=2).astype(int)
    sc_c=(asc/100)+(esc/800)
    CMAP={'Arts':0,'Commerce':1,'Engineering':2,'Medical':3,'Science':4}
    LMAP={'Diploma':0,'UG':1,'PG':2,'PhD':3}
    cse=np.array([CMAP[x] for x in cs]); cle=np.array([LMAP[x] for x in cl])
    sc=((pcib>=700)*4+(asc>=60)*3+(tdb<=.55)*3+(istop==1)*2+(esc>=300)*2+(pinc>=30000)*2+rng.randn(n)*.3)
    status=(sc>=12).astype(int)
    amt=np.where(status==1,np.minimum(lreq*rng.uniform(.6,1,n),pinc*60),0)
    rate=np.where(status==1,np.clip(10-(pcib-300)/130+rng.randn(n)*.3,5,13),0)
    F=['age','asc','esc','cse','cle','te','isfor','pinc','pcib','pdti',
       'pemi','lreq','tuit','liv','dp','ty','tc','l2i','fc','ee_emi','tdb','istop','sc_c']
    df=pd.DataFrame(dict(zip(F,[age,asc,esc,cse,cle,te,isfor,pinc,pcib,pdti,
                                 pemi,lreq,tuit,liv,dp,ty,tc,l2i,fc,ee_emi,tdb,istop,sc_c])))
    df['status']=status; df['amt']=amt; df['rate']=rate
    df['course']=cs; df['tier']=it
    # ★ Best algorithm: Gradient Boosting (best for academic + financial mix)
    clf=GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=.08,random_state=42).fit(df[F],df['status'])
    app=df[df['status']==1]
    amtm=GradientBoostingRegressor(n_estimators=150,random_state=42).fit(app[F],app['amt'])
    ratem=Ridge(alpha=1.0).fit(app[F],app['rate'])
    acc=accuracy_score(df['status'],clf.predict(df[F]))
    return clf,amtm,ratem,F,df,CMAP,LMAP,TMAP,acc


# ═══════════════════════════════════════════════════════════════
#  BUSINESS RULE VALIDATORS
# ═══════════════════════════════════════════════════════════════

def rules_personal(age,income,cibil,defs,dti,inq,bal,lreq):
    R=[("🎂 Applicant age 21–65 ke beech hona chahiye",        21<=age<=65),
       ("💳 CIBIL Score minimum 650 hona chahiye",              cibil>=650),
       ("🚫 Past Defaults bilkul zero hone chahiye",            defs==0),
       ("💰 Monthly Income minimum ₹20,000 hona chahiye",      income>=20000),
       ("📊 DTI Ratio 0.50 se zyada nahi hona chahiye",         dti<=0.50),
       ("🔍 Last 6 months mein max 3 loan inquiries chalenge", inq<=3),
       ("🏦 Bank average balance ₹10,000 se zyada hona chahiye",bal>=10000),
       ("💵 Loan amount income ka maximum 30 guna ho sakta hai",lreq<=income*30)]
    return [("✅ " if ok else "❌ ")+lb for lb,ok in R], all(ok for _,ok in R)

def rules_home(age,income,cibil,defs,dti,dp,prop,inq):
    dp_p=dp/(prop+1)
    R=[("🎂 Applicant age 22–60 ke beech hona chahiye",            22<=age<=60),
       ("💳 CIBIL Score minimum 700 hona chahiye",                  cibil>=700),
       ("🚫 Past Defaults zero hone chahiye",                       defs==0),
       ("💰 Monthly Income minimum ₹30,000 hona chahiye",          income>=30000),
       ("📊 DTI Ratio 0.55 se zyada nahi hona chahiye",             dti<=0.55),
       ("🏠 Down Payment property value ka minimum 15% chahiye",    dp_p>=0.15),
       ("🔍 Last 6 months mein maximum 2 loan inquiries chalenge",  inq<=2)]
    return [("✅ " if ok else "❌ ")+lb for lb,ok in R], all(ok for _,ok in R)

def rules_vehicle(age,income,cibil,defs,dti,dp,vp,inq):
    dp_p=dp/(vp+1)
    R=[("🎂 Applicant age 21–58 ke beech hona chahiye",           21<=age<=58),
       ("💳 CIBIL Score minimum 650 hona chahiye",                 cibil>=650),
       ("🚫 Past Defaults zero hone chahiye",                      defs==0),
       ("💰 Monthly Income minimum ₹20,000 hona chahiye",         income>=20000),
       ("📊 DTI Ratio 0.50 se zyada nahi hona chahiye",            dti<=0.50),
       ("🚗 Down Payment vehicle price ka minimum 15% chahiye",    dp_p>=0.15),
       ("🔍 Last 6 months mein maximum 3 loan inquiries chalenge", inq<=3)]
    return [("✅ " if ok else "❌ ")+lb for lb,ok in R], all(ok for _,ok in R)

def rules_gold(age,income,cibil,ltv,dti,gpur):
    R=[("🎂 Applicant age 21–65 ke beech hona chahiye",         21<=age<=65),
       ("💳 CIBIL Score minimum 550 hona chahiye",               cibil>=550),
       ("💰 Monthly Income minimum ₹10,000 hona chahiye",       income>=10000),
       ("📊 DTI Ratio 0.60 se zyada nahi hona chahiye",          dti<=0.60),
       ("⚖️ LTV Ratio 0.75 se zyada nahi hona chahiye",          ltv<=0.75),
       ("💛 Gold purity minimum 20 Karat hona chahiye",           gpur>=20)]
    return [("✅ " if ok else "❌ ")+lb for lb,ok in R], all(ok for _,ok in R)

def rules_education(age,pinc,pcib,asc,esc,tdb,lreq):
    R=[("🎂 Student age 17–35 ke beech hona chahiye",              17<=age<=35),
       ("💳 Parent CIBIL Score minimum 700 hona chahiye",           pcib>=700),
       ("💰 Parent Monthly Income minimum ₹30,000 hona chahiye",  pinc>=30000),
       ("📚 Academic Score minimum 60% hona chahiye",              asc>=60),
       ("📝 Entrance Exam Score minimum 300 hona chahiye",         esc>=300),
       ("📊 Total Debt Burden 0.55 se zyada nahi hona chahiye",    tdb<=0.55),
       ("💵 Loan amount parent income ka max 60 guna ho sakta hai",lreq<=pinc*60)]
    return [("✅ " if ok else "❌ ")+lb for lb,ok in R], all(ok for _,ok in R)


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def emi_calc(p,r,m):
    rv=r/(12*100)
    if rv<=0: return p/max(m,1)
    return p*rv*(1+rv)**m/((1+rv)**m-1)

def risk_tag(cibil,dti,defs,prob):
    if cibil>=750 and dti<=.35 and defs==0 and prob>=.75: return "Low Risk","#1A8745"
    if cibil>=650 and dti<=.50 and defs<=1:              return "Medium Risk","#D68910"
    return "High Risk","#C0392B"

def cross_sell_offers(current,hist):
    pool={"🏠 Home Loan":"Ghar kharidne ke liye best rates",
          "💼 Personal Loan":"Kisi bhi zarurat ke liye — no collateral",
          "🚗 Vehicle Loan":"New car/bike ke liye special offer",
          "💛 Gold Loan":"Gold pe instant loan",
          "🎓 Education Loan":"Bacchon ki education mein invest karo"}
    filtered={k:v for k,v in pool.items() if current.lower() not in k.lower()}
    if hist=="Excellent": return list(filtered.items())[:3]
    if hist=="Good":      return list(filtered.items())[:2]
    return []

def show_conditions(checks):
    st.markdown('<div class="sec-t">📋 Business Rule Validation</div>',unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="cond-wrap">',unsafe_allow_html=True)
        c1,c2=st.columns(2); mid=len(checks)//2+len(checks)%2
        for i,line in enumerate(checks):
            (c1 if i<mid else c2).markdown(f"**{line}**")
        st.markdown('</div>',unsafe_allow_html=True)

def show_result(is_app,prob,app_amt,int_rate,t_months,
                cibil,dti,defs,loan_type,phist,req_amt):
    if not is_app:
        st.markdown("""<div class="card-rejected">
        <div class="res-big" style="color:#E74C3C;">❌ LOAN APPLICATION REJECTED</div>
        <div class="res-sub" style="color:rgba(255,255,255,.78);">
          Kuch mandatory conditions fail hui hain (red ❌ items dekho).<br>
          Un conditions ko improve karo aur dobara apply karo.
        </div></div>""",unsafe_allow_html=True)
        return

    emi=emi_calc(app_amt,int_rate,t_months)
    ty=t_months//12; rl,rc=risk_tag(cibil,dti,defs,prob)

    st.markdown(f"""<div class="card-approved">
    <div class="res-big" style="color:#1A8745;">✅ LOAN APPROVED!</div>
    <div class="res-sub" style="color:rgba(255,255,255,.78);">
      Sabhi conditions pass hui! Aapka <strong>{loan_type}</strong> approve ho gaya.
    </div>
    <div class="mg">
      <div class="mb"><span class="mv" style="color:#1A8745;">₹{app_amt:,.0f}</span>
        <div class="ml">Approved Amount</div></div>
      <div class="mb"><span class="mv" style="color:#00C896;">{int_rate:.2f}%</span>
        <div class="ml">Interest Rate p.a.</div></div>
      <div class="mb"><span class="mv" style="color:#FFB800;">₹{emi:,.0f}</span>
        <div class="ml">Monthly EMI</div></div>
      <div class="mb"><span class="mv" style="color:#7BA7E0;">{ty}Y {t_months%12}M</span>
        <div class="ml">Loan Tenure</div></div>
    </div></div>""",unsafe_allow_html=True)

    st.markdown("")
    m1,m2,m3=st.columns(3)
    with m1: st.metric("🎯 CIBIL Score",f"{cibil}","Excellent" if cibil>750 else "Good" if cibil>650 else "Fair")
    with m2: st.metric("⚠️ Risk Level",rl)
    with m3: st.metric("📊 Approval Probability",f"{prob*100:.1f}%")

    if app_amt<req_amt*.95:
        pct=(req_amt-app_amt)/req_amt*100
        st.markdown(f"""<div class="yb">⚠️ <strong>Partial Approval:</strong>
        Aapne ₹{req_amt:,.0f} maange the — ₹{app_amt:,.0f} approve hue ({pct:.1f}% kam).
        CIBIL improve karo aur income badhao — limit aur badhegi.</div>""",unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="ib">✅ <strong>Full Amount Approved!</strong>
        Requested amount poori approve hui hai. Badhai ho!</div>""",unsafe_allow_html=True)

    st.markdown('<hr class="dv">',unsafe_allow_html=True)
    st.markdown('<div class="sec-t">📈 Payment History — Future Loan Impact</div>',unsafe_allow_html=True)
    ph1,ph2=st.columns(2)
    with ph1:
        if phist=="Excellent":
            st.markdown(f"""<div class="ib">🌟 <strong>Excellent Track Record!</strong><br>
            Sabhi loans time pe bhare hain. Agali baar limit
            <strong>30% badh sakti hai → ₹{app_amt*1.3:,.0f}</strong></div>""",unsafe_allow_html=True)
        elif phist=="Good":
            st.markdown(f"""<div class="ib">✅ <strong>Good Track Record!</strong><br>
            Consistent payments se limit
            <strong>15% badh sakti hai → ₹{app_amt*1.15:,.0f}</strong></div>""",unsafe_allow_html=True)
        elif phist=="Average":
            st.markdown(f"""<div class="yb">⚠️ <strong>Average Track Record.</strong><br>
            Kuch late payments the. Limit
            <strong>10% ghath sakti hai → ₹{app_amt*.90:,.0f}</strong>. Samay pe EMI bharo!</div>""",unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="wb">🔴 <strong>Poor Track Record!</strong><br>
            Missed payments detect hue. Limit
            <strong>25% ghath sakti hai → ₹{app_amt*.75:,.0f}</strong>. Turant improve karo!</div>""",unsafe_allow_html=True)
    with ph2:
        offers=cross_sell_offers(loan_type,phist)
        if offers:
            badges="".join([f'<span class="lb">{nm}</span>' for nm,_ in offers])
            desc=offers[0][1]
            st.markdown(f"""<div class="ob">
            <div class="ob-t">🎁 Aapke Liye Exclusive Bank Offers</div>
            <p style="color:rgba(255,255,255,.6);font-size:.82rem;margin-bottom:10px;">
            Excellent track record ke basis pe ye loans bhi available hain:</p>
            {badges}
            <p style="color:#FFB800;font-size:.78rem;margin-top:10px;">💡 {desc}</p>
            </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""<div class="yb">⚠️ Payment history improve karo —
            exclusive cross-sell offers unlock honge.</div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HOME PAGE  — Unique charts per loan type + algo comparison
# ═══════════════════════════════════════════════════════════════

def home_page():
    st.markdown("""<div class="bank-header">
    <div class="bh-badge">🏦 Analytics Dashboard</div>
    <h1>📊 Bank Finance System — Intelligence Overview</h1>
    <p>Har loan type ka unique data analytics, algorithm performance, aur approval trends</p>
    </div>""",unsafe_allow_html=True)

    with st.spinner("🔄 All models loading aur charts generate ho rahe hain..."):
        _,_,_,_,_,_,df_p,acc_p = build_personal()
        _,_,_,_,_,_,df_h,acc_h = build_home()
        _,_,_,_,_,_,_,df_v,acc_v = build_vehicle()
        _,_,_,_,_,_,_,df_g,acc_g = build_gold()
        _,_,_,_,df_e,_,_,_,acc_e = build_education()

    # ── KPI Strip ──
    tot_apps = len(df_p)+len(df_h)+len(df_v)+len(df_g)+len(df_e)
    tot_app  = (df_p['status'].sum()+df_h['status'].sum()+df_v['status'].sum()+
                df_g['status'].sum()+df_e['status'].sum())
    avg_acc  = np.mean([acc_p,acc_h,acc_v,acc_g,acc_e])*100
    st.markdown(f"""<div class="kpi-row">
    <div class="kpi-card" style="--kpi-color:#00C896;">
      <div class="kpi-val">{tot_apps:,}</div><div class="kpi-lbl">Total Records</div></div>
    <div class="kpi-card" style="--kpi-color:#1A8745;">
      <div class="kpi-val">{tot_app/tot_apps*100:.1f}%</div><div class="kpi-lbl">Overall Approval Rate</div></div>
    <div class="kpi-card" style="--kpi-color:#2980B9;">
      <div class="kpi-val">5</div><div class="kpi-lbl">Loan Categories</div></div>
    <div class="kpi-card" style="--kpi-color:#8E44AD;">
      <div class="kpi-val">{avg_acc:.1f}%</div><div class="kpi-lbl">Avg Model Accuracy</div></div>
    <div class="kpi-card" style="--kpi-color:#E67E22;">
      <div class="kpi-val">3</div><div class="kpi-lbl">ML Algorithms Used</div></div>
    </div>""",unsafe_allow_html=True)

    BG="#0C1A2E"
    plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':'#2A3F5F',
        'text.color':'#DDE8F5','xtick.color':'#8899AA','ytick.color':'#8899AA',
        'grid.color':'#1E2F45','grid.alpha':.45,'font.family':'sans-serif'})

    # ════════════════════════════════════
    # 1. PERSONAL LOAN — CIBIL + DTI + Approval + Feature Importance
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">💼 Personal Loan — Gradient Boosting Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Records",f"{len(df_p):,}"); m2.metric("Approval Rate",f"{df_p['status'].mean()*100:.1f}%")
    m3.metric("Best Algorithm","Gradient Boosting",delta=f"{acc_p*100:.1f}% Accuracy")
    m4.metric("Avg Approved Amt",f"₹{df_p[df_p['status']==1]['amt'].mean()/100000:.1f}L")
    fig,axes=plt.subplots(1,4,figsize=(18,3.8)); fig.patch.set_facecolor(BG)
    # Chart 1: CIBIL dist
    ax=axes[0]; bins=np.linspace(300,900,30)
    ax.hist(df_p[df_p['status']==1]['cibil'],bins=bins,alpha=.75,color='#4A90D9',label='Approved',edgecolor='none')
    ax.hist(df_p[df_p['status']==0]['cibil'],bins=bins,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(650,color='#FFB800',ls='--',lw=1.5,label='Min=650')
    ax.set_title("CIBIL Score Distribution",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 2: DTI vs Approval
    ax=axes[1]
    dti_app=df_p[df_p['status']==1]['dti'].clip(0,1); dti_rej=df_p[df_p['status']==0]['dti'].clip(0,1)
    ax.hist(dti_app,bins=25,alpha=.75,color='#00C896',label='Approved',edgecolor='none')
    ax.hist(dti_rej,bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(0.5,color='#FFB800',ls='--',lw=1.5,label='Max DTI=0.5')
    ax.set_title("DTI Ratio Analysis",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 3: Income distribution
    ax=axes[2]
    ax.hist(df_p[df_p['status']==1]['income']/1000,bins=25,alpha=.78,color='#4A90D9',edgecolor='none',label='Approved')
    ax.hist(df_p[df_p['status']==0]['income']/1000,bins=25,alpha=.55,color='#E74C3C',edgecolor='none',label='Rejected')
    ax.set_title("Monthly Income (₹K)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 4: Approved amount dist
    ax=axes[3]
    amts=df_p[df_p['status']==1]['amt']/100000
    ax.hist(amts,bins=28,color='#4A90D9',edgecolor='none',alpha=.82)
    ax.axvline(amts.mean(),color='#FFB800',ls='--',lw=1.5,label=f"Mean ₹{amts.mean():.1f}L")
    ax.set_title("Approved Loan Amount (₹L)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    plt.tight_layout(pad=1.4); st.pyplot(fig); plt.close(fig)
    st.markdown('<hr class="dv">',unsafe_allow_html=True)

    # ════════════════════════════════════
    # 2. HOME LOAN — Property + LTV + EMI analysis
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">🏠 Home Loan — Random Forest Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Records",f"{len(df_h):,}"); m2.metric("Approval Rate",f"{df_h['status'].mean()*100:.1f}%")
    m3.metric("Best Algorithm","Random Forest",delta=f"{acc_h*100:.1f}% Accuracy")
    m4.metric("Avg Property Val",f"₹{df_h['prop'].mean()/100000:.0f}L")
    fig,axes=plt.subplots(1,4,figsize=(18,3.8)); fig.patch.set_facecolor(BG)
    # Chart 1: Property value distribution
    ax=axes[0]
    ax.hist(df_h[df_h['status']==1]['prop']/100000,bins=25,alpha=.75,color='#27AE60',label='Approved',edgecolor='none')
    ax.hist(df_h[df_h['status']==0]['prop']/100000,bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.set_title("Property Value (₹L)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 2: Down payment ratio
    ax=axes[1]
    dp_ratio_app=df_h[df_h['status']==1]['dp']/df_h[df_h['status']==1]['prop']*100
    dp_ratio_rej=df_h[df_h['status']==0]['dp']/df_h[df_h['status']==0]['prop']*100
    ax.hist(dp_ratio_app,bins=25,alpha=.75,color='#27AE60',label='Approved',edgecolor='none')
    ax.hist(dp_ratio_rej,bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(15,color='#FFB800',ls='--',lw=1.5,label='Min 15%')
    ax.set_title("Down Payment % of Property",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 3: CIBIL dist (Home-specific)
    ax=axes[2]
    ax.hist(df_h[df_h['status']==1]['cibil'],bins=25,alpha=.75,color='#27AE60',label='Approved',edgecolor='none')
    ax.hist(df_h[df_h['status']==0]['cibil'],bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(700,color='#FFB800',ls='--',lw=1.5,label='Min=700')
    ax.set_title("CIBIL Score (Home Loan)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 4: Loan tenure distribution
    ax=axes[3]
    tenures_app=df_h[df_h['status']==1]['tm']; tenures_rej=df_h[df_h['status']==0]['tm']
    t_vals=[120,180,240,300,360]; t_lbls=['10Y','15Y','20Y','25Y','30Y']
    app_c=[len(tenures_app[tenures_app==t]) for t in t_vals]
    rej_c=[len(tenures_rej[tenures_rej==t]) for t in t_vals]
    x=np.arange(5)
    ax.bar(x-.18,app_c,.35,color='#27AE60',label='Approved',edgecolor='none',alpha=.85)
    ax.bar(x+.18,rej_c,.35,color='#E74C3C',label='Rejected',edgecolor='none',alpha=.75)
    ax.set_xticks(x); ax.set_xticklabels(t_lbls,fontsize=8)
    ax.set_title("Tenure Preference",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(axis='y',alpha=.3)
    plt.tight_layout(pad=1.4); st.pyplot(fig); plt.close(fig)
    st.markdown('<hr class="dv">',unsafe_allow_html=True)

    # ════════════════════════════════════
    # 3. VEHICLE LOAN — Vehicle type + DTI + Price analysis
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">🚗 Vehicle Loan — Extra Trees Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Records",f"{len(df_v):,}"); m2.metric("Approval Rate",f"{df_v['status'].mean()*100:.1f}%")
    m3.metric("Best Algorithm","Extra Trees",delta=f"{acc_v*100:.1f}% Accuracy")
    m4.metric("Avg Vehicle Price",f"₹{df_v['vp'].mean()/100000:.1f}L")
    fig,axes=plt.subplots(1,4,figsize=(18,3.8)); fig.patch.set_facecolor(BG)
    # Chart 1: Vehicle type approval rate
    ax=axes[0]
    vtypes=['Two Wheeler','Four Wheeler','Commercial']
    v_app=[df_v[(df_v['vtype']==vt)&(df_v['status']==1)].shape[0] for vt in vtypes]
    v_rej=[df_v[(df_v['vtype']==vt)&(df_v['status']==0)].shape[0] for vt in vtypes]
    v_rate=[a/(a+r+1)*100 for a,r in zip(v_app,v_rej)]
    colors_v=['#E74C3C','#3498DB','#E67E22']
    bars=ax.bar(['2W','4W','Comm.'],v_rate,color=colors_v,edgecolor='none',alpha=.88)
    for b,v in zip(bars,v_rate):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,f'{v:.1f}%',
                ha='center',va='bottom',color='#DDE8F5',fontsize=9,fontweight='bold')
    ax.set_title("Approval Rate by Vehicle Type",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.set_ylabel("Approval Rate %",fontsize=8); ax.grid(axis='y',alpha=.3)
    # Chart 2: Vehicle price distribution
    ax=axes[1]
    ax.hist(df_v[df_v['status']==1]['vp']/100000,bins=25,alpha=.75,color='#E74C3C',label='Approved',edgecolor='none')
    ax.hist(df_v[df_v['status']==0]['vp']/100000,bins=25,alpha=.55,color='#8E44AD',label='Rejected',edgecolor='none')
    ax.set_title("Vehicle Price (₹L)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 3: DTI analysis
    ax=axes[2]
    ax.hist(df_v[df_v['status']==1]['dti'].clip(0,1),bins=25,alpha=.75,color='#E74C3C',label='Approved',edgecolor='none')
    ax.hist(df_v[df_v['status']==0]['dti'].clip(0,1),bins=25,alpha=.55,color='#8E44AD',label='Rejected',edgecolor='none')
    ax.axvline(.5,color='#FFB800',ls='--',lw=1.5,label='Max=0.5')
    ax.set_title("DTI Ratio (Vehicle)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 4: Approved amount
    ax=axes[3]
    amts=df_v[df_v['status']==1]['amt']/100000
    ax.hist(amts,bins=28,color='#E74C3C',edgecolor='none',alpha=.82)
    ax.axvline(amts.mean(),color='#FFB800',ls='--',lw=1.5,label=f"Mean ₹{amts.mean():.1f}L")
    ax.set_title("Approved Amount (₹L)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    plt.tight_layout(pad=1.4); st.pyplot(fig); plt.close(fig)
    st.markdown('<hr class="dv">',unsafe_allow_html=True)

    # ════════════════════════════════════
    # 4. GOLD LOAN — Gold value + LTV + Purity analysis
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">💛 Gold Loan — Random Forest Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Records",f"{len(df_g):,}"); m2.metric("Approval Rate",f"{df_g['status'].mean()*100:.1f}%")
    m3.metric("Best Algorithm","Random Forest",delta=f"{acc_g*100:.1f}% Accuracy")
    m4.metric("Avg Gold Value",f"₹{df_g['gval'].mean()/1000:.1f}K")
    fig,axes=plt.subplots(1,4,figsize=(18,3.8)); fig.patch.set_facecolor(BG)
    # Chart 1: Gold purity approval rate
    ax=axes[0]
    purities=[18,20,22,24]
    p_app=[df_g[(df_g['gpur_raw']==p)&(df_g['status']==1)].shape[0] for p in purities]
    p_rej=[df_g[(df_g['gpur_raw']==p)&(df_g['status']==0)].shape[0] for p in purities]
    p_rate=[a/(a+r+1)*100 for a,r in zip(p_app,p_rej)]
    clrs_p=['#E74C3C','#F39C12','#FFB800','#F1C40F']
    bars=ax.bar([f'{p}K' for p in purities],p_rate,color=clrs_p,edgecolor='none',alpha=.88)
    for b,v in zip(bars,p_rate):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,f'{v:.1f}%',
                ha='center',va='bottom',color='#DDE8F5',fontsize=9,fontweight='bold')
    ax.axhline(0,color='#FFB800',ls='--',lw=.5)
    ax.set_title("Approval Rate by Purity",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.set_ylabel("Approval %",fontsize=8); ax.grid(axis='y',alpha=.3)
    # Chart 2: Gold value vs loan amount scatter
    ax=axes[1]
    sample=df_g[df_g['status']==1].sample(min(600,df_g['status'].sum()),random_state=42)
    ax.scatter(sample['gval']/1000,sample['lreq']/1000,alpha=.4,s=10,
               c=sample['ltv'],cmap='YlOrRd',edgecolors='none')
    ax.set_title("Gold Value vs Loan Amount",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.set_xlabel("Gold Value (₹K)",fontsize=8); ax.set_ylabel("Loan Amount (₹K)",fontsize=8); ax.grid(True,alpha=.3)
    # Chart 3: LTV ratio distribution
    ax=axes[2]
    ax.hist(df_g[df_g['status']==1]['ltv'],bins=25,alpha=.75,color='#FFB800',label='Approved',edgecolor='none')
    ax.hist(df_g[df_g['status']==0]['ltv'],bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(.75,color='#00C896',ls='--',lw=1.5,label='Max LTV=0.75')
    ax.set_title("LTV Ratio Distribution",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 4: Tenure distribution
    ax=axes[3]
    tvals=[3,6,9,12,18,24]; t_app=[df_g[df_g['status']==1]['tm'].value_counts().get(t,0) for t in tvals]
    t_rej=[df_g[df_g['status']==0]['tm'].value_counts().get(t,0) for t in tvals]
    x=np.arange(len(tvals))
    ax.bar(x-.18,t_app,.35,color='#FFB800',label='Approved',edgecolor='none',alpha=.88)
    ax.bar(x+.18,t_rej,.35,color='#E74C3C',label='Rejected',edgecolor='none',alpha=.75)
    ax.set_xticks(x); ax.set_xticklabels([f'{t}M' for t in tvals],fontsize=8)
    ax.set_title("Loan Tenure Preference",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(axis='y',alpha=.3)
    plt.tight_layout(pad=1.4); st.pyplot(fig); plt.close(fig)
    st.markdown('<hr class="dv">',unsafe_allow_html=True)

    # ════════════════════════════════════
    # 5. EDUCATION LOAN — Academic + Institute tier + Debt burden
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">🎓 Education Loan — Gradient Boosting Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Records",f"{len(df_e):,}"); m2.metric("Approval Rate",f"{df_e['status'].mean()*100:.1f}%")
    m3.metric("Best Algorithm","Gradient Boosting",delta=f"{acc_e*100:.1f}% Accuracy")
    m4.metric("Foreign Study",f"{df_e['isfor'].mean()*100:.1f}%")
    fig,axes=plt.subplots(1,4,figsize=(18,3.8)); fig.patch.set_facecolor(BG)
    # Chart 1: Institute tier approval
    ax=axes[0]
    tiers=['Tier3','Tier2','Tier1','ForeignOther','ForeignTop']
    clr_t=['#E74C3C','#E67E22','#27AE60','#3498DB','#9B59B6']
    t_rates=[]
    for ti in tiers:
        mask=df_e['tier']==ti
        t_rates.append(df_e[mask]['status'].mean()*100 if mask.sum()>0 else 0)
    bars=ax.bar([t.replace('Foreign','F.') for t in tiers],t_rates,
                color=clr_t,edgecolor='none',alpha=.88)
    for b,v in zip(bars,t_rates):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,f'{v:.0f}%',
                ha='center',va='bottom',color='#DDE8F5',fontsize=8,fontweight='bold')
    ax.set_title("Approval Rate by Institute Tier",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.set_ylabel("Approval %",fontsize=8); ax.grid(axis='y',alpha=.3)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20,fontsize=7.5)
    # Chart 2: Academic score distribution
    ax=axes[1]
    ax.hist(df_e[df_e['status']==1]['asc'],bins=25,alpha=.75,color='#9B59B6',label='Approved',edgecolor='none')
    ax.hist(df_e[df_e['status']==0]['asc'],bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.axvline(60,color='#FFB800',ls='--',lw=1.5,label='Min=60%')
    ax.set_title("Academic Score Distribution",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    # Chart 3: Course stream approval
    ax=axes[2]
    streams=['Engineering','Medical','Science','Commerce','Arts']
    s_rates=[df_e[df_e['course']==s]['status'].mean()*100 if (df_e['course']==s).sum()>0 else 0 for s in streams]
    clr_s=['#3498DB','#E74C3C','#27AE60','#E67E22','#9B59B6']
    bars=ax.bar([s[:6] for s in streams],s_rates,color=clr_s,edgecolor='none',alpha=.88)
    for b,v in zip(bars,s_rates):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.3,f'{v:.0f}%',
                ha='center',va='bottom',color='#DDE8F5',fontsize=8,fontweight='bold')
    ax.set_title("Approval Rate by Course",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.set_ylabel("Approval %",fontsize=8); ax.grid(axis='y',alpha=.3)
    # Chart 4: Parent income distribution
    ax=axes[3]
    ax.hist(df_e[df_e['status']==1]['pinc']/1000,bins=25,alpha=.75,color='#9B59B6',label='Approved',edgecolor='none')
    ax.hist(df_e[df_e['status']==0]['pinc']/1000,bins=25,alpha=.55,color='#E74C3C',label='Rejected',edgecolor='none')
    ax.set_title("Parent Income (₹K)",color='#DDE8F5',fontsize=10,fontweight='bold')
    ax.legend(fontsize=7,facecolor='#1E2F45',edgecolor='none',labelcolor='#DDE8F5'); ax.grid(True,alpha=.3)
    plt.tight_layout(pad=1.4); st.pyplot(fig); plt.close(fig)
    st.markdown('<hr class="dv">',unsafe_allow_html=True)

    # ════════════════════════════════════
    # Algorithm Comparison — Professional Bar Chart
    # ════════════════════════════════════
    st.markdown('<div class="sec-t">🏆 Algorithm Performance Comparison — All Loan Types</div>',unsafe_allow_html=True)

    algo_data={
        "Personal Loan":  ("Gradient Boosting",  f"{acc_p*100:.1f}%", "#4A90D9",  "GradientBoostingClassifier"),
        "Home Loan":      ("Random Forest",       f"{acc_h*100:.1f}%", "#27AE60",  "RandomForestClassifier"),
        "Vehicle Loan":   ("Extra Trees",         f"{acc_v*100:.1f}%", "#E74C3C",  "ExtraTreesClassifier"),
        "Gold Loan":      ("Random Forest",       f"{acc_g*100:.1f}%", "#FFB800",  "RandomForestClassifier"),
        "Education Loan": ("Gradient Boosting",   f"{acc_e*100:.1f}%", "#9B59B6",  "GradientBoostingClassifier"),
    }
    st.markdown("""<table class="styled-table">
    <tr><th>#</th><th>Loan Type</th><th>Best Algorithm</th><th>Sklearn Class</th>
    <th>Accuracy</th><th>Why This Algorithm?</th></tr>""",unsafe_allow_html=True)
    reasons={
        "Personal Loan": "Credit-scoring data pe GBM best perform karta hai — sequential error correction se",
        "Home Loan": "Property-backed structured data ke liye Random Forest ideal — high variance handle karta hai",
        "Vehicle Loan": "Mixed features (type+price+DTI) ke liye Extra Trees — faster aur less overfitting",
        "Gold Loan": "LTV + purity + weight features pe Random Forest best — non-linear boundaries",
        "Education Loan": "Academic + financial mix ke liye GBM — weak learners combine karke strong prediction",
    }
    icons={"Personal Loan":"💼","Home Loan":"🏠","Vehicle Loan":"🚗","Gold Loan":"💛","Education Loan":"🎓"}
    for i,(loan,(algo,acc,clr,cls)) in enumerate(algo_data.items(),1):
        st.markdown(f"""<tr>
        <td><strong>{i}</strong></td>
        <td>{icons[loan]} <strong>{loan}</strong></td>
        <td><span class="algo-badge" style="background:{clr}22;color:{clr};border:1px solid {clr}44;">{algo}</span></td>
        <td><code style="font-size:.78rem;">{cls}</code></td>
        <td><strong style="color:#1A8745;">{acc}</strong></td>
        <td style="font-size:.82rem;color:#4A6080;">{reasons[loan]}</td>
        </tr>""",unsafe_allow_html=True)
    st.markdown("</table>",unsafe_allow_html=True)
    st.markdown("")

    fig3,ax3=plt.subplots(figsize=(13,4))
    fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG)
    names=list(algo_data.keys()); accs_v=[float(v[1].replace('%','')) for v in algo_data.values()]
    clrs_v=[v[2] for v in algo_data.values()]
    bars=ax3.bar(names,accs_v,color=clrs_v,edgecolor='none',alpha=.88,width=.55)
    for b,v in zip(bars,accs_v):
        ax3.text(b.get_x()+b.get_width()/2,b.get_height()+.15,f'{v:.1f}%',
                 ha='center',va='bottom',color='#DDE8F5',fontsize=11,fontweight='bold')
    ax3.set_ylim(60,100); ax3.set_ylabel("Classifier Accuracy %",color='#DDE8F5',fontsize=10)
    ax3.set_title("Best Algorithm Accuracy per Loan Type",color='#DDE8F5',fontsize=13,fontweight='bold',pad=12)
    ax3.grid(axis='y',alpha=.3)
    for spine in ax3.spines.values(): spine.set_color('#2A3F5F')
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)


# ═══════════════════════════════════════════════════════════════
#  ABOUT PAGE
# ═══════════════════════════════════════════════════════════════

def about_page():
    st.markdown("""<div class="bank-header">
    <div class="bh-badge">📖 Project Documentation</div>
    <h1>🏦 Bank Finance System</h1>
    <p>Complete AI-powered loan intelligence platform — designed for internship project showcase</p>
    </div>""",unsafe_allow_html=True)

    sc1,sc2,sc3,sc4=st.columns(4)
    for col,(icon,title,desc) in zip([sc1,sc2,sc3,sc4],[
        ("🏦","5 Loan Tables","Personal · Home · Vehicle · Gold · Education — har ek alag model"),
        ("🌲","3 ML Algorithms","Gradient Boosting · Random Forest · Extra Trees — best per loan"),
        ("📋","Business Rules","CIBIL · DTI · Income · LTV — strict conditions pehle, ML baad mein"),
        ("📊","Smart Analytics","Cross-sell · Payment impact · Partial approval · Risk scoring"),
    ]):
        with col:
            st.markdown(f"""<div class="ac" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
            <h3 style="font-size:.98rem;">{title}</h3>
            <p style="font-size:.82rem;">{desc}</p></div>""",unsafe_allow_html=True)

    st.markdown('<hr class="dv">',unsafe_allow_html=True)
    st.markdown('<div class="sec-t">📋 5 Loan Tables — Complete Introduction</div>',unsafe_allow_html=True)

    loans_info=[
        ("💼","Personal Loan","#4A90D9","GradientBoostingClassifier",
         "Kisi bhi zarurat ke liye — no collateral, instant processing",
         """Personal Loan ek **unsecured** loan hai — ghar, gaadi ya koi bhi property collateral nahi lagti.
         Medical emergency, shaadi, travel, renovation — kisi bhi personal zarurat ke liye applicable hai.
         Approval ke liye sabse important factors hain: **CIBIL ≥650, Monthly Income ≥₹20,000,
         DTI ≤0.50, Past Defaults = 0, Bank Balance ≥₹10,000**. Loan amount income ka 30x tak milta hai.
         Interest rate **8%–22% p.a.** hoti hai. Algorithm — Gradient Boosting isliye ki credit scoring
         mein sequential error correction best results deta hai.""",
         ["No Collateral Required","Tenure 1–7 Years","Loan up to ₹25 Lakh","CIBIL ≥ 650"]),
        ("🏠","Home Loan","#27AE60","RandomForestClassifier",
         "Apna ghar — property collateral ke saath long-term housing finance",
         """Home Loan mein purchased property khud collateral hoti hai. Kyunki amount badi hoti hai
         (₹15L–₹2Cr+), conditions strict hain: **CIBIL ≥700, Income ≥₹30,000,
         Down Payment ≥15% of property value, DTI ≤0.55, Loan Inquiries ≤2**.
         Tenure 10–30 saal tak available hai. Interest rate **6.5%–14% p.a.** —
         personal se kaafi kam kyunki property collateral hai aur bank ka risk kam hota hai.
         Algorithm — Random Forest isliye ki property-backed structured data pe best accuracy deta hai.""",
         ["Property Collateral","Tenure 10–30 Years","Tax Benefits U/S 80C","CIBIL ≥ 700"]),
        ("🚗","Vehicle Loan","#E74C3C","ExtraTreesClassifier",
         "Two Wheeler se Commercial Vehicle — apni ride loan pe lo",
         """Vehicle Loan mein financed vehicle khud collateral hoti hai. Vehicle type (2W/4W/Commercial)
         approval aur amount ko affect karta hai. Key conditions: **CIBIL ≥650, Income ≥₹20,000,
         Down Payment ≥15%, DTI ≤0.50**. Derived features jaise Loan Utilization, EMI Burden aur
         Down Payment Ratio bhi model mein important hain. Tenure 1–6 saal, rate **7%–18% p.a.**.
         Algorithm — Extra Trees isliye ki mixed categorical + numerical features pe fast aur
         accurate prediction deta hai with less overfitting.""",
         ["Vehicle as Collateral","Tenure 1–6 Years","Down Payment ≥15%","CIBIL ≥ 650"]),
        ("💛","Gold Loan","#FFB800","RandomForestClassifier",
         "Sona rakho collateral mein — instant cash lo bina zyada paperwork ke",
         """Gold Loan mein applicant apna physical gold (jewellery/coins/bars) bank ke paas pledge karta hai
         aur immediate cash milti hai. CIBIL requirement relaxed hai — **CIBIL ≥550** kaafi hai!
         Critical factors: **Gold Weight (grams), Purity ≥20 Karat, LTV Ratio ≤0.75, Gold Estimated Value**.
         LTV (Loan-to-Value) matlab gold value ka maximum 75% loan milega. Tenure **3–24 months** —
         sabse short tenure. Rate **9%–18% p.a.** Algorithm — Random Forest isliye ki
         non-linear relationships (LTV + purity + weight) ko best capture karta hai.""",
         ["Gold as Collateral","Instant Processing","CIBIL ≥ 550 Only","LTV ≤ 75%"]),
        ("🎓","Education Loan","#9B59B6","GradientBoostingClassifier",
         "Har sapne ko degree mein badlo — padhai ab finance se nahi rukegi",
         """Education Loan students ko higher education (UG/PG/PhD/Diploma) ke liye diya jata hai.
         Parent ya guardian compulsory co-borrower hote hain. Key factors: **Parent CIBIL ≥700,
         Parent Income ≥₹30,000, Academic Score ≥60%, Entrance Score ≥300,
         Institute Tier** (Tier1 aur ForeignTop pe better approval chances), Total Debt Burden ≤0.55.
         Foreign study ke liye higher loan available hai. Tenure 3–15 saal, rate **5%–13% p.a.** —
         sabse kam rate kyunki education productive investment hai. Algorithm — Gradient Boosting
         academic + financial features ke complex mix ko best handle karta hai.""",
         ["UG/PG/PhD/Diploma","Parent Co-Borrower","Foreign Study Covered","Tenure 3–15 Years"]),
    ]

    for icon,name,color,algo,tagline,desc,feats in loans_info:
        c1,c2=st.columns([1,2])
        with c1:
            fh="".join([f'<span style="display:inline-block;background:rgba(0,0,0,0.05);'
                        f'border:1px solid #DDE8F5;border-radius:6px;padding:3px 10px;'
                        f'font-size:.77rem;margin:3px;color:#4A6080;">{f}</span>' for f in feats])
            st.markdown(f"""<div class="ac" style="border-left:4px solid {color};min-height:225px;">
            <div style="font-size:2.3rem;margin-bottom:8px;">{icon}</div>
            <h3 style="color:{color};font-size:1.15rem;">{name}</h3>
            <p style="font-style:italic;color:#6B80A0;font-size:.83rem;margin-bottom:12px;">{tagline}</p>
            {fh}
            <p style="margin-top:12px;font-size:.76rem;">
              <span class="algo-badge" style="background:{color}22;color:{color};border:1px solid {color}44;">
              🏆 {algo}</span></p>
            </div>""",unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="ac" style="min-height:225px;">
            <h3 style="font-size:.99rem;">📖 Detailed Introduction</h3>
            <p style="font-size:.86rem;line-height:1.75;">{desc}</p>
            </div>""",unsafe_allow_html=True)
        st.markdown("")

    st.markdown('<hr class="dv">',unsafe_allow_html=True)
    st.markdown('<div class="sec-t">🛠️ Technology Stack & Project Architecture</div>',unsafe_allow_html=True)
    t1,t2,t3,t4,t5=st.columns(5)
    for col,(icon,name,desc) in zip([t1,t2,t3,t4,t5],[
        ("🐍","Python 3","Core — pandas, numpy, scikit-learn"),
        ("🌊","Streamlit","Interactive web app framework"),
        ("🌲","Random Forest","Home + Gold loan models"),
        ("⚡","Gradient Boost","Personal + Education models"),
        ("🌳","Extra Trees","Vehicle loan model"),
    ]):
        with col:
            st.markdown(f"""<div class="ac" style="text-align:center;">
            <div style="font-size:1.9rem;">{icon}</div>
            <h3 style="font-size:.9rem;margin:8px 0 5px;">{name}</h3>
            <p style="font-size:.79rem;">{desc}</p></div>""",unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;padding:20px 0;color:#6B80A0;font-size:.79rem;">
    🏦 <strong>Bank Finance System</strong> &nbsp;|&nbsp; Internship Project &nbsp;|&nbsp;
    5 Loan Types &nbsp;|&nbsp; 3 ML Algorithms &nbsp;|&nbsp; Business Rule Engine &nbsp;|&nbsp;
    Real-time Analytics Dashboard
    </div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  LOAN PAGE TEMPLATE
# ═══════════════════════════════════════════════════════════════

def loan_header(icon,name,color,algo,tagline):
    st.markdown(f"""<div class="bank-header">
    <div class="bh-badge" style="background:linear-gradient(135deg,{color},{color}BB);">
      {icon} LOAN APPLICATION</div>
    <h1>{icon} {name}</h1>
    <p>{tagline}</p>
    <p style="margin-top:8px;font-size:.78rem;color:rgba(200,220,240,.6);">
    🏆 Best Algorithm: <strong style="color:#00C896;">{algo}</strong></p>
    </div>""",unsafe_allow_html=True)

def show_cond(checks):
    st.markdown('<div class="sec-t">📋 Mandatory Eligibility Check</div>',unsafe_allow_html=True)
    st.markdown('<div class="cond-wrap">',unsafe_allow_html=True)
    cl,cr=st.columns(2); mid=len(checks)//2+len(checks)%2
    for i,line in enumerate(checks): (cl if i<mid else cr).markdown(f"**{line}**")
    st.markdown('</div>',unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""<div class="sb-brand">
    <div style="font-size:3rem;">🏦</div>
    <h2>Bank Finance System</h2>
    <p>AI Loan Intelligence Platform</p>
    </div>""",unsafe_allow_html=True)
    st.markdown('<div class="sb-nav-head">Navigation</div>',unsafe_allow_html=True)
    nav=st.radio("Navigation",[
        "🏠 Home Dashboard","ℹ️ About Project",
        "── Applications ──",
        "💼 Personal Loan","🏠 Home Loan",
        "🚗 Vehicle Loan","💛 Gold Loan","🎓 Education Loan"
    ],index=0,label_visibility="collapsed")

    if nav not in ["🏠 Home Dashboard","ℹ️ About Project","── Applications ──"]:
        st.markdown('<hr style="border-color:rgba(0,200,150,.2);margin:14px 0;">',unsafe_allow_html=True)
        st.markdown('<div class="sb-nav-head">Customer History</div>',unsafe_allow_html=True)
        phist=st.selectbox("Payment History",["Excellent","Good","Average","Poor"],
                           index=0,label_visibility="collapsed")
    else:
        phist="Good"

    st.markdown('<hr style="border-color:rgba(0,200,150,.15);margin:14px 0;">',unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.71rem;color:rgba(200,220,240,.38);line-height:2.1;">
    ⚡ Gradient Boosting Classifier<br>🌲 Random Forest Classifier<br>
    🌳 Extra Trees Classifier<br>📋 Business Rule Engine<br>
    🎁 Cross-Sell Engine<br>📊 Analytics Dashboard
    </div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════

if nav=="🏠 Home Dashboard":
    home_page()

elif nav=="ℹ️ About Project":
    about_page()

elif nav=="── Applications ──":
    st.markdown("""<div class="bank-header"><h1>🏦 Loan Application Portal</h1>
    <p>Sidebar mein koi bhi loan type chuniye — application form show hoga</p>
    </div>""",unsafe_allow_html=True)
    st.info("👈 Sidebar se Personal / Home / Vehicle / Gold / Education Loan chuniye")

# ──────────────────────────────────────────────────────────────
# PERSONAL LOAN
# ──────────────────────────────────────────────────────────────
elif nav=="💼 Personal Loan":
    clf,amtm,ratem,le_c,le_e,F,_,_ = build_personal()
    loan_header("💼","Personal Loan","#4A90D9","Gradient Boosting Classifier",
                "Bina collateral ke — kisi bhi zarurat ke liye instant personal finance")
    st.markdown('<div class="sec-t">👤 Personal & Employment</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: age=st.number_input("Age (years)",18,70,32); city=st.selectbox("City Tier",["Metro","Urban","Semi-Urban"])
    with c2: emp=st.selectbox("Employment Type",["Salaried","Self-Employed","Business Owner"]); jt=st.number_input("Job Tenure (months)",0,360,36)
    with c3: ins=st.selectbox("Insurance Present?",["No","Yes"]); co=st.selectbox("Co-Applicant?",["No","Yes"])
    st.markdown('<div class="sec-t">💰 Financial Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: income=st.number_input("Monthly Income (₹)",5000,500000,45000,step=1000); other=st.number_input("Other Income (₹)",0,100000,5000,step=500)
    with c2: bal=st.number_input("Bank Balance 6M Avg (₹)",0,1000000,80000,step=5000); emi=st.number_input("Existing EMI (₹)",0,100000,8000,step=500)
    with c3: cibil=st.slider("CIBIL Score",300,900,720); defs=st.number_input("Past Defaults",0,10,0)
    st.markdown('<div class="sec-t">🏷️ Loan Request</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: lreq=st.number_input("Loan Amount (₹)",10000,5000000,500000,step=10000)
    with c2: tm=st.selectbox("Tenure (months)",[12,24,36,48,60,72,84],index=2)
    with c3: inq=st.number_input("Loan Inquiries (last 6M)",0,15,1)
    if st.button("🔍 Eligibility Check & Apply"):
        ti=income+other; dti=(emi+lreq/tm)/(ti+1)
        checks,passed=rules_personal(age,income,cibil,defs,dti,inq,bal,lreq)
        show_cond(checks)
        if not passed: show_result(False,0,0,0,tm,cibil,dti,defs,"Personal Loan",phist,lreq)
        else:
            try: ce=le_c.transform([city])[0]
            except: ce=0
            try: ee=le_e.transform([emp])[0]
            except: ee=0
            row=pd.DataFrame([[age,ce,ee,income,other,bal,jt,cibil,defs,inq,emi,dti,lreq,tm,int(ins=="Yes"),int(co=="Yes")]],columns=F)
            prob=clf.predict_proba(row)[0][1]; ia=clf.predict(row)[0]
            amt=amtm.predict(row)[0] if ia else 0; rate=ratem.predict(row)[0] if ia else 0
            show_result(bool(ia),prob,amt,rate,tm,cibil,dti,defs,"Personal Loan",phist,lreq)

# ──────────────────────────────────────────────────────────────
# HOME LOAN
# ──────────────────────────────────────────────────────────────
elif nav=="🏠 Home Loan":
    clf,amtm,ratem,le_c,le_e,F,_,_ = build_home()
    loan_header("🏠","Home Loan","#27AE60","Random Forest Classifier",
                "Apna ghar — property collateral ke saath long-term housing finance")
    st.markdown('<div class="sec-t">👤 Personal & Employment</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: age=st.number_input("Age",18,65,35); city=st.selectbox("City Tier",["Metro","Urban","Semi-Urban"])
    with c2: emp=st.selectbox("Employment Type",["Salaried","Self-Employed","Business Owner"]); jt=st.number_input("Job Tenure (months)",0,420,60)
    with c3: defs=st.number_input("Past Defaults",0,10,0); inq=st.number_input("Loan Inquiries (6M)",0,10,0)
    st.markdown('<div class="sec-t">💰 Financial Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: income=st.number_input("Monthly Income (₹)",5000,1000000,80000,step=5000); other=st.number_input("Other Income (₹)",0,200000,10000,step=2000)
    with c2: bal=st.number_input("Bank Balance 6M Avg (₹)",0,2000000,200000,step=10000); emi=st.number_input("Existing EMI (₹)",0,200000,15000,step=1000)
    with c3: cibil=st.slider("CIBIL Score",300,900,750)
    st.markdown('<div class="sec-t">🏡 Property & Loan Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: prop=st.number_input("Property Value (₹)",500000,100000000,5000000,step=100000); dp=st.number_input("Down Payment (₹)",0,20000000,1000000,step=50000)
    with c2: lreq=st.number_input("Loan Amount (₹)",100000,80000000,max(prop-dp,100000),step=100000); tm=st.selectbox("Tenure (months)",[120,180,240,300,360],index=2)
    with c3:
        tot=income+other; dti=(emi+lreq/tm)/(tot+1); dp_pct=dp/(prop+1)*100
        st.info(f"💡 DTI: **{dti:.3f}** (≤0.55 required)"); st.info(f"💡 Down Payment: **{dp_pct:.1f}%** (≥15% required)")
    if st.button("🔍 Eligibility Check & Apply"):
        tot=income+other; dti=(emi+lreq/tm)/(tot+1)
        checks,passed=rules_home(age,income,cibil,defs,dti,dp,prop,inq)
        show_cond(checks)
        if not passed: show_result(False,0,0,0,tm,cibil,dti,defs,"Home Loan",phist,lreq)
        else:
            try: ce=le_c.transform([city])[0]
            except: ce=0
            try: ee=le_e.transform([emp])[0]
            except: ee=0
            emr=emi/(tot+1); lir=lreq/(tot*12+1)
            row=pd.DataFrame([[age,ce,ee,income,other,bal,jt,cibil,defs,inq,emi,prop,dp,lreq,tm,tot,emr,lir,dti]],columns=F)
            prob=clf.predict_proba(row)[0][1]; ia=clf.predict(row)[0]
            amt=amtm.predict(row)[0] if ia else 0; rate=ratem.predict(row)[0] if ia else 0
            show_result(bool(ia),prob,amt,rate,tm,cibil,dti,defs,"Home Loan",phist,lreq)

# ──────────────────────────────────────────────────────────────
# VEHICLE LOAN
# ──────────────────────────────────────────────────────────────
elif nav=="🚗 Vehicle Loan":
    clf,amtm,ratem,le_c,le_e,le_v,F,_,_ = build_vehicle()
    loan_header("🚗","Vehicle Loan","#E74C3C","Extra Trees Classifier",
                "Two Wheeler, Four Wheeler ya Commercial — apni ride loan pe lo")
    st.markdown('<div class="sec-t">👤 Personal & Vehicle</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: age=st.number_input("Age",18,62,30); city=st.selectbox("City Tier",["Metro","Urban","Semi-Urban"])
    with c2: emp=st.selectbox("Employment Type",["Salaried","Self-Employed","Business Owner"]); jt=st.number_input("Job Tenure (months)",0,300,24)
    with c3: vtype=st.selectbox("Vehicle Type",["Two Wheeler","Four Wheeler","Commercial"]); inq=st.number_input("Loan Inquiries (6M)",0,10,1)
    st.markdown('<div class="sec-t">💰 Financial Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: inc=st.number_input("Monthly Income (₹)",5000,500000,40000,step=1000); bal=st.number_input("Bank Balance 6M Avg (₹)",0,1000000,60000,step=5000)
    with c2: emi=st.number_input("Existing EMI (₹)",0,100000,5000,step=500); defs=st.number_input("Past Defaults",0,10,0)
    with c3: cibil=st.slider("CIBIL Score",300,900,710)
    st.markdown('<div class="sec-t">🚗 Vehicle & Loan</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: vp=st.number_input("Vehicle Price (₹)",30000,10000000,800000,step=10000); dp=st.number_input("Down Payment (₹)",0,5000000,160000,step=10000)
    with c2: lreq=st.number_input("Loan Amount (₹)",10000,8000000,max(vp-dp,10000),step=10000); tm=st.selectbox("Tenure (months)",[12,24,36,48,60,72],index=2)
    with c3:
        dp_pct=dp/(vp+1)*100; dti_v=(emi+lreq/tm)/(inc+1)
        st.info(f"💡 Down Payment: **{dp_pct:.1f}%** (≥15% required)"); st.info(f"💡 DTI: **{dti_v:.3f}** (≤0.50 required)")
    if st.button("🔍 Eligibility Check & Apply"):
        dti_v=(emi+lreq/tm)/(inc+1)
        checks,passed=rules_vehicle(age,inc,cibil,defs,dti_v,dp,vp,inq)
        show_cond(checks)
        if not passed: show_result(False,0,0,0,tm,cibil,dti_v,defs,"Vehicle Loan",phist,lreq)
        else:
            try: ce=le_c.transform([city])[0]
            except: ce=0
            try: ee=le_e.transform([emp])[0]
            except: ee=0
            try: ve=le_v.transform([vtype])[0]
            except: ve=0
            lu=lreq/(vp+1); eb=emi/(inc+1); ni=inc-emi; dpr=dp/(vp+1); b2i=bal/(inc+1); hr=int((defs>0)and(cibil<650))
            row=pd.DataFrame([[age,ce,ee,ve,inc,bal,jt,cibil,defs,inq,emi,vp,dp,lreq,tm,lu,eb,ni,dpr,b2i,dti_v,hr]],columns=F)
            prob=clf.predict_proba(row)[0][1]; ia=clf.predict(row)[0]
            amt=amtm.predict(row)[0] if ia else 0; rate=ratem.predict(row)[0] if ia else 0
            show_result(bool(ia),prob,amt,rate,tm,cibil,dti_v,defs,"Vehicle Loan",phist,lreq)

# ──────────────────────────────────────────────────────────────
# GOLD LOAN
# ──────────────────────────────────────────────────────────────
elif nav=="💛 Gold Loan":
    clf,amtm,ratem,le_g,le_c,le_e,F,_,_ = build_gold()
    loan_header("💛","Gold Loan","#FFB800","Random Forest Classifier",
                "Apna sona pledge karo — instant cash lo bina zyada paperwork ke")
    st.markdown('<div class="sec-t">👤 Personal Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: age=st.number_input("Age",18,70,38); gen=st.selectbox("Gender",["Male","Female"])
    with c2: city=st.selectbox("City Tier",["Metro","Urban","Rural"]); emp=st.selectbox("Employment",["Salaried","Self-Employed","Farmer","Business Owner"])
    with c3: cibil=st.slider("CIBIL Score",300,900,640); defs=st.number_input("Past Defaults",0,10,0)
    st.markdown('<div class="sec-t">💰 Financial Details</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: inc=st.number_input("Monthly Income (₹)",5000,300000,35000,step=1000)
    with c2: emi=st.number_input("Existing EMI (₹)",0,80000,3000,step=500)
    st.markdown('<div class="sec-t">💛 Gold Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: gw=st.number_input("Gold Weight (grams)",1.0,500.0,50.0,step=0.5); gpur=st.selectbox("Gold Purity (Karat)",[18,20,22,24],index=2)
    with c2:
        gval=int(gw*gpur*300); st.metric("📊 Estimated Gold Value",f"₹{gval:,}"); ltv=st.slider("LTV Ratio",0.50,0.85,0.70)
    with c3:
        max_l=int(gval*ltv); st.metric("💰 Max Eligible Amount",f"₹{max_l:,}")
        lreq=st.number_input("Loan Amount (₹)",1000,max_l+1,min(max_l,100000),step=1000); tm=st.selectbox("Tenure (months)",[3,6,9,12,18,24],index=3)
    if st.button("🔍 Eligibility Check & Apply"):
        dti_g=(emi+lreq/tm)/(inc+1)
        checks,passed=rules_gold(age,inc,cibil,ltv,dti_g,gpur)
        show_cond(checks)
        if not passed: show_result(False,0,0,0,tm,cibil,dti_g,defs,"Gold Loan",phist,lreq)
        else:
            try: ge=le_g.transform([gen])[0]
            except: ge=0
            try: ce=le_c.transform([city])[0]
            except: ce=0
            try: ee=le_e.transform([emp])[0]
            except: ee=0
            gu=lreq/(gval+1); eb=emi/(inc+1); lpg=lreq/(gw+1); ni=inc-emi
            row=pd.DataFrame([[age,ge,ce,ee,inc,emi,cibil,gw,gpur,gval,ltv,lreq,tm,gu,eb,lpg,ni,dti_g]],columns=F)
            prob=clf.predict_proba(row)[0][1]; ia=clf.predict(row)[0]
            amt=amtm.predict(row)[0] if ia else 0; rate=ratem.predict(row)[0] if ia else 0
            show_result(bool(ia),prob,amt,rate,tm,cibil,dti_g,defs,"Gold Loan",phist,lreq)

# ──────────────────────────────────────────────────────────────
# EDUCATION LOAN
# ──────────────────────────────────────────────────────────────
elif nav=="🎓 Education Loan":
    clf,amtm,ratem,F,_,CMAP,LMAP,TMAP,_ = build_education()
    loan_header("🎓","Education Loan","#9B59B6","Gradient Boosting Classifier",
                "Higher education ke liye — India ya videsh mein, sapne ko degree mein badlo")
    st.markdown('<div class="sec-t">🎓 Student Academic Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: age=st.number_input("Student Age",15,40,21); cs_v=st.selectbox("Course Stream",["Engineering","Medical","Arts","Commerce","Science"])
    with c2: cl_v=st.selectbox("Course Level",["UG","PG","PhD","Diploma"]); it_v=st.selectbox("Institute Tier",["Tier1","Tier2","Tier3","ForeignTop","ForeignOther"])
    with c3: cn=st.selectbox("Country",["India","USA","UK","Australia","Canada","Germany"]); asc=st.slider("Academic Score (%)",30.0,100.0,76.0)
    c1,c2=st.columns(2)
    with c1: esc=st.number_input("Entrance Score (0–800)",0,800,440)
    with c2: ty=st.selectbox("Loan Tenure (years)",[3,5,7,10,12,15],index=2)
    st.markdown('<div class="sec-t">👨‍👩‍👦 Parent / Co-Borrower Details</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: pinc=st.number_input("Parent Monthly Income (₹)",5000,1000000,75000,step=5000); pcib=st.slider("Parent CIBIL Score",300,900,735)
    with c2: pdti=st.slider("Parent DTI",0.0,0.9,0.28); pemi=st.number_input("Parent Existing EMI (₹)",0,200000,8000,step=1000)
    with c3: tuit=st.number_input("Annual Tuition Fee (₹)",10000,5000000,500000,step=10000); liv=st.number_input("Annual Living Expenses (₹)",10000,2000000,200000,step=10000)
    st.markdown('<div class="sec-t">🏷️ Loan Request</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: dp=st.number_input("Down Payment (₹)",0,500000,50000,step=10000)
    with c2:
        tc=(tuit+liv)*ty; lreq=st.number_input("Loan Amount (₹)",50000,10000000,min(tc-dp,5000000),step=50000)
    with c3:
        ee_emi=lreq/(ty*12+1); tdb=(pemi+ee_emi)/(pinc+1)
        st.info(f"💡 Total Education Cost: **₹{tc:,}**"); st.info(f"💡 Total Debt Burden: **{tdb:.3f}** (≤0.55)")
    if st.button("🔍 Eligibility Check & Apply"):
        ee_emi=lreq/(ty*12+1); tdb=(pemi+ee_emi)/(pinc+1)
        checks,passed=rules_education(age,pinc,pcib,asc,esc,tdb,lreq)
        show_cond(checks)
        if not passed: show_result(False,0,0,0,ty*12,pcib,tdb,0,"Education Loan",phist,lreq)
        else:
            cse=CMAP.get(cs_v,2); cle=LMAP.get(cl_v,1); te=TMAP.get(it_v,1)
            isfor=int(cn!="India"); istop=int(te>=2); l2i=lreq/(pinc+1); fc=lreq/(tc+1)
            sc_c=(asc/100)+(esc/800)
            row=pd.DataFrame([[age,asc,esc,cse,cle,te,isfor,pinc,pcib,pdti,pemi,lreq,tuit,liv,dp,ty,tc,l2i,fc,ee_emi,tdb,istop,sc_c]],columns=F)
            prob=clf.predict_proba(row)[0][1]; ia=clf.predict(row)[0]
            amt=amtm.predict(row)[0] if ia else 0; rate=ratem.predict(row)[0] if ia else 0
            show_result(bool(ia),prob,amt,rate,ty*12,pcib,tdb,0,"Education Loan",phist,lreq)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown('<hr class="dv">',unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#8899AA;font-size:.76rem;padding:6px 0 16px;">
🏦 <strong>Bank Finance System</strong> &nbsp;·&nbsp; &nbsp;·&nbsp;
5 Loan Types &nbsp;·&nbsp; Gradient Boosting · Random Forest · Extra Trees &nbsp;·&nbsp;
Business Rule Engine &nbsp;·&nbsp; AI-Powered Decisions
</div>""",unsafe_allow_html=True)
