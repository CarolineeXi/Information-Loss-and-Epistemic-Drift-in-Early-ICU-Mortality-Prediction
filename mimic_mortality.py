"""
mimic_mortality_fullpaper.py
FAST + STABLE VERSION FOR FINAL PAPER
- ICU Early Labs: 36h, 24h, 12h
- LR, GBM, Transformer
- Feature importance plots
- AUROC plot
- Transformer encoder heatmap
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Create output directory
output_dir = "/Users/xiyuehuang/Desktop/FInal Paper 560/outputs"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------
# SAFE AUROC
# -------------------------------------------------------
def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)

# -------------------------------------------------------
# TRANSFORMER SETUP
# -------------------------------------------------------
USE_TRANSFORMER = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    print("Transformer ENABLED")
except:
    USE_TRANSFORMER = False
    print("Transformer DISABLED")

def train_transformer(Xtr, ytr, Xte, yte, epochs=3):
    if not USE_TRANSFORMER:
        return np.nan, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr.values, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=256)

    n_features = Xtr.shape[1]
    d_model = 32

    class TabTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_proj = nn.Linear(1, d_model)
            self.field_emb  = nn.Embedding(n_features, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.out = nn.Linear(d_model, 1)

        def forward(self, x):
            B,F = x.shape
            idx = torch.arange(F, device=x.device).repeat(B,1)
            v = self.value_proj(x.unsqueeze(-1))
            f = self.field_emb(idx)
            h = self.encoder(v+f)
            return self.out(h.mean(1)).squeeze(-1)

    model = TabTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(probs)
            trues.append(yb.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return safe_auc(trues, preds), model


# -------------------------------------------------------
# LOAD MIMIC-IV DATA
# -------------------------------------------------------
data_dir = "/Users/xiyuehuang/Desktop/FInal Paper 560"

admissions = pd.read_csv(f"{data_dir}/admissions.csv", low_memory=False)
for c in ["admittime","dischtime","deathtime"]:
    admissions[c] = pd.to_datetime(admissions[c], errors="coerce")

icustays = pd.read_csv(f"{data_dir}/icustays.csv", low_memory=False)
for c in ["intime","outtime"]:
    icustays[c] = pd.to_datetime(icustays[c], errors="coerce")

labs = pd.read_csv(
    f"{data_dir}/labevents.csv",
    nrows=500000,
    low_memory=False
)
labs["charttime"]=pd.to_datetime(labs["charttime"], errors="coerce")

items = pd.read_csv(f"{data_dir}/d_labitems.csv")
items["label"]=items["label"].str.lower()
labs = labs.merge(items[["itemid","label"]], on="itemid", how="left")


# -------------------------------------------------------
# MORTALITY LABEL
# -------------------------------------------------------
icu_first = (
    icustays.sort_values(["hadm_id","intime"])
            .groupby("hadm_id").first().reset_index()
)
merged = admissions.merge(
    icu_first[["hadm_id","intime","outtime","los"]],
    on="hadm_id", how="inner"
)

valid = merged.dropna(subset=["outtime"]).copy()
days = (valid["deathtime"] - valid["outtime"]).dt.total_seconds()/86400

valid["mortality_30d"] = (
    (~valid["deathtime"].isna()) & (days<=30)
).astype(int)

y = valid[["hadm_id","mortality_30d"]]
print("Mortality prevalence =", y["mortality_30d"].mean())


# -------------------------------------------------------
# LAB WINDOWS
# -------------------------------------------------------
target_labs = [
    "creatinine","sodium","potassium","bun","glucose",
    "hemoglobin","white blood cells","platelets"
]

labs = labs[labs["label"].isin(target_labs)]

anchor = icu_first[["hadm_id","intime"]].rename(columns={"intime":"anchor_time"})
labs = labs.merge(anchor, on="hadm_id", how="left")

def summarize(hours):
    df=labs.copy()
    S=df["anchor_time"]
    E=df["anchor_time"]+pd.to_timedelta(hours, "hour")
    df=df[(df["charttime"]>=S)&(df["charttime"]<=E)]

    agg=df.groupby(["hadm_id","label"])["valuenum"].agg(
        ["mean","min","max","first","last"]
    ).reset_index()
    agg["delta"]=agg["last"]-agg["first"]
    wide=agg.pivot(index="hadm_id", columns="label")
    wide.columns=[f"{a}_{b}" for a,b in wide.columns]
    return wide.reset_index()

labs36=summarize(36)
labs24=summarize(24)
labs12=summarize(12)


# -------------------------------------------------------
# BASELINE
# -------------------------------------------------------
base = valid[["hadm_id","admission_type","insurance","race","los"]]
base = pd.get_dummies(base, drop_first=True)


# -------------------------------------------------------
# BUILD MATRIX
# -------------------------------------------------------
def build_matrix(labdf, disc):
    df=y.merge(labdf, on="hadm_id", how="left").merge(base, on="hadm_id", how="left")

    lab_cols = [c for c in df.columns if any(lb in c for lb in target_labs)]

    # impute numeric
    for c in df.columns:
        if df[c].dtype in [np.float64,np.int64]:
            df[c]=df[c].fillna(df[c].median())

    # discretization
    if disc=="moderate":
        for c in lab_cols:
            q=np.percentile(df[c],[33,66])
            df[c]=np.digitize(df[c],q)
    elif disc=="high":
        for c in lab_cols:
            q1,q3=np.percentile(df[c],[25,75])
            df[c]=((df[c]<q1)|(df[c]>q3)).astype(int)

    X=df.drop(columns=["hadm_id","mortality_30d"])
    return X,df["mortality_30d"]


X1,y1=build_matrix(labs36,"none")
X2,y2=build_matrix(labs24,"moderate")
X3,y3=build_matrix(labs12,"high")


# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def run_model(X,y,name):
    print(f"\n=== TRAINING {name} ===")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y)

    scaler=StandardScaler()
    Xtr_s=scaler.fit_transform(Xtr)
    Xte_s=scaler.transform(Xte)

    # LR
    lr=LogisticRegression(max_iter=1500,class_weight="balanced")
    lr.fit(Xtr_s,ytr)
    auc_lr=safe_auc(yte, lr.predict_proba(Xte_s)[:,1])

    # GBM
    gb=GradientBoostingClassifier(n_estimators=120)
    gb.fit(Xtr_s,ytr)
    auc_gb=safe_auc(yte, gb.predict_proba(Xte_s)[:,1])

    # Transformer
    auc_tf, tf_model = train_transformer(Xtr_s,ytr,Xte_s,yte)

    print(f"{name}: LR={auc_lr:.3f}, GB={auc_gb:.3f}, TF={auc_tf:.3f}")

    # Feature importance plots (GBM)
    gb_imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8,6))
    gb_imp.sort_values().plot(kind="barh")
    plt.title(f"Top 20 GBM Features – {name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_{name}.png")
    plt.close()

    # LR coefficients
    lr_imp=pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8,6))
    lr_imp.sort_values().plot(kind="barh")
    plt.title(f"Top 20 LR Coefficients – {name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lr_coeff_{name}.png")
    plt.close()

    # Transformer heatmap
    if tf_model is not None:
        with torch.no_grad():
            sample=torch.tensor(Xte_s[0:1],dtype=torch.float32)
            B,F=sample.shape
            idx=torch.arange(F)
            v=tf_model.value_proj(sample.unsqueeze(-1))
            f=tf_model.field_emb(idx)
            h=tf_model.encoder(v+f)[0].cpu().numpy()

        plt.figure(figsize=(10,4))
        plt.imshow(h,aspect="auto",cmap="viridis")
        plt.title(f"Transformer Encoder Map – {name}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tf_map_{name}.png")
        plt.close()

    return {"pipeline":name,"LR":auc_lr,"GB":auc_gb,"TF":auc_tf}


# -------------------------------------------------------
# RUN THE 3 PIPELINES
# -------------------------------------------------------
res=[]
res.append(run_model(X1,y1,"high_preservation_36h"))
res.append(run_model(X2,y2,"moderate_loss_24h"))
res.append(run_model(X3,y3,"high_loss_12h"))

res_df=pd.DataFrame(res)
print("\n=== FINAL RESULTS ===")
print(res_df)

# AUROC plot
plt.figure(figsize=(7,5))
plt.plot(res_df["pipeline"],res_df["LR"],marker="o",label="LR")
plt.plot(res_df["pipeline"],res_df["GB"],marker="o",label="GBM")
plt.plot(res_df["pipeline"],res_df["TF"],marker="o",label="Transformer")
plt.xticks(rotation=25)
plt.ylabel("AUROC")
plt.title("AUROC vs Information Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/auroc_pipelines.png")
plt.close()

# Save JSON summary
with open(f"{output_dir}/results_summary.json","w") as f:
    json.dump(res, f, indent=2)