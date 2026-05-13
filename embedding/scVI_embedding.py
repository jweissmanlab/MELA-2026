"""Train scVI model and save latent embeddings to data/scvi.csv."""

from pathlib import Path
import pandas as pd
import treedata as td
import scvi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "scvi_model"

N_LATENT = 50
N_LAYERS = 2
N_HIDDEN = 1024
MAX_EPOCHS = 15
BATCH_SIZE = 512

print("Loading data...")
tdata = td.read_h5td(DATA_DIR / "counts.h5td")
hvgs = pd.read_csv(DATA_DIR / "hvg.csv", header=None)[0].tolist()
tdata = tdata[:, hvgs].copy()
print(f"{tdata.n_obs} cells, {tdata.n_vars} genes")

print("Training scVI model...")
scvi.model.SCVI.setup_anndata(tdata, categorical_covariate_keys = ["embryo","stage"])
model = scvi.model.SCVI(
    tdata,
    n_latent=N_LATENT,
    n_layers=N_LAYERS,
    n_hidden=N_HIDDEN,
    gene_likelihood="nb",
)
model.train(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
    early_stopping_patience=10,plan_kwargs={"kl_weight": 0.0},
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_DIR, overwrite=True)

embeddings = model.get_latent_representation()
print(f"Embeddings shape: {embeddings.shape}")

pd.DataFrame(
    embeddings,
    index=tdata.obs_names,
    columns=[f"scVI{i}" for i in range(N_LATENT)],
).to_csv(DATA_DIR / "scvi.csv")
print(f"Embeddings saved to {DATA_DIR / 'scvi.csv'}")
