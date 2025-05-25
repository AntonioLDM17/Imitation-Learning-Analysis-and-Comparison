"""from sb3_contrib import TRPO
import torch, zipfile, pickle, json, pprint

path = "gaifo_halfcheetah_2_disc_epochs10_batch_size256_lambda_gp1.660941233998641_disc_lr3.989020006157259e-05_iterations300.zip"

# 1) Cargar el modelo (esto solo necesita SB3)
model = TRPO.load(path)

# 2) Imprimir hiper-parámetros relevantes
print("TRPO kwargs:")
pprint.pprint(model.policy_kwargs)
print("learning_rate:", model.learning_rate)
print("n_steps (rollout_length):", model.n_steps)
print("gamma:", model.gamma)
print("gae_lambda:", model.gae_lambda)
print("max_kl:", model.max_kl)
print("seed:", model.seed)
"""
import zipfile, pathlib, pprint, pickle, json

path = pathlib.Path("gail_halfcheetah_1000000_4.zip")

def load_sb3_metadata(zip_path: pathlib.Path):
    """Return the dict con todos los hiper-parámetros guardados por SB3,
    sea pickle (v≤1.7) o JSON (algunas builds custom)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "data.pkl" in names:                             # formato clásico ≤1.5
            raw = zf.read("data.pkl")
            return pickle.loads(raw)
        elif "data/params.pkl" in names:                    # formato 1.8+
            raw = zf.read("data/params.pkl")
            return pickle.loads(raw)
        elif "data" in names:                               # tu caso: JSON
            raw = zf.read("data")
            if raw[:1] in (b"{", b"["):                     # comienza con llave
                return json.loads(raw.decode())             # <-- parsea JSON
            else:                                           # por si fuera pickle binario sin extensión
                return pickle.loads(raw)
        else:
            raise FileNotFoundError("No metadata file found in zip")

meta = load_sb3_metadata(path)

# En algunos dumps JSON la estructura va anidada:
# { "data": { "hyperparameters": {...}, "policy_kwargs": {...} } }
if "data" in meta and "hyperparameters" in meta["data"]:
    meta = meta["data"]

print("\n== TRPO hyper-parameters ==")
pprint.pprint(meta.get("hyperparameters", meta), sort_dicts=False)

print("\n== Policy kwargs ==")
pprint.pprint(meta.get("policy_kwargs", {}), sort_dicts=False)

# We write everything to a file
with open("model_tracking.json", "w") as f:
    json.dump(meta, f, indent=4)
