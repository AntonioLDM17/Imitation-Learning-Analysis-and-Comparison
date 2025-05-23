import json, zipfile, pathlib, pprint

ckpt = pathlib.Path("gaifo_halfcheetah_2_disc_epochs10_batch_size256_lambda_gp1.660941233998641_disc_lr3.989020006157259e-05_iterations300.zip")

def get_hp(path):
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("data")          # tu checkpoint guarda el JSON aquí
    data = json.loads(raw.decode())
    # Campos que sí necesitas para lanzar la misma run
    keep = ["seed", "n_steps", "batch_size", "learning_rate",
            "gamma", "gae_lambda",
            "cg_max_steps", "cg_damping",
            "target_kl", "n_critic_updates"]
    return {k: data[k] for k in keep}

hp = get_hp(ckpt)
print("Hiper-parámetros reproducibles:")
pprint.pprint(hp, sort_dicts=False)
