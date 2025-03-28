import os

def patch_file(filepath):
    if not os.path.exists(filepath):
        print("No se encontró el archivo:", filepath)
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = content.replace("'-fopenmp',", "")
    new_content = new_content.replace('"-fopenmp",', "")
    # Elimina cualquier aparición suelta de -fopenmp
    new_content = new_content.replace("-fopenmp", "")
    if new_content == content:
        print("No se encontró '-fopenmp' en", filepath)
        return False
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Parche aplicado exitosamente en", filepath)
        return True

# Ruta al archivo builder.py de mujoco-py
builder_path = os.path.join(
    os.getcwd(), ".venv", "Lib", "site-packages", "mujoco_py", "builder.py"
)

patched = patch_file(builder_path)

if patched:
    # Opcional: eliminar la carpeta generada para forzar recompilación
    gen_dir = os.path.join(
        os.getcwd(), ".venv", "Lib", "site-packages", "mujoco_py", "generated"
    )
    if os.path.exists(gen_dir):
        try:
            import shutil
            shutil.rmtree(gen_dir)
            print("Se eliminó la carpeta 'generated' para forzar recompilación.")
        except Exception as e:
            print("Error al eliminar la carpeta generated:", e)
    else:
        print("No se encontró la carpeta 'generated'.")
else:
    print("El parche no se aplicó, tal vez ya estaba aplicado.")
