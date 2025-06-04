import pandas as pd, argparse, os, re, subprocess, sys

def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ModuleNotFoundError:
            print("openpyxl no encontrado. Instalando…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
            return pd.read_excel(path)
    elif ext == ".csv":
        # acepta coma o punto como separador decimal
        for dec in [".", ","]:
            try:
                return pd.read_csv(path, decimal=dec)
            except UnicodeDecodeError:
                continue
        raise ValueError("No se pudo leer el CSV con codificación estándar.")
    else:
        raise ValueError("Extensión de archivo no soportada.")

def add_metrics(df):
    df.columns = [c.strip().lower() for c in df.columns]

    # localizar filas con “EXPERT”
    mask_expert = df["algoritmo"].str.contains("expert", flags=re.I, na=False)
    if mask_expert.sum() == 0:
        raise ValueError("No hay fila con 'EXPERT' en la columna algoritmo.")

    expert_means = df.loc[mask_expert].set_index("env")["media"].to_dict()

    df["cv"] = df["std"] / df["media"]
    df["% expert"] = [
        (row.media / expert_means.get(row.env, float("nan"))) * 100
        if not is_exp else 100.0
        for row, is_exp in zip(df.itertuples(), mask_expert)
    ]
    return df

def write_table(df, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif ext == ".csv":
        dec = "," if df["media"].astype(str).str.contains(",").any() else "."
        df.to_csv(path, index=False, decimal=dec)
    else:
        raise ValueError("Extensión de salida no soportada.")

def main(infile, outfile):
    df = read_table(infile)
    df = add_metrics(df)
    write_table(df, outfile)
    print("Archivo guardado con columnas 'cv' y '% expert' ➜", outfile)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--infile",  required=True, help="CSV o XLSX de entrada")
    p.add_argument("--outfile", required=True, help="Ruta de salida (csv/xlsx)")
    args = p.parse_args()
    main(args.infile, args.outfile)
