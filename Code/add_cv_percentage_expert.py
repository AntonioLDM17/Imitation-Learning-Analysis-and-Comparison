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
        # Accept both comma and dot as decimal separators
        for dec in [".", ","]:
            try:
                return pd.read_csv(path, decimal=dec)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read CSV file with either decimal separator.")
    else:
        raise ValueError("File format not supported. Use CSV or XLSX.")

def add_metrics(df):
    df.columns = [c.strip().lower() for c in df.columns]

    # localizar filas con “EXPERT”
    mask_expert = df["algoritmo"].str.contains("expert", flags=re.I, na=False)
    if mask_expert.sum() == 0:
        raise ValueError("The input file must contain at least one row with 'expert' in the 'algoritmo' column.")

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
        raise ValueError("Output format not supported. Use CSV or XLSX.")

def main(infile, outfile):
    df = read_table(infile)
    df = add_metrics(df)
    write_table(df, outfile)
    print("File saved successfully to", outfile)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--infile",  required=True, help="input file (csv/xlsx)")
    p.add_argument("--outfile", required=True, help="output file (csv/xlsx)")
    args = p.parse_args()
    main(args.infile, args.outfile)
