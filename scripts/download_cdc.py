"""
Download do CDC Diabetes Health Indicators via UCI ML Repository.
Salva o dataset completo (253.680 instâncias, 21 features) em data/cdc_diabetes.csv

Uso:
    pip install ucimlrepo pandas
    python scripts/download_cdc.py
"""

import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "cdc", "cdc_diabetes.csv")


def main():
    print("Baixando CDC Diabetes Health Indicators (id=891)...")
    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features  # 21 features
    y = dataset.data.targets   # Diabetes_012

    # Junta features + target em um único DataFrame
    df = pd.concat([X, y], axis=1)

    print(f"  Instâncias: {len(df):,}")
    print(f"  Features:   {X.shape[1]}")
    print(f"  Target:     {y.columns.tolist()}")
    print(f"  Missing:    {df.isnull().sum().sum()}")
    print()

    # Distribuição das classes
    print("Distribuição das classes:")
    counts = y.value_counts()
    for cls, count in counts.items():
        pct = count / len(y) * 100
        print(f"  {cls}: {count:>7,} ({pct:.1f}%)")
    print()

    # Salva como CSV com separador ; (mesmo padrão dos datasets do baseline)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, sep=";")
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Salvo em: {OUTPUT_FILE}")
    print(f"Tamanho:  {size_mb:.1f} MB")

    # Mostra primeiras linhas como verificação
    print(f"\nPrimeiras 3 linhas:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
