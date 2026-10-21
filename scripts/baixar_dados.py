"""Baixa o arquivo de avaliações do Olist (espelho no Hugging Face) e valida.

Uso: uv run python scripts/baixar_dados.py
Dados originais: Brazilian E-Commerce Public Dataset by Olist (Kaggle), licença CC BY-NC-SA 4.0.
"""
import hashlib
from pathlib import Path

import pandas as pd
import requests

# espelho público do arquivo original (o Kaggle exige credencial)
URL = ("https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main/"
       "olist_order_reviews_dataset.csv")
SHA256 = "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7"
DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw" / "olist_order_reviews_dataset.csv"

COLUNAS = ["review_id", "order_id", "review_score", "review_comment_title",
           "review_comment_message", "review_creation_date", "review_answer_timestamp"]
LINHAS_ESPERADAS = 99224


def sha256_do_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    if DESTINO.exists():
        print(f"{DESTINO} já existe; só vou validar.")
    else:
        DESTINO.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {URL}")
        resposta = requests.get(URL, timeout=120)
        resposta.raise_for_status()
        DESTINO.write_bytes(resposta.content)

    # validações
    assert sha256_do_arquivo(DESTINO) == SHA256, "sha256 diferente do esperado"
    df = pd.read_csv(DESTINO)
    assert list(df.columns) == COLUNAS, f"colunas inesperadas: {list(df.columns)}"
    assert len(df) == LINHAS_ESPERADAS, f"linhas inesperadas: {len(df)}"
    assert df["review_score"].between(1, 5).all(), "nota fora de 1 a 5"
    print(f"OK: {len(df)} linhas, colunas {COLUNAS}")


if __name__ == "__main__":
    main()
