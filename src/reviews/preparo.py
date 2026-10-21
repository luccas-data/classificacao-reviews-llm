"""Limpeza dos dados, criação dos rótulos e split treino/teste."""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_BRUTO = RAIZ / "data" / "raw" / "olist_order_reviews_dataset.csv"
ARQUIVO_IDS_TESTE = RAIZ / "data" / "ids_teste.csv"

SEMENTE = 42


def limpar_texto(texto: str) -> str:
    """Limpeza leve: minúsculas, sem quebras de linha e sem espaços repetidos."""
    texto = texto.lower()
    texto = texto.replace("\r", " ").replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def nota_para_rotulo(nota: int) -> str | None:
    """1-2 = negativo, 4-5 = positivo. A nota 3 (neutra) devolve None."""
    if nota <= 2:
        return "negativo"
    if nota >= 4:
        return "positivo"
    return None


def carregar_bruto(caminho: Path = ARQUIVO_BRUTO) -> pd.DataFrame:
    """Lê o CSV original do Olist (baixe antes com scripts/baixar_dados.py)."""
    return pd.read_csv(caminho)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Fica só com avaliações que têm texto e nota diferente de 3.

    Devolve as colunas: review_id, nota, texto, rotulo.
    """
    df = df[df["review_comment_message"].notna()].copy()
    df["texto"] = df["review_comment_message"].apply(limpar_texto)
    df = df[df["texto"] != ""]  # textos que eram só espaços

    df["nota"] = df["review_score"]
    df["rotulo"] = df["nota"].apply(nota_para_rotulo)
    df = df[df["rotulo"].notna()]  # nota 3 fica de fora

    # no Olist alguns review_id aparecem repetidos; o cache usa review_id como chave
    df = df.drop_duplicates(subset="review_id")
    return df[["review_id", "nota", "texto", "rotulo"]].reset_index(drop=True)


def dividir(df: pd.DataFrame, salvar_ids: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80/20 estratificado pelo rótulo. Opcionalmente salva os ids do teste."""
    treino, teste = train_test_split(
        df, test_size=0.2, stratify=df["rotulo"], random_state=SEMENTE
    )
    treino = treino.reset_index(drop=True)
    teste = teste.reset_index(drop=True)
    if salvar_ids:
        teste[["review_id"]].to_csv(ARQUIVO_IDS_TESTE, index=False)
    return treino, teste


def amostrar_teste(teste: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Amostra estratificada de n avaliações do teste (mesma amostra sempre)."""
    amostra, _ = train_test_split(
        teste, train_size=n, stratify=teste["rotulo"], random_state=SEMENTE
    )
    return amostra.reset_index(drop=True)


def carregar_dados(salvar_ids: bool = False):
    """Atalho: bruto -> preparado -> (treino, teste)."""
    return dividir(preparar(carregar_bruto()), salvar_ids=salvar_ids)
