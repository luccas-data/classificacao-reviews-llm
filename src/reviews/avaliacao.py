"""Métricas e comparação entre baseline, LLM zero-shot e LLM few-shot."""
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from reviews.llm import MOTIVOS, chave_cache, custo_dolares

CLASSES = ["negativo", "positivo"]
CLASSES_MATRIZ = CLASSES + ["falha"]  # "falha" = LLM sem resposta válida


def metricas(reais, previstos) -> dict:
    """Acurácia, F1 macro e matriz de confusão (linhas = real, colunas = previsto).

    Previsões ausentes (None, falha do LLM) contam como erro na acurácia e aparecem na
    coluna "falha" da matriz.
    """
    previstos = [p if p in CLASSES else "falha" for p in previstos]
    return {
        "acuracia": accuracy_score(reais, previstos),
        "f1_macro": f1_score(reais, previstos, labels=CLASSES, average="macro"),
        "matriz": confusion_matrix(reais, previstos, labels=CLASSES_MATRIZ),
    }


def registros_da_amostra(amostra: pd.DataFrame, modo: str, cache: dict) -> list[dict | None]:
    """Busca no cache a resposta de cada review da amostra (None se não houver)."""
    return [cache.get(chave_cache(r, modo)) for r in amostra["review_id"]]


def resumo_llm(registros: list) -> dict:
    """Custo (US$), tempo total (s), nº de falhas e respostas faltando."""
    feitos = [r for r in registros if r is not None]
    return {
        "custo": custo_dolares(sum(r["tokens_entrada"] for r in feitos),
                               sum(r["tokens_saida"] for r in feitos)),
        "tempo": sum(r["segundos"] for r in feitos),
        "falhas": sum(not r["ok"] for r in feitos),
        "faltando": len(registros) - len(feitos),
    }


def contar_motivos(registros: list) -> pd.Series:
    """Quantas avaliações negativas por motivo (só respostas válidas)."""
    contagem = Counter(r["motivo"] or "sem motivo" for r in registros
                       if r is not None and r["ok"] and r["sentimento"] == "negativo")
    ordem = MOTIVOS + [m for m in contagem if m not in MOTIVOS]
    return pd.Series({m: contagem.get(m, 0) for m in ordem})


def comparar(amostra: pd.DataFrame, previsoes_baseline, registros_por_modo: dict) -> pd.DataFrame:
    """Tabela comparando os métodos NA MESMA amostra.

    registros_por_modo: {"zero-shot": [...], "few-shot": [...]} (modos sem respostas são ignorados).
    """
    reais = list(amostra["rotulo"])
    linhas = {}
    m = metricas(reais, previsoes_baseline)
    linhas["Baseline (TF-IDF + LR)"] = {"acuracia": m["acuracia"], "f1_macro": m["f1_macro"],
                                        "custo_US$": 0.0, "tempo_s": None, "falhas": 0, "faltando": 0}
    for modo, registros in registros_por_modo.items():
        if all(r is None for r in registros):
            continue
        previstos = [r["sentimento"] if r and r["ok"] else None for r in registros]
        res = resumo_llm(registros)
        acuracia = f1 = None
        if res["faltando"] > 0:
            # amostra incompleta: não dá para chamar isso de desempenho do modelo
            print(f"AVISO: LLM {modo} tem {res['faltando']} avaliações sem resposta no cache; "
                  "métricas não calculadas. Rode o comando com --confirmar.")
        else:
            m = metricas(reais, previstos)
            acuracia, f1 = m["acuracia"], m["f1_macro"]
        linhas[f"LLM {modo}"] = {"acuracia": acuracia, "f1_macro": f1, "custo_US$": res["custo"],
                                 "tempo_s": res["tempo"], "falhas": res["falhas"],
                                 "faltando": res["faltando"]}
    return pd.DataFrame(linhas).T
