import pandas as pd

from reviews.avaliacao import comparar, metricas


def _reg(sentimento, ok=True):
    return {"ok": ok, "sentimento": sentimento, "motivo": None, "tokens_entrada": 100,
            "tokens_saida": 10, "segundos": 1.0}


def test_falha_conta_na_matriz_e_na_acuracia():
    m = metricas(["negativo", "positivo"], ["negativo", None])
    assert m["acuracia"] == 0.5
    assert m["matriz"].shape == (3, 3)
    assert m["matriz"][1][2] == 1  # positivo real, "falha" prevista


def test_comparar_nao_apresenta_linha_incompleta(capsys):
    amostra = pd.DataFrame({"rotulo": ["negativo", "positivo"]})
    registros = {"zero-shot": [_reg("negativo"), None]}  # uma resposta faltando
    tabela = comparar(amostra, ["negativo", "positivo"], registros)
    assert tabela.loc["LLM zero-shot", "faltando"] == 1
    assert pd.isna(tabela.loc["LLM zero-shot", "acuracia"])
    assert "AVISO" in capsys.readouterr().out
