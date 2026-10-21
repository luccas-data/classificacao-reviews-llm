import pandas as pd

from reviews.preparo import dividir, limpar_texto, nota_para_rotulo, preparar


def _df_exemplo(n=100):
    """DataFrame fictício no formato do CSV do Olist."""
    notas = [1, 2, 3, 4, 5] * (n // 5)
    return pd.DataFrame({
        "review_id": [f"r{i}" for i in range(len(notas))],
        "review_score": notas,
        "review_comment_message": [f"Texto  número {i}\n" for i in range(len(notas))],
    })


def test_limpar_texto():
    assert limpar_texto("  Ótimo\r\nProduto   BOM \n") == "ótimo produto bom"


def test_rotulos():
    assert nota_para_rotulo(1) == "negativo"
    assert nota_para_rotulo(2) == "negativo"
    assert nota_para_rotulo(3) is None
    assert nota_para_rotulo(4) == "positivo"
    assert nota_para_rotulo(5) == "positivo"


def test_preparar_descarta_nota_3_e_sem_texto():
    df = _df_exemplo()
    df.loc[0, "review_comment_message"] = None  # nota 1 sem texto
    df.loc[1, "review_comment_message"] = "   "  # só espaços
    prep = preparar(df)
    assert 3 not in set(prep["nota"])
    assert set(prep["rotulo"]) == {"negativo", "positivo"}
    assert "r0" not in set(prep["review_id"]) and "r1" not in set(prep["review_id"])
    assert not prep["texto"].str.contains("\n").any()


def test_split_estratificado_e_reprodutivel():
    prep = preparar(_df_exemplo(500))
    treino, teste = dividir(prep)
    assert len(teste) == round(len(prep) * 0.2)
    # a proporção de cada classe é praticamente a mesma nos dois lados
    assert abs((treino["rotulo"] == "negativo").mean() - (teste["rotulo"] == "negativo").mean()) < 0.02
    assert set(treino["review_id"]).isdisjoint(set(teste["review_id"]))
    _, teste2 = dividir(prep)
    assert list(teste["review_id"]) == list(teste2["review_id"])
