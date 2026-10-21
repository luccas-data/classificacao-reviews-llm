"""Modelo clássico de referência: TF-IDF + Regressão Logística."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def treinar_baseline(textos, rotulos) -> Pipeline:
    """Treina o pipeline TF-IDF (unigramas + bigramas) + regressão logística."""
    modelo = Pipeline([
        # min_df=5: ignora termos que aparecem em menos de 5 avaliações
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5)),
        # class_weight="balanced" compensa o desbalanceamento (mais positivos que negativos)
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    modelo.fit(textos, rotulos)
    return modelo


def palavras_mais_fortes(modelo: Pipeline, n: int = 15) -> dict:
    """Termos com maiores coeficientes de cada classe."""
    termos = np.array(modelo.named_steps["tfidf"].get_feature_names_out())
    coef = modelo.named_steps["lr"].coef_[0]
    # em classes_ = [negativo, positivo], o coeficiente positivo aponta para "positivo"
    classes = modelo.named_steps["lr"].classes_
    ordem = np.argsort(coef)
    return {
        classes[1]: list(termos[ordem[::-1][:n]]),
        classes[0]: list(termos[ordem[:n]]),
    }
