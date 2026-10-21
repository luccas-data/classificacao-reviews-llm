"""Testes do módulo llm com cliente falso: nenhum toca a rede."""
from types import SimpleNamespace

import anthropic
import httpx2
import pydantic
import pandas as pd
import pytest

from reviews import llm


class ClienteFalso:
    """Imita client.messages.parse; devolve (ou levanta) o que for programado."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0
        self.messages = self  # permite cliente.messages.parse(...)

    def parse(self, **kwargs):
        self.chamadas += 1
        r = self.respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def resposta(sentimento="negativo", motivo="entrega", stop_reason="end_turn", parsed=True):
    saida = llm.Classificacao(sentimento=sentimento, motivo=motivo) if parsed else None
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=saida,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=10))


def erro_rate_limit():
    req = httpx2.Request("POST", "http://teste")
    return anthropic.RateLimitError("limite", response=httpx2.Response(429, request=req), body=None)


def test_resposta_valida():
    c = ClienteFalso([resposta()])
    reg = llm.classificar_review(c, "r1", "chegou atrasado", "zero-shot")
    assert reg["ok"] and reg["sentimento"] == "negativo" and reg["motivo"] == "entrega"
    assert reg["tokens_entrada"] == 100 and reg["tokens_saida"] == 10


def test_positivo_nao_guarda_motivo():
    c = ClienteFalso([resposta("positivo", "outro")])
    assert llm.classificar_review(c, "r1", "ótimo", "zero-shot")["motivo"] is None


def test_recusa_e_max_tokens_viram_falha():
    for motivo in ("refusal", "max_tokens"):
        c = ClienteFalso([resposta(stop_reason=motivo)])
        reg = llm.classificar_review(c, "r1", "texto", "zero-shot")
        assert not reg["ok"] and reg["erro"] == motivo


def test_saida_vazia_vira_falha():
    c = ClienteFalso([resposta(parsed=False)])
    assert llm.classificar_review(c, "r1", "texto", "zero-shot")["erro"] == "saida_invalida"


def test_erro_da_api_vira_falha_sem_cache(tmp_path):
    caminho = tmp_path / "cache.jsonl"
    amostra = pd.DataFrame({"review_id": ["r1"], "texto": ["x"]})
    c = ClienteFalso([erro_rate_limit()])
    res = llm.classificar_amostra(c, amostra, "zero-shot", caminho)
    assert not res[0]["ok"] and res[0]["erro"] == "RateLimitError"
    assert llm.ler_cache(caminho) == {}  # erro da API não vai para o cache


def test_erro_de_conexao_vira_falha():
    req = httpx2.Request("POST", "http://teste")
    c = ClienteFalso([anthropic.APIConnectionError(request=req)])
    assert llm.classificar_review(c, "r1", "texto", "zero-shot")["erro"] == "APIConnectionError"


def test_saida_nao_json_vira_falha_e_vai_para_o_cache(tmp_path):
    caminho = tmp_path / "cache.jsonl"
    amostra = pd.DataFrame({"review_id": ["r1"], "texto": ["x"]})
    try:
        llm.Classificacao.model_validate_json('{"sentimento": "posi')  # JSON truncado
    except pydantic.ValidationError as e:
        invalido = e
    c = ClienteFalso([invalido])
    res = llm.classificar_amostra(c, amostra, "zero-shot", caminho)
    assert res[0]["erro"] == "saida_invalida"
    segundo = ClienteFalso([])
    llm.classificar_amostra(segundo, amostra, "zero-shot", caminho)
    assert segundo.chamadas == 0  # não paga de novo


def test_chave_invalida_interrompe():
    req = httpx2.Request("POST", "http://teste")
    erro = anthropic.AuthenticationError("chave", response=httpx2.Response(401, request=req), body=None)
    c = ClienteFalso([erro])
    with pytest.raises(anthropic.AuthenticationError):
        llm.classificar_review(c, "r1", "texto", "zero-shot")


def test_chave_do_cache_inclui_modelo():
    assert llm.chave_cache("r1", "zero-shot", "v1", "modelo-a") != llm.chave_cache("r1", "zero-shot", "v1", "modelo-b")


def test_exemplos_few_shot_3_negativos_3_positivos():
    sentimentos = [e[2] for e in llm.EXEMPLOS]
    assert sentimentos.count("negativo") == 3 and sentimentos.count("positivo") == 3
    assert {e[3] for e in llm.EXEMPLOS if e[2] == "negativo"} == {"entrega", "produto", "atendimento"}


def test_cache_nao_chama_cliente_quando_chave_existe(tmp_path):
    caminho = tmp_path / "cache.jsonl"
    amostra = pd.DataFrame({"review_id": ["r1"], "texto": ["chegou atrasado"]})

    primeiro = ClienteFalso([resposta()])
    llm.classificar_amostra(primeiro, amostra, "zero-shot", caminho)
    assert primeiro.chamadas == 1

    segundo = ClienteFalso([])  # se chamar, dá IndexError
    resultado = llm.classificar_amostra(segundo, amostra, "zero-shot", caminho)
    assert segundo.chamadas == 0
    assert resultado[0]["sentimento"] == "negativo"

    # outro modo = outra chave: precisa chamar de novo
    terceiro = ClienteFalso([resposta()])
    llm.classificar_amostra(terceiro, amostra, "few-shot", caminho)
    assert terceiro.chamadas == 1


def test_prompt_few_shot_tem_exemplos_e_zero_shot_nao():
    assert len(llm.EXEMPLOS) in range(4, 7)
    assert "Exemplos" in llm.montar_prompt("oi", "few-shot")
    assert "Exemplos" not in llm.montar_prompt("oi", "zero-shot")


def test_estimativa_few_shot_custa_mais():
    zero = llm.estimar_custo(["texto curto"] * 10, "zero-shot")["custo"]
    few = llm.estimar_custo(["texto curto"] * 10, "few-shot")["custo"]
    assert few > zero > 0
