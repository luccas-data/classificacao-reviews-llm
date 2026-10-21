"""Classificação de avaliações com Claude (zero-shot e few-shot), com cache em disco.

Uso (só estimativa de custo):
    python -m reviews.llm --amostra 500 --modo zero-shot
Uso (chama a API de verdade, gasta dinheiro):
    python -m reviews.llm --amostra 500 --modo zero-shot --confirmar
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional, get_args

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from reviews.preparo import RAIZ, amostrar_teste, carregar_dados

MODELO = "claude-haiku-5-5"
MAX_TOKENS = 1024  # o Haiku 5.5 pensa antes de responder; 200 cortava ~4% das respostas
VERSAO_PROMPT = "v1"  # mude ao alterar o prompt, para não reaproveitar cache antigo
ARQUIVO_CACHE = RAIZ / "data" / "cache_llm.jsonl"

# preços em US$ por token (US$ 0,10 / 1M de entrada e US$ 0,50 / 1M de saída)
PRECO_ENTRADA = 0.10 / 1_000_000
PRECO_SAIDA = 0.50 / 1_000_000

Motivo = Literal["entrega", "produto", "atendimento", "cobrança", "outro"]
MOTIVOS = list(get_args(Motivo))

# margem para os tokens que a saída estruturada acrescenta ao prompt (valor aproximado)
MARGEM_ENTRADA_ESTRUTURADA = 450  # medido na execução real: o esquema da saída estruturada pesa


class Classificacao(BaseModel):
    """Formato da resposta que o modelo precisa devolver."""

    sentimento: Literal["positivo", "negativo"]
    # só faz sentido para avaliações negativas
    motivo: Optional[Motivo] = None


INSTRUCAO = (
    "Você classifica avaliações de clientes de uma loja online brasileira.\n"
    "Diga se o sentimento geral da avaliação é 'positivo' ou 'negativo'.\n"
    "Se for negativo, informe também o motivo principal: "
    "'entrega' (atraso, produto não chegou), 'produto' (defeito, qualidade, produto errado), "
    "'atendimento' (vendedor ou suporte), 'cobrança' (pagamento, estorno, reembolso) ou 'outro'.\n"
    "Se for positivo, deixe o motivo vazio."
)

# Exemplos fixos tirados do conjunto de TREINO (nunca do teste). O motivo foi rotulado à mão.
EXEMPLOS = [
    ("56ae08808a1fc0462281893a7c331341",
     "o produto tem forro em eva quando tirei a etiqueta o tapete rasgou! qualidade péssimo! "
     "em relação a aparência, é bonito! mas isso não basta!", "negativo", "produto"),
    ("b91ec4b047be78b2822cf8aaf61bf024",
     "provavelmente evitarei comprar novamente, pois a demora na entrega é muito grande.",
     "negativo", "entrega"),
    ("cfaa17f2822cfbd9d6bab7a15a1e978b",
     "não recebi o produto e enviei duas mesagens para o vendedor e nenhuma foi respondida, "
     "descaso total.", "negativo", "atendimento"),
    ("fc18ccf1e624cbbd0e39b800e6981a70",
     "produto muito bom conforme o que eu esperava e a entrega super mega rápida.",
     "positivo", None),
    ("f925d5e5f1fd667ca0ec5b851f9adc5d",
     "recebi o produto antes do prazo, era o que esperava ficou muito lindo na minha mesa de jantar.",
     "positivo", None),
    ("2fec8c3eafd7b90589781376afab7334",
     "super recomendo...foi entregue antes do dia previsto...", "positivo", None),
]


def montar_prompt(texto: str, modo: str) -> str:
    """Monta o texto enviado ao modelo. No few-shot inclui os exemplos fixos."""
    partes = [INSTRUCAO]
    if modo == "few-shot":
        partes.append("\nExemplos:")
        for _, ex_texto, sentimento, motivo in EXEMPLOS:
            resposta = f"sentimento={sentimento}, motivo={motivo or 'vazio'}"
            partes.append(f'Avaliação: "{ex_texto}"\nResposta: {resposta}')
    partes.append(f'\nAvaliação a classificar:\n"{texto}"')
    return "\n".join(partes)


# ---------------------------------------------------------------- cache

def chave_cache(review_id: str, modo: str, versao: str = VERSAO_PROMPT, modelo: str = MODELO) -> str:
    return f"{review_id}|{modo}|{versao}|{modelo}"


def ler_cache(caminho: Path = ARQUIVO_CACHE) -> dict:
    """Lê o jsonl e devolve {chave: registro}. Arquivo inexistente = cache vazio."""
    cache = {}
    if not Path(caminho).exists():
        return cache
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                reg = json.loads(linha)
                cache[chave_cache(reg["review_id"], reg["modo"], reg["versao_prompt"], reg["modelo"])] = reg
    return cache


def gravar_cache(registro: dict, caminho: Path = ARQUIVO_CACHE) -> None:
    """Acrescenta um registro ao fim do arquivo (uma linha por resposta)."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- chamada à API

def _registro(review_id, modo, **campos) -> dict:
    """Registro-base; as respostas e as falhas usam os mesmos campos."""
    base = {
        "review_id": review_id, "modo": modo, "versao_prompt": VERSAO_PROMPT, "modelo": MODELO,
        "ok": False, "sentimento": None, "motivo": None, "erro": None,
        "tokens_entrada": 0, "tokens_saida": 0, "segundos": 0.0,
    }
    base.update(campos)
    return base


def classificar_review(cliente, review_id: str, texto: str, modo: str) -> dict:
    """Classifica uma avaliação e devolve um registro (ok=False se falhou).

    As tentativas automáticas (rate limit, erro de rede, 5xx) ficam por conta do SDK
    (max_retries no cliente). Chave inválida ou parâmetro inválido interrompem tudo.
    """
    prompt = montar_prompt(texto, modo)
    inicio = time.time()
    try:
        resposta = cliente.messages.parse(
            model=MODELO,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_format=Classificacao,
        )
    except ValidationError:
        # saída truncada ou fora do formato: a API cobrou, então vai para o cache como falha
        # (a exceção não traz o usage, por isso os tokens ficam 0)
        return _registro(review_id, modo, erro="saida_invalida", segundos=time.time() - inicio)
    except (anthropic.AuthenticationError, anthropic.BadRequestError):
        raise  # repetir 500 vezes não adianta; quem chamou mostra a mensagem e para
    except anthropic.APIError as e:
        # RateLimitError, APIStatusError, APIConnectionError... depois dos retries do SDK
        return _registro(review_id, modo, erro=type(e).__name__, segundos=time.time() - inicio)

    tokens = {"tokens_entrada": resposta.usage.input_tokens,
              "tokens_saida": resposta.usage.output_tokens,
              "segundos": time.time() - inicio}
    # antes de ler o conteúdo, confere por que o modelo parou
    if resposta.stop_reason in ("refusal", "max_tokens"):
        return _registro(review_id, modo, erro=resposta.stop_reason, **tokens)
    saida = resposta.parsed_output
    if saida is None:
        return _registro(review_id, modo, erro="saida_invalida", **tokens)
    motivo = saida.motivo if saida.sentimento == "negativo" else None
    return _registro(review_id, modo, ok=True, sentimento=saida.sentimento, motivo=motivo, **tokens)


def classificar_amostra(cliente, amostra, modo: str, caminho_cache: Path = ARQUIVO_CACHE) -> list[dict]:
    """Classifica cada linha da amostra (colunas review_id e texto), usando o cache."""
    cache = ler_cache(caminho_cache)
    resultados = []
    for _, linha in tqdm(amostra.iterrows(), total=len(amostra), desc=modo):
        chave = chave_cache(linha["review_id"], modo)
        if chave in cache:  # já respondido: não chama a API
            resultados.append(cache[chave])
            continue
        reg = classificar_review(cliente, linha["review_id"], linha["texto"], modo)
        # erros da API (rede, limite...) não vão para o cache: podem funcionar depois;
        # respostas cobradas (ok, recusa, max_tokens, saída inválida) vão, para não pagar de novo
        if reg["ok"] or reg["erro"] in ("refusal", "max_tokens", "saida_invalida"):
            gravar_cache(reg, caminho_cache)
            cache[chave] = reg
        resultados.append(reg)
    return resultados


# ---------------------------------------------------------------- custo

def custo_dolares(tokens_entrada: int, tokens_saida: int) -> float:
    return tokens_entrada * PRECO_ENTRADA + tokens_saida * PRECO_SAIDA


def estimar_custo(textos, modo: str) -> dict:
    """Estimativa aproximada: ~3 caracteres por token em português, ~45 tokens de saída
    e uma margem fixa de entrada por chamada para a saída estruturada."""
    tokens_entrada = sum(len(montar_prompt(t, modo)) / 3 + MARGEM_ENTRADA_ESTRUTURADA for t in textos)
    tokens_saida = 45 * len(textos)
    return {"tokens_entrada": int(tokens_entrada), "tokens_saida": int(tokens_saida),
            "custo": custo_dolares(tokens_entrada, tokens_saida)}


# ---------------------------------------------------------------- linha de comando

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classifica avaliações do teste com Claude.")
    parser.add_argument("--amostra", type=int, default=500, help="tamanho da amostra do teste")
    parser.add_argument("--modo", choices=["zero-shot", "few-shot"], required=True)
    parser.add_argument("--confirmar", action="store_true",
                        help="sem isto, só imprime a estimativa de custo e sai")
    args = parser.parse_args(argv)

    _, teste = carregar_dados()
    amostra = amostrar_teste(teste, args.amostra)

    cache = ler_cache()
    pendentes = amostra[[chave_cache(r, args.modo) not in cache for r in amostra["review_id"]]]
    est = estimar_custo(pendentes["texto"], args.modo)
    print(f"Modelo: {MODELO} | modo: {args.modo} | amostra: {len(amostra)} "
          f"({len(amostra) - len(pendentes)} já no cache)")
    print(f"Chamadas necessárias: {len(pendentes)}")
    print(f"Tokens estimados: {est['tokens_entrada']} de entrada, {est['tokens_saida']} de saída")
    print(f"Custo estimado (aproximado): US$ {est['custo']:.4f}")

    if not args.confirmar:
        print("Nada foi enviado. Rode de novo com --confirmar para chamar a API.")
        return 0

    load_dotenv(RAIZ / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY não encontrada. Copie .env.example para .env e preencha.")
        return 1

    cliente = anthropic.Anthropic(max_retries=5)  # o SDK repete sozinho erros temporários
    try:
        resultados = classificar_amostra(cliente, amostra, args.modo)
    except anthropic.AuthenticationError:
        print("Erro de autenticação: confira a ANTHROPIC_API_KEY no .env. Execução interrompida.")
        return 1
    except anthropic.BadRequestError as e:
        print(f"Requisição inválida (parâmetro errado?): {e}. Execução interrompida.")
        return 1
    falhas = sum(not r["ok"] for r in resultados)
    gasto = custo_dolares(sum(r["tokens_entrada"] for r in resultados),
                          sum(r["tokens_saida"] for r in resultados))
    print(f"Concluído: {len(resultados) - falhas} ok, {falhas} falhas. "
          f"Custo registrado: US$ {gasto:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
