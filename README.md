# Classificação de avaliações: LLM sem treino vs. modelo clássico

**Pergunta:** um LLM sem treino (zero-shot e few-shot) classifica avaliações de clientes melhor
que um modelo clássico de machine learning (TF-IDF + Regressão Logística)?

Projeto de estudo feito por Luccas Nunes ([github.com/luccas-data](https://github.com/luccas-data)).

## Dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
arquivo `olist_order_reviews_dataset.csv` (99.224 avaliações). Licença
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): uso não comercial, com
atribuição e mesma licença. **Os CSV não estão neste repositório**; o script
`scripts/baixar_dados.py` baixa o arquivo de um espelho público no Hugging Face
(`bulutttt/olist-raw-data`) e confere o sha256.

## Método

1. **Preparo** (`src/reviews/preparo.py`): só avaliações com comentário (41% do total); limpeza leve
   (minúsculas, sem quebras de linha); rótulo binário: notas 1-2 = negativo, 4-5 = positivo.
   A **nota 3 fica de fora** (ambígua). Sobram 37.113 avaliações (71% positivas, 29% negativas).
   Split treino/teste 80/20 estratificado, `random_state=42`; os ids do teste ficam em
   `data/ids_teste.csv`.
2. **Baseline** (`baseline.py`): `TfidfVectorizer` (unigramas + bigramas, `min_df=5`) +
   `LogisticRegression(class_weight="balanced")`.
3. **LLM** (`llm.py`): modelo `claude-haiku-5-5` via SDK oficial `anthropic`, com saída
   estruturada (Pydantic): sentimento e, nos negativos, motivo (entrega, produto, atendimento,
   cobrança, outro). Dois modos: zero-shot e few-shot (6 exemplos fixos do treino: 3 negativos e 3 positivos). Cada resposta
   vai para `data/cache_llm.jsonl` (chave: review_id + modo + versão do prompt), então nada é
   cobrado duas vezes.
4. **Comparação** (`avaliacao.py`): os três métodos **na mesma amostra de 500 avaliações do teste**
   (estratificada, seed 42): acurácia, F1 macro, matriz de confusão, custo e tempo.

Notebooks: `01_exploracao` (dados), `02_baseline` (modelo clássico), `03_llm_vs_baseline`
(comparação; funciona sem cache e avisa como gerar as respostas).

## Resultados

### Baseline (teste completo, 7.423 avaliações)

| Métrica | Valor |
|---|---|
| Acurácia | 0,925 |
| F1 macro | 0,912 |

Matriz de confusão (linhas = real, colunas = previsto):

| | previsto negativo | previsto positivo |
|---|---|---|
| **real negativo** | 2012 | 140 |
| **real positivo** | 419 | 4852 |

Na amostra de 500 usada na comparação, o baseline fez acurácia 0,930 e F1 macro 0,918.

### Baseline × LLM (amostra de 500 do teste)

| Método | Acurácia | F1 macro | Custo (US$) | Tempo |
|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0,930 | 0,918 | 0 | 0,6 s de treino; prever 500 leva 0,01 s |
| LLM zero-shot | 0,938 | 0,928 | 0,042 | ~11 min (500 chamadas em sequência) |
| LLM few-shot | 0,944 | 0,935 | 0,062 | ~11 min |

Nenhuma resposta do LLM falhou nos dois modos.

**A diferença não é significativa.** Como os três métodos classificam as mesmas 500 avaliações, dá
para comparar caso a caso com o teste de McNemar, que olha só as avaliações em que um acerta e o
outro erra:

| Comparação | só o 1º acertou | só o 2º acertou | p-valor |
|---|---|---|---|
| Baseline × zero-shot | 11 | 15 | 0,56 |
| Baseline × few-shot | 9 | 16 | 0,23 |
| Zero-shot × few-shot | 2 | 5 | 0,45 |

Os três erram nos mesmos lugares: 18 avaliações foram erradas pelos três. Lendo esses 18 casos,
quase todos têm texto e nota em direções opostas: 15 são reclamações com nota 4 ou 5 ("o pedido
foi feito na cor pérola e veio branca") e 3 são elogios com nota 1 ou 2 ("produto bom minha filha
amou"). O erro está no rótulo, não nos modelos. Pelo mesmo motivo, o erro mais comum dos três é
chamar de negativa uma avaliação de nota alta que reclama de algo (25 a 27 casos cada).

### Motivos das avaliações negativas (segundo o LLM)

| Motivo | zero-shot | few-shot |
|---|---|---|
| entrega | 105 | 99 |
| produto | 50 | 52 |
| outro | 11 | 8 |
| cobrança | 1 | 5 |
| atendimento | 1 | 3 |

Cerca de dois terços das reclamações são sobre entrega. Esse é o tipo de resposta que o modelo
clássico não dá sem um conjunto de dados rotulado por motivo.

### Conclusão

Para **sentimento**, o modelo clássico basta: empata com o LLM dentro da margem de erro, é
gratuito, roda em menos de um segundo e não depende de API externa. O LLM se paga em outro lugar:
classificar o **motivo** das reclamações sem nenhum exemplo rotulado, com um custo de cerca de
US$ 8 a cada 100 mil avaliações no modo zero-shot. Na prática, um caminho razoável é usar o
modelo clássico para o sentimento e mandar ao LLM só as avaliações negativas.

## Custo

Preço do `claude-haiku-5-5`: US$ 0,10 por 1M de tokens de entrada e US$ 0,50 por 1M de saída.
Custo real das 500 avaliações, calculado com os tokens devolvidos pela API (`response.usage`)
e gravados no cache:

| Modo | Tokens de entrada por chamada | Tokens de saída por chamada | Custo |
|---|---|---|---|
| zero-shot | 620 | 42 | US$ 0,042 |
| few-shot | 1.026 (o prompt leva os exemplos) | 43 | US$ 0,062 |

Antes de chamar a API, o script imprime uma estimativa. A primeira versão errava por quase o
dobro, porque o esquema da saída estruturada acrescenta cerca de 450 tokens a cada chamada; a
margem foi recalibrada com os números reais.

**Um problema encontrado na execução:** com `max_tokens=200`, cerca de 4% das respostas vinham
cortadas. O Haiku 5.5 "pensa" antes de responder, e às vezes esse raciocínio gastava o limite.
Com 1024 as falhas sumiram, e o custo não muda, porque só se paga o que é usado.

## Limitações

- O LLM é avaliado em **500 avaliações** (não nas 7.423 do teste) para limitar o custo. Com essa
  amostra, diferenças de 1 a 2 pontos não são conclusivas (ver o teste de McNemar acima).
- O **rótulo vem da nota**, não de leitura humana do texto. Há notas que contradizem o comentário
  (ex.: texto "excelente" com nota 1), o que limita a acurácia máxima de qualquer método.
- A **nota 3 está fora** da comparação principal; os resultados não dizem nada sobre avaliações neutras.
- Os motivos das avaliações negativas não têm rótulo de referência: só dá para contar o que o LLM
  respondeu, não medir acerto.
- Só avaliações com texto, em português, de uma única loja/marketplace.

## Como reproduzir

```bash
uv sync                                  # instala as dependências
uv run python scripts/baixar_dados.py    # baixa e valida os dados
cp .env.example .env                     # depois edite o .env e coloque ANTHROPIC_API_KEY
uv run pytest -q                         # testes (não usam a rede)

# estimativa de custo (não chama a API)
uv run python -m reviews.llm --amostra 500 --modo zero-shot
# chamadas reais (gastam dinheiro)
uv run python -m reviews.llm --amostra 500 --modo zero-shot --confirmar
uv run python -m reviews.llm --amostra 500 --modo few-shot --confirmar
```

Depois, execute os notebooks em `notebooks/` (o 03 lê `data/cache_llm.jsonl`). A chave fica
apenas no `.env`, que está no `.gitignore`.
