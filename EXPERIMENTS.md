# Experimentos do GEAR-LLM

Este documento resume as fases experimentais implementadas no GEAR-LLM e o que cada uma tenta validar.

## 1. Token Analysis e `rho`

A primeira fase mede cada token do prompt e calcula um score `rho` para estimar sua criticidade.

O score combina:

- **Entropia**: incerteza da distribuição de próximo token.
- **Surprisal**: quão inesperado foi o token atual dado o contexto anterior.
- **Novidade geométrica**: distância do hidden state atual em relação ao histórico recente.
- **Curvatura semântica**: mudança de direção no espaço de hidden states.
- **Importância estrutural**: boost para tokens matemáticos, numéricos, lógicos, operadores, pontuação estrutural e delimitadores.

Forma geral:

```text
rho =
  entropy_weight    * entropy_norm
+ surprisal_weight  * surprisal_norm
+ novelty_weight    * novelty_norm
+ curvature_weight  * curvature_norm
+ structural_weight * structural_importance
```

O resultado por token é salvo em CSV com rota:

```text
cheap | medium | expensive
```

## 2. Balanced Ablation

A ablation simples substitui tokens de uma classe e mede o aumento de loss. A versão balanceada torna a comparação mais justa:

- define `k` como o número de tokens `expensive`;
- compara grupos com o mesmo tamanho;
- remove/substitui:
  - `top_k_expensive`
  - `bottom_k_cheap`
  - `top_k_medium`
  - `random_k`
- calcula `delta_loss` e `delta_loss_per_token`;
- roda múltiplos baselines aleatórios.

Critério desejado:

```text
expensive_delta_per_token > cheap_delta_per_token
expensive_delta_per_token > random_mean_delta_per_token
```

## 3. Teacher Calibration

A teacher calibration compara o modelo barato com o modelo caro no mesmo contexto.

Para cada passo de geração, mede:

- entropia normalizada do modelo barato;
- probabilidade top-1;
- probabilidade top-2;
- margem top-1 menos top-2;
- se o top-1 do barato bate com o top-1 do caro;
- se o top-1 do barato está no top-k do caro.

Depois faz grid search de thresholds:

```text
entropy_threshold: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55
margin_threshold : 0.10, 0.15, 0.20, 0.25, 0.30
```

Objetivo: encontrar thresholds que aceitem o modelo barato quando ele tende a concordar com o modelo caro.

## 4. Policy Replay

O replay de políticas usa os contextos já coletados pela teacher calibration.

Isso evita path-dependence da geração online: todas as políticas são comparadas nos mesmos passos e nos mesmos logits salvos.

Políticas atuais:

- `old_0.45_0.20`
- `calibrated_0.35_0.20`
- `strict_0.30_0.20`
- `loose_0.50_0.15`

Métricas:

- accept rate;
- exact precision accept;
- top-k precision accept;
- false accept rate;
- estimated saved percent.

## 5. Quality Benchmark

O benchmark de qualidade compara modos de geração contra `expensive_only`.

Modos principais:

- `cheap_only`
- `expensive_only`
- `adaptive_calibrated`
- `adaptive_guarded`
- `adaptive_guarded_v2`
- `adaptive_guarded_v3`

Métricas:

- economia estimada;
- chamadas ao modelo caro;
- similaridade com `expensive_only` via `difflib.SequenceMatcher`;
- Jaccard de palavras normalizadas;
- taxa de repetição de 3-gramas;
- taxa de repetição de 4-gramas.

## 6. Speculative Tuning

O speculative decoding usa o modelo barato para gerar um bloco de tokens de rascunho e o modelo caro para verificar esse bloco em uma única passagem.

A fase de tuning testa combinações de:

- `initial_draft_length`
- `verify_top_k`
- `min_draft_length`
- `max_draft_length`

Score usado:

```text
score =
    similarity_to_expensive
  + 0.25 * jaccard_to_expensive
  - 0.50 * repeated_3gram_rate
  + 0.20 * max(0, estimated_saved_percent / 100)
  - 0.75 * max(0, -estimated_saved_percent / 100)
```

Configuração atual escolhida:

```text
initial_draft_length = 6
verify_top_k = 3
min_draft_length = 2
max_draft_length = 8
```

## 7. Hybrid Benchmark

O hybrid router escolhe automaticamente um modo de geração com base no tipo de prompt.

Classificações:

- `logic`
- `math`
- `code`
- `long_simple`
- `general`

Política atual:

```text
logic       -> adaptive_guarded_v3
math        -> speculative_adaptive
code        -> adaptive_calibrated
long_simple -> adaptive_calibrated
general     -> adaptive_calibrated
```

O benchmark compara:

- `adaptive_calibrated`
- `adaptive_guarded_v3`
- `speculative_adaptive`
- `hybrid`

Objetivo: evitar casos ruins do speculative em lógica e prompts longos simples, mantendo ganhos em matemática.
