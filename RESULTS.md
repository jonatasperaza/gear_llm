# Resultados Atuais

Snapshot dos resultados experimentais atuais do GEAR-LLM.

Data do snapshot: 2026-07-04.

Modelos usados:

- barato: `HuggingFaceTB/SmolLM2-135M-Instruct`
- caro: `HuggingFaceTB/SmolLM2-360M-Instruct`

Aviso: estes resultados são iniciais, dependem dos prompts de benchmark e ainda não representam aceleração real do modelo. A economia é uma estimativa de custo teórico.

## Teacher Calibration

A calibração offline indicou que a configuração calibrada abaixo é mais conservadora que a política antiga:

```text
entropy_threshold = 0.35
margin_threshold  = 0.20
```

Resultado agregado observado:

```text
precision_accept = 97.90%
false_accept     = 2.10%
saved_percent    = 36.50%
```

Interpretação: quando o modelo barato é aceito sob esses thresholds, ele tende a concordar muito bem com o modelo caro nos contextos calibrados.

## Guarded v3 vs v2

O `adaptive_guarded_v2` mostrou um problema importante: em alguns prompts, especialmente `easy`, o guard de repetição chamava o modelo caro demais e podia gerar economia negativa.

O `adaptive_guarded_v3` adicionou:

- separação entre fallbacks obrigatórios e opcionais;
- budget cap para fallbacks opcionais;
- repetition guard condicionado à incerteza;
- cooldown para repetition guard.

Comparação principal em `results/quality_benchmark.csv`:

| prompt | v2 saved | v2 calls | v3 saved | v3 calls | observação |
| --- | ---: | ---: | ---: | ---: | --- |
| easy | -2.50% | 54 | 53.75% | 9 | v3 removeu o caso de economia negativa |
| math | 26.25% | 31 | 46.25% | 15 | v3 reduziu chamadas caras |
| logic_negation | 3.75% | 49 | 42.50% | 18 | v3 preservou parte da melhora com muito menos custo |
| code | 50.00% | 12 | 50.00% | 12 | custo equivalente |
| long_simple | 18.75% | 37 | 21.25% | 35 | leve melhora de custo |

## Speculative Benchmark

Configuração speculative atual:

```text
initial_draft_length = 6
verify_top_k         = 3
min_draft_length     = 2
max_draft_length     = 8
```

Resumo de `results/speculative_benchmark.csv`:

| prompt | mode | saved | expensive calls | acceptance | similarity |
| --- | --- | ---: | ---: | ---: | ---: |
| easy | speculative_adaptive | 45.19% | 12 | 96.25% | 0.1050 |
| math | speculative_adaptive | 33.25% | 17 | 90.00% | 0.5940 |
| logic_negation | speculative_adaptive | 14.31% | 22 | 81.25% | 0.0550 |
| code | speculative_adaptive | 49.06% | 11 | 97.50% | 0.5820 |
| long_simple | speculative_adaptive | 11.69% | 22 | 76.25% | 0.0360 |

Leitura rápida:

- `math`: speculative melhorou similaridade contra `adaptive_calibrated` neste snapshot.
- `code`: speculative manteve economia alta.
- `logic_negation` e `long_simple`: speculative foi pior em similaridade, motivando o hybrid router.

## Hybrid Benchmark

O hybrid router escolhe:

```text
math          -> speculative_adaptive
logic         -> adaptive_guarded_v3
code/general  -> adaptive_calibrated
```

Resumo de `results/hybrid_benchmark.csv`:

| prompt | prompt_type | selected_mode | saved | expensive calls | similarity |
| --- | --- | --- | ---: | ---: | ---: |
| easy | general | adaptive_calibrated | 57.50% | 6 | 0.0856 |
| math | math | speculative_adaptive | 33.25% | 17 | 0.5940 |
| logic_negation | logic | adaptive_guarded_v3 | 42.50% | 18 | 0.4557 |
| code | code | adaptive_calibrated | 50.00% | 12 | 0.6597 |
| long_simple | general | adaptive_calibrated | 21.25% | 35 | 0.1190 |

Critério atingido:

- evita o pior caso do speculative em `logic_negation`;
- evita speculative em `long_simple`;
- mantém o ganho do speculative no prompt `math`;
- mantém políticas simples e interpretáveis.

## Próximos Passos

- ampliar o conjunto de prompts;
- separar prompts por idioma e domínio;
- medir latência real com KV cache;
- comparar com speculative decoding mais fiel ao algoritmo clássico;
- usar métricas de qualidade mais fortes que similaridade textual superficial;
- investigar thresholds específicos por tipo de prompt.
