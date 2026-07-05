# GEAR-LLM

GEAR-LLM significa **Geometric-Entropy Adaptive Routing for LLM Inference**.

Este projeto é um MVP experimental em Python, PyTorch e Hugging Face para estudar se todos os tokens de um prompt realmente precisam usar a mesma quantidade de computação durante inferência de LLMs.

## Hipótese

A hipótese central é simples: alguns tokens são fáceis, previsíveis ou redundantes; outros carregam estrutura, surpresa, negação, matemática, código ou mudança semântica. O GEAR-LLM mede esses sinais e estima um score `rho` para separar tokens em rotas de custo:

- `cheap`
- `medium`
- `expensive`

O projeto ainda não modifica os internals do modelo principal. Ele mede, simula, valida e compara políticas de roteamento.

## Modos Implementados

- **Token analysis / rho**: calcula entropia, surprisal, novidade geométrica, curvatura semântica e importância estrutural por token.
- **Ablation**: remove ou substitui tokens por rota para medir impacto na loss.
- **Balanced ablation**: compara grupos com o mesmo número de tokens para evitar viés por tamanho da classe.
- **Compute simulation**: estima economia teórica usando custos diferentes por rota.
- **adaptive_calibrated**: geração online usando um modelo barato e chamando o modelo caro quando entropia/margem indicam incerteza.
- **adaptive_guarded_v3**: versão adaptativa com quality guards, budget cap e controle de repetição.
- **speculative_adaptive**: speculative decoding em blocos com modelo barato gerando drafts e modelo caro verificando.
- **hybrid_router**: escolhe automaticamente entre `adaptive_calibrated`, `adaptive_guarded_v3` e `speculative_adaptive` com heurísticas simples por tipo de prompt.
- **Latency benchmark**: mede tempo real de execução, tokens por segundo e memória de pico para comparar economia teórica com wall-clock time.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Modelo barato padrão:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

Modelo caro padrão:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

## Comandos Principais

Analisar tokens e salvar CSV:

```powershell
python run_analyze.py --prompt "Explique por que a inversa de f(x)=5x+1 é (x-1)/5." --csv results/math.csv
```

Rodar benchmark base de análise:

```powershell
python benchmark.py
```

Rodar ablation:

```powershell
python run_ablation.py --prompt "Explique por que a inversa de f(x)=5x+1 é (x-1)/5."
```

Rodar ablation balanceada:

```powershell
python run_ablation.py --prompt "Explique por que a inversa de f(x)=5x+1 é (x-1)/5." --balanced
```

Rodar simulação de custo:

```powershell
python run_compute_sim.py --prompt "Explique em uma frase o que é água."
```

Rodar geração adaptativa:

```powershell
python run_adaptive_generate.py --prompt "Explique em uma frase o que é água."
```

Rodar speculative decoding:

```powershell
python run_speculative_generate.py --prompt "Explique por que a inversa de f(x)=5x+1 é (x-1)/5."
```

Rodar roteador híbrido:

```powershell
python run_hybrid_generate.py --prompt "Explique por que a inversa de f(x)=5x+1 é (x-1)/5."
```

Rodar benchmarks específicos:

```powershell
python benchmark.py --quality-benchmark
python benchmark.py --teacher-calibration
python benchmark.py --policy-replay
python benchmark.py --speculative-generate
python benchmark.py --speculative-tuning
python benchmark.py --hybrid-benchmark
python benchmark.py --latency-benchmark --max-new-tokens 32
```

## Exemplo de Uso

```powershell
.\.venv\Scripts\activate
python run_hybrid_generate.py --prompt "Se não chover e apenas se o vento parar, então podemos sair; exceto se houver alerta."
```

Saída esperada:

```text
prompt_type            : logic
selected_mode          : adaptive_guarded_v3
estimated_saved_percent: ...
expensive_model_calls  : ...
```

## Relatório preliminar

Este projeto consiste em uma pesquisa experimental sobre o roteamento de LLMs (de baixo ou alto custo) com foco na latência.

- [Relatório Técnico](docs/TECHNICAL_REPORT.md)
- [Rascunho para o Medium](docs/MEDIUM_DRAFT.md)

## Aviso Experimental

GEAR-LLM é um projeto de pesquisa e validação. Os números atuais são dependentes dos prompts, modelos, thresholds e hardware usados. A economia reportada é uma estimativa de custo teórico, não uma aceleração real garantida.
