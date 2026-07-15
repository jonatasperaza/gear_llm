# GEAR-LLM

GEAR-LLM significa **Geometric-Entropy Adaptive Routing for LLM Inference**.

Este projeto é um MVP experimental em Python, PyTorch e Hugging Face para
estudar quando uma requisição pode usar um modelo barato e quando precisa de um
modelo caro. A pesquisa começou em nível de token e agora também avalia
roteamento em nível de prompt.

## Hipótese

A hipótese central é simples: alguns tokens são fáceis, previsíveis ou redundantes; outros carregam estrutura, surpresa, negação, matemática, código ou mudança semântica. O GEAR-LLM mede esses sinais e estima um score `rho` para separar tokens em rotas de custo:

- `cheap`
- `medium`
- `expensive`

O projeto ainda não modifica os internals do modelo principal. Ele mede,
simula, valida e compara políticas de roteamento. Os experimentos mais recentes
mostram que roteamento token-level pode somar o custo dos dois modelos, enquanto
prompt-level routing escolhe apenas um modelo antes da geração.

## Modos Implementados

- **Token analysis / rho**: calcula entropia, surprisal, novidade geométrica, curvatura semântica e importância estrutural por token.
- **Ablation**: remove ou substitui tokens por rota para medir impacto na loss.
- **Balanced ablation**: compara grupos com o mesmo número de tokens para evitar viés por tamanho da classe.
- **Compute simulation**: estima economia teórica usando custos diferentes por rota.
- **adaptive_calibrated**: geração online usando um modelo barato e chamando o modelo caro quando entropia/margem indicam incerteza.
- **adaptive_guarded_v3**: versão adaptativa com quality guards, budget cap e controle de repetição.
- **adaptive_code_quality**: perfil token-level mais conservador para código.
- **speculative_adaptive**: speculative decoding em blocos com modelo barato gerando drafts e modelo caro verificando.
- **hybrid_router**: escolhe entre modos adaptativos por tipo de prompt; código usa `adaptive_code_quality` na política atual.
- **prompt_router_v1/v2**: escolhem `cheap_only` ou `expensive_only` antes da geração usando heurísticas manuais.
- **prompt_router_ml_v1**: roteador prompt-level com TF-IDF e Logistic Regression treinado a partir de labels de oracle.
- **prompt_router_ml_v2**: roteador aprendido sobre split MBPP fixo. O protocolo completo de 427 tarefas selecionou em validação um classifier TF-IDF sem probing; no teste held-out, ele passou 43/85 tarefas contra 42/85 do `expensive_only`, roteando 35 prompts para o modelo barato.
- **Task evaluation**: executa testes de código, verifica respostas matemáticas e classifica respostas lógicas.
- **Runtime profiling**: separa tempo de cheap forward, expensive forward, decisão, guards, tokenização e avaliação.
- **Latency benchmark**: mede tempo real de execução, tokens por segundo e memória de pico para comparar economia teórica com wall-clock time.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para os testes CUDA locais deste repositório, use a venv dedicada:

```powershell
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
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
python benchmark.py --task-evaluation
```

Avaliar o prompt router ML com o modelo arquivado do Kaggle:

```powershell
python run_task_evaluation.py `
  --dataset data/external_eval_tasks_90.jsonl `
  --categories code `
  --modes prompt_router_ml_v1 `
  --prompt-router-model results/kaggle/prompt_router_ml_v1/seed123_train/model.joblib
```

Preparar e executar o protocolo fixo do `prompt_router_ml_v2`:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/build_mbpp_split.py

.\.venv-cuda\Scripts\python.exe scripts/build_router_dataset_v2.py `
  --cheap-model Qwen/Qwen2.5-Coder-0.5B-Instruct `
  --expensive-model Qwen/Qwen2.5-Coder-3B-Instruct `
  --device cuda --torch-dtype float16 `
  --max-new-tokens 256 `
  --output-dir results/router_dataset_v2

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss classifier `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/classifier_probing

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss classifier `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/classifier_tfidf `
  --no-probing

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss l2d `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/l2d_probing

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss l2d `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/l2d_tfidf `
  --no-probing

.\.venv-cuda\Scripts\python.exe scripts/select_router_v2_policy.py `
  --validation-csv results/router_dataset_v2/val_features.csv `
  --candidates-root results/router_v2 `
  --output-dir results/router_v2/frozen_validation_policy

.\.venv-cuda\Scripts\python.exe scripts/eval_router_v2.py `
  --test-csv results/router_dataset_v2/test_features.csv `
  --model results/router_v2/frozen_validation_policy/model.joblib `
  --policy-meta results/router_v2/frozen_validation_policy/policy_meta.json `
  --output-dir results/router_v2/frozen_validation_policy
```

O builder é retomável por tarefa. O protocolo completo foi executado com 257
tarefas de treino, 85 de validação e 85 de teste, sem sobreposição. O teste
oficial já foi consumido uma vez; `eval_router_v2.py` agora recusa sobrescrever
esse relatório sem `--overwrite`. As features de concordância fazem um prefill
com ambos os modelos, mas a política selecionada não usa probing e escolhe um
único modelo a partir do prompt.

Os artefatos canônicos anteriores do Kaggle estão documentados em
[results/kaggle/README.md](results/kaggle/README.md). O protocolo fixo mais
recente está em `results/router_dataset_v2/` e
`results/router_v2/frozen_validation_policy/`; esses diretórios não devem ser
sobrescritos por smoke tests.

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

## Article

I wrote a preliminary article about the first GEAR-LLM results:

- [I Built a Small Experimental LLM Router — and Found When It Actually Gets Faster](https://medium.com/@jonatassilvaperaza/i-built-a-small-experimental-llm-router-and-found-when-it-actually-gets-faster-f5ed4c46230f)


## Relatório preliminar

Este projeto consiste em uma pesquisa experimental sobre o roteamento de LLMs (de baixo ou alto custo) com foco na latência.

- [Relatório Técnico](docs/TECHNICAL_REPORT.md)
- [Rascunho para o Medium](docs/MEDIUM_DRAFT.md)

## Aviso Experimental

GEAR-LLM é um projeto de pesquisa e validação. Os números atuais são dependentes
dos prompts, modelos, thresholds e hardware usados. O projeto demonstrou
speedup real em configurações específicas, mas ainda não demonstrou preservação
robusta de qualidade nem generalização confiável do roteador para prompts
inéditos.
