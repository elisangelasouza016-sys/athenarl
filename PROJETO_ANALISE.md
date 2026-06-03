# PROJETO_ANALISE.md — Athena-RL

> Documento de análise técnica do projeto **athenarl**, gerado a partir da leitura integral do código-fonte presente na branch `v.2` (commit base `1710aac`).
> Esta análise é **descritiva**: documenta apenas o que existe hoje no repositório, sem propor alterações de código.

---

## 1. Visão geral do sistema

O **Athena-RL** é uma aplicação web monolítica de página única, escrita em Python e construída sobre o framework [Streamlit](https://streamlit.io/). O sistema é um **demonstrador interativo (proof-of-concept)** de um agente de mediação que utiliza **Aprendizado por Reforço (Reinforcement Learning)** — mais especificamente o algoritmo **Q-Learning tabular** — para decidir qual ação de governança aplicar diante de uma mensagem textual potencialmente prejudicial em ambientes técnicos colaborativos (GitHub, fóruns, comunidades STEM).

A aplicação:

- recebe um texto digitado pela pessoa usuária;
- classifica esse texto em uma das 12 categorias de interação predefinidas (estado `s`);
- escolhe uma de 6 ações de mediação possíveis (`a`) através de uma política **ε-greedy**;
- recebe um sinal composto de recompensa (`R = R_humano + R_impacto`);
- atualiza a Q-Table conforme a fórmula clássica de Q-Learning;
- exibe o histórico, a evolução da matriz de aprendizado e uma explicação acadêmica.

Toda a execução acontece em memória (`st.session_state`), sem persistência em disco ou banco de dados.

## 2. Objetivo do projeto

Conforme declarado no `README.md`:

> *"Combater as microagressões de gênero, o silenciamento e o descrédito técnico que historicamente impulsionam a evasão feminina na área da computação. O sistema atua como um escudo ativo, processando interações em tempo real e decidindo a ação de governança que maximiza a saúde e a segurança do ecossistema a longo prazo."*

Em termos técnicos, os objetivos observáveis no código são:

1. **Demonstrar** uma arquitetura híbrida em duas camadas — **percepção (classificação)** + **decisão (RL)** — onde o RL não classifica o texto, apenas escolhe a melhor ação de mediação.
2. **Tornar tangível** o ciclo `estado → ação → recompensa → atualização` por meio de uma interface didática com 5 abas.
3. **Combinar** dois tipos de sinal na recompensa: feedback humano direto e métricas observáveis do ambiente (comentário editado, PR aceito, reincidência, abandono, etc.).
4. **Servir** como artefato acadêmico/pedagógico — a aba "Explicação" expõe a fórmula matemática e a tabela de elementos do RL.

## 3. Tecnologias utilizadas

| Categoria | Tecnologia | Onde aparece |
|---|---|---|
| Linguagem | Python 3 | `app.py` |
| Framework UI | Streamlit | `import streamlit as st` (linha 1) |
| Manipulação de dados | pandas | `pd.DataFrame` para a Q-Table e o histórico |
| Computação numérica | NumPy | `np.random` para inicialização e política ε-greedy |
| Data/hora | `datetime` (stdlib) | Carimbo de cada episódio no histórico |
| Estilo visual | HTML/CSS embutido | Bloco `<style>` injetado via `st.markdown(unsafe_allow_html=True)` (linhas 12–41) |
| Renderização matemática | LaTeX via `st.latex` | Fórmula de Q-Learning na aba "Explicação" |
| Controle de versão | Git + GitHub | Repositório `elisangelasouza016-sys/athenarl` |

**Ausências relevantes para um novo desenvolvedor:**

- Não há `requirements.txt`, `pyproject.toml`, `Pipfile`, `setup.py` nem `environment.yml`. As dependências precisam ser instaladas manualmente.
- Não há `.gitignore`, `Dockerfile`, `Makefile`, scripts de CI/CD, nem configuração de linter/formatter.
- Não há suíte de testes automatizados.
- Não há arquivos de configuração de ambiente (`.env`, `config.toml`, `secrets.toml`).
- Não há `LICENSE`.

## 4. Estrutura de diretórios

```
athenarl/
├── .git/                    # Repositório Git (metadados de versionamento)
├── README.md                # Apresentação textual curta (3 linhas) do propósito
└── app.py                   # Aplicação Streamlit completa (596 linhas)
```

O projeto é **single-file**: toda a lógica — modelo de domínio, lógica de aprendizado, classificação, interface, estilos — está concentrada em `app.py`.

## 5. Principais módulos e responsabilidades

Embora não existam módulos físicos separados, o arquivo `app.py` está organizado em **blocos lógicos** delimitados por cabeçalhos de comentário. Cada bloco pode ser entendido como um "módulo conceitual":

### 5.1 Configuração da página (linhas 6–10)
`st.set_page_config(...)` define título, ícone (🛡️) e layout *wide*.

### 5.2 Estilos visuais (linhas 12–41)
CSS embutido. Define paleta de cores (`#6B5B95` roxo principal, `#FF6F61` coral, fundos `#F4EFF7`, `#FFECEA`, `#EAF2F8`) e dois cards visuais (`warning-card`, `success-card`).

### 5.3 Constantes de domínio (linhas 47–73)

- **`CATEGORIES`** (12 estados `s`): categorias de interação, incluindo `Neutral (Saudável)` como estado positivo de referência. As 11 demais representam diferentes tipos de agressão ou descrédito.
- **`ACTIONS`** (6 ações `a`): do mais brando ("Silêncio Operacional") ao mais severo ("Reportar p/ Governança Institucional").
- **Hiperparâmetros default**: `ALPHA = 0.30`, `GAMMA = 0.80`, `EPSILON = 0.20`.

### 5.4 Funções de domínio (linhas 79–224)

| Função | Linhas | Responsabilidade |
|---|---|---|
| `inicializar_q_table()` | 79–96 | Cria DataFrame 12×6, valores aleatórios uniformes em [0.1, 0.5] com `seed=42`, depois "semeia" 6 células com valores mais altos representando ações tidas como adequadas a priori. |
| `classificar_estado(texto)` | 99–135 | Classificador **baseado em regras** (`if any(p in texto)`). Lista de palavras-chave para cada categoria; default `Neutral (Saudável)`. Caixa-baixa, sem stemming/lemmatization/PLN. |
| `escolher_acao(estado, q_table, epsilon)` | 138–141 | Política **ε-greedy**: com probabilidade ε escolhe aleatoriamente (exploração), caso contrário escolhe a ação com maior valor Q (explotação). Retorna `(ação, modo)`. |
| `calcular_impacto(...)` | 144–180 | Soma ponderada de 8 sinais booleanos do ambiente: positivos (+1 a +5) e negativos (-2 a -5). Resultado é um inteiro. |
| `atualizar_q_learning(...)` | 183–193 | Atualização canônica de Q-Learning: `Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]`. Retorna `(q_antigo, q_novo, melhor_q_futuro)`. |
| `simular_proximo_estado()` | 196–197 | Sorteia `s'` uniformemente entre as 12 categorias. **É uma simulação**, não derivada do ambiente real. |
| `registrar_episodio(...)` | 200–224 | Acrescenta um dicionário ao `st.session_state.historico` com 11 campos do episódio. |

### 5.5 Inicialização do `session_state` (linhas 231–247)
Seis chaves: `q_table`, `historico`, `ultimo_estado`, `ultima_acao`, `ultimo_modo`, `ultimo_texto`. Inicializadas apenas se ausentes, preservando estado entre reruns do Streamlit.

### 5.6 Sidebar (linhas 253–295)
Três sliders para os hiperparâmetros (`alpha`, `gamma`, `epsilon`), uma caixa informativa e um botão "🔄 Reiniciar aprendizado" que reseta tudo via `st.rerun()`.

### 5.7 Interface principal — 5 abas (linhas 313–595)

| Aba | Linhas | Função |
|---|---|---|
| **🔍 Auditoria** | 325–372 | Captura texto, dispara classificação + escolha de ação, exibe resultado em card colorido. |
| **🎯 Impacto** | 378–470 | Captura feedback humano (radio com 5 níveis) e 8 checkboxes de métrica ambiental; calcula recompensa; aplica atualização da Q-Table; registra episódio. |
| **📊 Q-Table** | 476–494 | Exibe DataFrame da Q-Table inteira; permite selecionar um estado e visualizar gráfico de barras das ações para esse estado; mostra a ação preferida atual. |
| **🧠 Histórico** | 500–513 | Tabela de episódios + dois gráficos de linha (evolução de recompensa total e do valor Q). |
| **📚 Explicação** | 519–595 | Texto acadêmico, tabela de mapeamento dos elementos do RL para o domínio, fórmula em LaTeX, equação da recompensa total. |

## 6. Fluxo de funcionamento da aplicação

1. **Inicialização** (toda vez que o Streamlit re-renderiza):
   - Configura página.
   - Injeta CSS.
   - Inicializa `session_state` (uma única vez por sessão).
   - Renderiza sidebar com os hiperparâmetros atuais.

2. **Ciclo de uso típico** (um episódio):
   1. Pessoa usuária abre a aba **Auditoria**, digita uma mensagem e clica em "Analisar mensagem".
   2. `classificar_estado()` retorna o **estado** `s` (uma das 12 categorias).
   3. `escolher_acao()` retorna a **ação** `a` segundo política ε-greedy aplicada à linha `s` da Q-Table.
   4. Resultado é armazenado em `session_state` e exibido em card.
   5. Pessoa usuária vai para a aba **Impacto**, avalia a ação (feedback humano) e marca métricas observáveis.
   6. `calcular_impacto()` produz `R_impacto`; `R_total = R_humano + R_impacto`.
   7. Clique em "Aplicar recompensa": `simular_proximo_estado()` sorteia `s'`; `atualizar_q_learning()` aplica a fórmula e modifica a Q-Table in-place; `registrar_episodio()` arquiva o registro.
   8. Pessoa usuária inspeciona o aprendizado nas abas **Q-Table** e **Histórico**.

3. **Reset**: botão da sidebar limpa Q-Table, histórico e estado da "última interação".

## 7. Padrões arquiteturais identificados

- **Arquitetura "script-first" do Streamlit**: o script é re-executado de cima para baixo a cada interação; persistência intra-sessão se dá via `st.session_state`.
- **Separação informal Percepção × Decisão**: ainda que dentro do mesmo arquivo, há divisão clara entre `classificar_estado` (camada perceptual, baseada em regras) e `escolher_acao` + `atualizar_q_learning` (camada decisional, baseada em RL).
- **Padrão "stateful single-page"**: cinco abas compartilham um único estado mutável (`session_state`).
- **Modelagem MDP tabular**: estados discretos + ações discretas + Q-Table 2D em pandas.
- **Política ε-greedy com hiperparâmetros expostos** na UI — adequado para finalidade pedagógica.
- **Recompensa composta multi-sinal** (`R_humano + R_impacto`): padrão de *shaped reward* que combina feedback explícito e implícito.

## 8. Pontos fortes

1. **Coesão didática**: a aba "Explicação" amarra a UI à formulação matemática, transformando o app em material acadêmico autoexplicativo.
2. **Separação Percepção × Decisão é explícita** no design (e reforçada no texto da aba 5). Permite trocar o classificador (regras → PLN → modelo supervisionado) sem mexer no RL.
3. **Recompensa composta** é conceitualmente mais rica que feedback humano puro — torna o agente sensível a consequências de médio prazo (retorno da pessoa afetada, novo PR, abandono).
4. **Q-Table inicializada com viés especialista**: as 6 células "semeadas" (linhas 89–94) codificam conhecimento de domínio, acelerando a convergência inicial e evitando que o agente parta de um chute completamente aleatório.
5. **Hiperparâmetros expostos na UI**: o uso de sliders para α, γ e ε é uma escolha pedagogicamente acertada — permite experimentação direta dos conceitos de RL.
6. **Histórico rastreável**: cada episódio guarda 11 campos (incluindo `Q antigo`/`Q novo`), permitindo auditoria do aprendizado.
7. **Interface visualmente consistente**: paleta deliberada, cards diferenciados por severidade, ícones temáticos.

## 9. Possíveis melhorias (observações, sem prescrição)

> *Listadas apenas como pontos observáveis no código atual; nenhuma alteração foi feita.*

### Engenharia de software

- **Ausência de `requirements.txt`** (ou equivalente): impede instalação reprodutível. Um novo desenvolvedor precisa adivinhar versões de `streamlit`, `pandas`, `numpy`.
- **Single-file de 596 linhas**: mistura constantes de domínio, lógica de RL, classificação e UI. Há espaço para extrair módulos (`domain/`, `agent/`, `ui/`).
- **Sem testes**: as funções puras (`calcular_impacto`, `atualizar_q_learning`, `classificar_estado`, `escolher_acao`) são triviais de testar em isolamento, mas não há suíte.
- **Sem `.gitignore`**: arquivos como `__pycache__/`, `.venv/`, `.streamlit/secrets.toml` poderiam vazar para o repositório.
- **Sem CI/CD ou linter** configurados.

### Modelagem e RL

- **`simular_proximo_estado()` sorteia uniformemente**: o `s'` retornado não tem relação causal com `s` nem com `a`, então o termo `γ·max Q(s',a')` adiciona ruído em vez de informação. Em uma versão real, `s'` deveria vir da próxima interação observada.
- **Classificador baseado em substring simples**: `classificar_estado` faz `palavra in texto.lower()`, sem tratamento de negação ("não é uma ameaça" cairia em `Threats`), contexto, ironia, ou normalização. O próprio código (aba 5) admite que essa camada pode/deveria ser substituída.
- **Inicialização aleatória das demais células** (`np.random.uniform(0.1, 0.5)`): pode mascarar a leitura da Q-Table em estágios iniciais (valores aleatórios próximos podem "ganhar" do viés especialista por sorte).
- **Não há persistência da Q-Table**: todo aprendizado se perde a cada fim de sessão. Para uso real, seria necessário serializar (parquet, json, sqlite).
- **Categoria `Anti-LGBTQ+` é detectada por substrings de identidades** ("lgbt", "gay", "lésbica", "trans"), o que pode causar **falso-positivo** em discussões legítimas envolvendo essas palavras. Pode merecer revisão da heurística.

### Documentação

- **`README.md` tem 3 linhas**: declara propósito, mas não explica como instalar, executar ou contribuir.
- **Não há documentação técnica** (até este arquivo) explicando a arquitetura ou o domínio de RL.

## 10. Hipóteses e dúvidas registradas

- **Hipótese — versões das dependências**: como não há `requirements.txt`, presume-se versões recentes (Streamlit ≥ 1.30, pandas ≥ 2.0, numpy ≥ 1.24). Não foi verificado em ambiente de execução.
- **Hipótese — público-alvo**: pelo tom do README e pela aba de "Explicação acadêmica", o projeto parece ter finalidade de **TCC, artigo acadêmico ou demo de pesquisa**, não produção. O nome `Athena-RL` reforça caráter de pesquisa.
- **Dúvida — origem do nome `athenarl`**: provável referência à deusa grega Atena (sabedoria, justiça, defesa) + sigla RL (Reinforcement Learning). Não documentado no código.
- **Dúvida — versão atual**: o título da aba sidebar diz "Athena-RL" sem número de versão; o histórico Git tem 5 commits prévios indicando refatorações iterativas. A nova branch `v.2` (alvo desta análise) sugere intenção de versão evolutiva.
- **Dúvida — escopo do "ambiente"**: o termo "GitHub, fórum técnico ou comunidade STEM" (aba 5) é o ambiente *conceitual*, mas o app é apenas um simulador isolado — não há integração real com nenhuma dessas plataformas.

---

**Arquivo gerado em**: 2026-06-03
**Branch analisada**: `v.2`
**Commit base**: `1710aac` (Refactor Q-Learning implementation and UI updates)
