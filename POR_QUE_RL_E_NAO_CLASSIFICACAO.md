# Por que o Athena-RL é um projeto de Aprendizado por Reforço e não de Classificação

> Documento de esclarecimento conceitual, gerado a partir de pergunta direta sobre a natureza do paradigma adotado no projeto.
> Toda a argumentação referencia trechos exatos de `app.py` (commit base `1710aac`, branch `v.2`).

---

## Pergunta original

**"Me explique, exatamente o porque esse projeto é de aprendizagem por reforço e não por classificação."**

---

## Resposta

Excelente pergunta — e é o ponto que distingue a tese central do Athena-RL de um projeto de NLP comum. A explicação está organizada em quatro níveis: definição, evidência no código, contraprova, e o que cada paradigma resolveria.

## 1. Definição operacional dos dois paradigmas

| | **Classificação supervisionada** | **Aprendizado por Reforço** |
|---|---|---|
| Saída do modelo | Um **rótulo** (label) | Uma **ação** (escolhida segundo uma política) |
| Sinal de treino | **Rótulo verdadeiro** dado *a priori* num dataset `(x, y)` | **Recompensa** observada *a posteriori*, depois da ação |
| Função objetivo | Minimizar erro de predição (cross-entropy, MSE…) | Maximizar a soma esperada de recompensas futuras |
| Trade-off típico | Bias × variância | Exploração × explotação |
| Atualização matemática | Gradiente sobre uma perda contra `y_true` | Equação de Bellman: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]` |

## 2. O que o código do Athena-RL realmente faz (evidência linha a linha)

O sistema executa **duas etapas distintas** para cada mensagem analisada:

### Etapa A — Percepção (camada de classificação, baseada em regras)

```python
# app.py:99–135
def classificar_estado(texto):
    if any(p in texto for p in ["ameaça", ...]):
        return "Threats"
    ...
    return "Neutral (Saudável)"
```

Isso é, sim, uma classificação — mas **codificada manualmente**, não aprendida. Ela produz o **estado `s`** que será entregue ao agente. A saída dessa função **não é a resposta final do sistema**; é apenas o ponto de partida.

### Etapa B — Decisão (camada de RL, que é onde está o aprendizado)

```python
# app.py:138–141
def escolher_acao(estado, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS), "Exploração"
    return q_table.loc[estado].idxmax(), "Explotação"
```

Aqui o agente recebe `s` e escolhe entre as **6 ações de mediação** (silêncio, sugerir reescrita, alerta, intervenção coletiva, mediação humana, reportar). A escolha **não vem de um rótulo treinado** — vem da Q-Table mais um termo de exploração aleatória (ε-greedy). Esse ε-greedy é **assinatura clássica de RL**; classificação não tem exploração.

### Etapa C — Aprendizado (fórmula de Bellman)

```python
# app.py:183–193
q_novo = q_antigo + alpha * (
    recompensa_total + gamma * melhor_q_futuro - q_antigo
)
```

Esta é, literalmente, a **equação de atualização do Q-Learning** (Watkins, 1989). Não há `loss.backward()`, não há `y_true`, não há `model.fit(X, y)`. A Q-Table não é treinada contra rótulos — ela é atualizada contra um **sinal de recompensa observado depois da ação**:

```python
# app.py:434
recompensa_total = feedback_humano + impacto
```

onde `impacto` vem de **8 métricas ambientais** (`calcular_impacto`, linhas 144–180): a pessoa afetada continuou participando? Enviou novo PR? Houve nova denúncia? Abandonou a discussão? Essas são **consequências observáveis no tempo**, não rótulos pré-definidos.

## 3. Contraprova: o que seria preciso para isso ser classificação?

Se o projeto fosse de classificação supervisionada, teríamos:

- Um dataset `(mensagem, ação_correta)` com milhares de exemplos rotulados por especialistas.
- Uma função de perda comparando a ação prevista com `ação_correta`.
- **Nenhuma necessidade de feedback humano nem de métricas de impacto durante a operação** — o modelo já saberia a resposta certa do treinamento.
- A política seria sempre a mesma: `argmax P(ação | mensagem)`. Sem ε, sem exploração, sem aprendizado online.

Nada disso existe no `app.py`. Não há dataset rotulado de pares (estado → ação correta), não há perda supervisionada, não há `model.fit`. Há, em vez disso, **uma matriz Q que evolui com cada episódio operacional**.

## 4. Por que RL é o paradigma correto para a tarefa de mediação

A escolha não é estilística — é forçada pela natureza do problema:

1. **Não existe "ação objetivamente correta" para um estado.** Para uma mensagem de "Descrédito", às vezes a melhor mediação é "Sugerir Reescrita"; em outra comunidade, ou após reincidência, talvez seja "Mediação Direta". A política precisa **adaptar**.

2. **A qualidade da ação só se revela depois.** Você só sabe se "Alerta de Viés Social" funcionou observando se a discussão continuou saudável, se a pessoa afetada permaneceu, se houve reincidência — isso é **recompensa atrasada**, característica de RL.

3. **A recompensa é composta e parcialmente subjetiva.** `R = R_humano + R_impacto` mistura juízo humano explícito com sinais observáveis. Classificação não modela isso; RL sim.

4. **O agente precisa explorar.** Se ele sempre escolher a ação que parece melhor agora (`argmax`), nunca descobrirá que outra estratégia pode ser superior naquele contexto. O ε na linha 139 garante essa exploração — algo inexistente em classificação.

## 5. O resumo está literalmente no próprio código

Nas linhas **581–583** de `app.py`, a aba "Explicação" declara a tese de defesa do projeto:

> "O Athena **não usa Aprendizado por Reforço para classificar o texto**. Ele usa Aprendizado por Reforço para aprender **qual ação de mediação gera melhor impacto** no ambiente."

Em uma frase: **a classificação produz o estado `s`; o RL aprende a política `π(a|s)` que maximiza a recompensa de longo prazo no ambiente.** São tarefas diferentes, em camadas diferentes, com naturezas matemáticas diferentes — e é exatamente essa separação que justifica chamar o projeto de RL, não de classificação.

---

**Arquivo gerado em**: 2026-06-03
**Branch**: `v.2`
**Referência principal**: `app.py` (linhas 99–135, 138–141, 144–180, 183–193, 434, 581–583)
