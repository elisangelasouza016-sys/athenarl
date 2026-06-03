import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Athena-RL | Q-Learning com Feedback Humano",
    layout="wide",
    page_icon="🛡️"
)

# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #F4EFF7;
}
.athena-title {
    color: #6B5B95;
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: bold;
    margin-bottom: 5px;
}
.athena-subtitle {
    color: #FF6F61;
    font-size: 1.1rem;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 12px;
    background-color: #FAFAFA;
    border-left: 6px solid #6B5B95;
    margin-bottom: 15px;
}
.warning-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #FFECEA;
    border-left: 6px solid #FF6F61;
    color: #922B21;
    margin-bottom: 15px;
}
.success-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #EAF2F8;
    border-left: 6px solid #6B5B95;
    color: #1F618D;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# BASE DO AMBIENTE
# ============================================================

CATEGORIES = [
    "Discredit (Descrédito)",
    "Stereotyping (Estereótipos)",
    "Dominance (Dominação)",
    "Dismissing (Desprezo)",
    "Sexual Harassment",
    "Threats",
    "Maternal Insults",
    "Objectification",
    "Anti-LGBTQ+",
    "Physical Appearance",
    "Moral Condemnation",
    "Neutral (Saudável)"
]

ACTIONS = [
    "Silêncio Operacional",
    "Sugerir Reescrita Técnico-Pedagógica",
    "Alerta de Viés Social",
    "Intervenção Educativa Coletiva",
    "Mediação Direta (Humano)",
    "Reportar p/ Governança Institucional"
]

# ============================================================
# HIPERPARÂMETROS DO Q-LEARNING
# ============================================================

DEFAULT_ALPHA = 0.30     # Taxa de aprendizado
DEFAULT_GAMMA = 0.80     # Fator de desconto
DEFAULT_EPSILON = 0.20   # Probabilidade de exploração

# ============================================================
# FUNÇÕES DO SISTEMA
# ============================================================

def inicializar_q_table():
    """
    Inicializa a Q-Table com valores pequenos e alguns pesos pedagógicos iniciais.
    Esses pesos não são o aprendizado final; são apenas uma base inicial.
    """
    np.random.seed(42)
    q_data = np.random.uniform(
        low=0.1,
        high=0.5,
        size=(len(CATEGORIES), len(ACTIONS))
    )

    q_table = pd.DataFrame(
        q_data,
        index=CATEGORIES,
        columns=ACTIONS
    )

    # Pesos iniciais coerentes para demonstração
    q_table.loc["Neutral (Saudável)", "Silêncio Operacional"] = 1.20
    q_table.loc["Discredit (Descrédito)", "Sugerir Reescrita Técnico-Pedagógica"] = 1.00
    q_table.loc["Stereotyping (Estereótipos)", "Alerta de Viés Social"] = 1.00
    q_table.loc["Dominance (Dominação)", "Mediação Direta (Humano)"] = 1.00
    q_table.loc["Sexual Harassment", "Reportar p/ Governança Institucional"] = 1.30
    q_table.loc["Threats", "Reportar p/ Governança Institucional"] = 1.40

    return q_table


def classificar_estado(texto):
    """
    Camada de percepção textual.
    Esta parte NÃO é o núcleo de Aprendizado por Reforço.
    Ela transforma o texto em um estado do ambiente.
    """
    texto = texto.lower()

    if any(p in texto for p in ["ameaça", "vou te pegar", "cuidado comigo", "você vai se arrepender"]):
        return "Threats"

    if any(p in texto for p in ["sair comigo", "encontrar depois", "te contrato se", "jantar comigo"]):
        return "Sexual Harassment"

    if any(p in texto for p in ["batom", "bonita", "linda", "foto", "aparência", "corpo"]):
        return "Physical Appearance"

    if any(p in texto for p in ["mulher geralmente", "coisa de mulher", "homem entende mais", "menina não"]):
        return "Stereotyping (Estereótipos)"

    if any(p in texto for p in ["amadora", "amador", "não entende", "não sabe programar", "sem capacidade"]):
        return "Discredit (Descrédito)"

    if any(p in texto for p in ["cale", "fica quieta", "ignore ela", "eu assumo daqui", "do meu jeito"]):
        return "Dominance (Dominação)"

    if any(p in texto for p in ["isso é besteira", "mimimi", "drama", "exagero"]):
        return "Dismissing (Desprezo)"

    if any(p in texto for p in ["mãe não dá conta", "vai cuidar dos filhos", "maternidade atrapalha"]):
        return "Maternal Insults"

    if any(p in texto for p in ["objeto", "gostosa", "delícia"]):
        return "Objectification"

    if any(p in texto for p in ["lgbt", "gay", "lésbica", "trans"]):
        return "Anti-LGBTQ+"

    if any(p in texto for p in ["imoral", "sem valor", "vergonha", "não presta"]):
        return "Moral Condemnation"

    return "Neutral (Saudável)"


def escolher_acao(estado, q_table, epsilon):
    """
    Política epsilon-greedy:
    - Com probabilidade epsilon, o agente explora uma ação aleatória.
    - Caso contrário, escolhe a melhor ação conhecida na Q-Table.
    """
    if np.random.rand() < epsilon:
        acao = np.random.choice(ACTIONS)
        modo = "Exploração"
    else:
        acao = q_table.loc[estado].idxmax()
        modo = "Explotação"

    return acao, modo


def atualizar_q_learning(q_table, estado, acao, recompensa, proximo_estado, alpha, gamma):
    """
    Atualização formal de Q-Learning:

    Q(s,a) = Q(s,a) + α * [r + γ * maxQ(s',a') - Q(s,a)]
    """
    q_antigo = q_table.loc[estado, acao]
    melhor_q_futuro = q_table.loc[proximo_estado].max()

    q_novo = q_antigo + alpha * (
        recompensa + gamma * melhor_q_futuro - q_antigo
    )

    q_table.loc[estado, acao] = q_novo

    return q_antigo, q_novo, melhor_q_futuro


def registrar_episodio(
    texto,
    estado,
    acao,
    modo,
    recompensa,
    q_antigo,
    q_novo,
    proximo_estado
):
    episodio = {
        "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Texto": texto,
        "Estado": estado,
        "Ação": acao,
        "Política": modo,
        "Recompensa": recompensa,
        "Q antigo": round(q_antigo, 4),
        "Q novo": round(q_novo, 4),
        "Próximo estado": proximo_estado
    }

    st.session_state.historico.append(episodio)


def simular_proximo_estado():
    """
    Em um sistema real, o próximo estado viria da próxima mensagem da comunidade.
    Para o protótipo acadêmico, usamos uma transição simulada.
    """
    return np.random.choice(CATEGORIES)


# ============================================================
# SESSION STATE
# ============================================================

if "q_table" not in st.session_state:
    st.session_state.q_table = inicializar_q_table()

if "historico" not in st.session_state:
    st.session_state.historico = []

if "ultimo_estado" not in st.session_state:
    st.session_state.ultimo_estado = None

if "ultima_acao" not in st.session_state:
    st.session_state.ultima_acao = None

if "ultimo_modo" not in st.session_state:
    st.session_state.ultimo_modo = None

if "ultimo_texto" not in st.session_state:
    st.session_state.ultimo_texto = None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🛡️ Athena-RL")
st.sidebar.markdown("**Governança ética com Q-Learning e feedback humano**")

st.sidebar.divider()

st.sidebar.markdown("### Hiperparâmetros")

alpha = st.sidebar.slider(
    "Alpha — taxa de aprendizado",
    min_value=0.01,
    max_value=1.00,
    value=DEFAULT_ALPHA,
    step=0.01
)

gamma = st.sidebar.slider(
    "Gamma — fator de desconto",
    min_value=0.00,
    max_value=1.00,
    value=DEFAULT_GAMMA,
    step=0.01
)

epsilon = st.sidebar.slider(
    "Epsilon — exploração",
    min_value=0.00,
    max_value=1.00,
    value=DEFAULT_EPSILON,
    step=0.01
)

st.sidebar.info(
    "Alpha controla o quanto o agente aprende. "
    "Gamma considera recompensas futuras. "
    "Epsilon permite explorar ações novas."
)

if st.sidebar.button("🔄 Reiniciar aprendizado"):
    st.session_state.q_table = inicializar_q_table()
    st.session_state.historico = []
    st.session_state.ultimo_estado = None
    st.session_state.ultima_acao = None
    st.session_state.ultimo_modo = None
    st.session_state.ultimo_texto = None
    st.rerun()

# ============================================================
# TÍTULO
# ============================================================

st.markdown(
    "<h1 class='athena-title'>🛡️ Athena-RL: Aprendizado por Reforço com Feedback Humano</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='athena-subtitle'>"
    "O sistema identifica um estado da interação e aprende qual ação de mediação é mais adequada."
    "</p>",
    unsafe_allow_html=True
)

# ============================================================
# ABAS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Auditoria",
    "📊 Q-Table",
    "🧠 Episódios",
    "📈 Simulação",
    "📚 Explicação"
])

# ============================================================
# ABA 1 — AUDITORIA
# ============================================================

with tab1:
    st.subheader("Auditoria de mensagem e decisão do agente")

    col_input, col_output = st.columns([1, 1])

    with col_input:
        comentario = st.text_area(
            "Insira uma mensagem simulada de fórum técnico:",
            height=140,
            placeholder="Exemplo: 'Você não entende muito de backend, melhor deixar essa parte para alguém mais experiente.'"
        )

        analisar = st.button("Analisar mensagem", use_container_width=True)

        if analisar and comentario.strip():
            estado = classificar_estado(comentario)
            acao, modo = escolher_acao(estado, st.session_state.q_table, epsilon)

            st.session_state.ultimo_estado = estado
            st.session_state.ultima_acao = acao
            st.session_state.ultimo_modo = modo
            st.session_state.ultimo_texto = comentario

    with col_output:
        if st.session_state.ultimo_estado:
            estado = st.session_state.ultimo_estado
            acao = st.session_state.ultima_acao
            modo = st.session_state.ultimo_modo

            if estado == "Neutral (Saudável)":
                st.markdown(
                    f"""
                    <div class='success-card'>
                    ✅ <b>Estado identificado:</b> {estado}<br>
                    🤖 <b>Ação escolhida:</b> {acao}<br>
                    🎯 <b>Política usada:</b> {modo}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='warning-card'>
                    ⚠️ <b>Estado identificado:</b> {estado}<br>
                    🤖 <b>Ação escolhida:</b> {acao}<br>
                    🎯 <b>Política usada:</b> {modo}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### Feedback humano")

            col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)

            feedbacks = [
                ("Excelente", 1.0, col_r1),
                ("Adequada", 0.5, col_r2),
                ("Neutra", 0.0, col_r3),
                ("Exagerada", -0.5, col_r4),
                ("Prejudicial", -1.0, col_r5),
            ]

            for label, recompensa, coluna in feedbacks:
                with coluna:
                    if st.button(label, use_container_width=True):
                        proximo_estado = simular_proximo_estado()

                        q_antigo, q_novo, melhor_q_futuro = atualizar_q_learning(
                            q_table=st.session_state.q_table,
                            estado=st.session_state.ultimo_estado,
                            acao=st.session_state.ultima_acao,
                            recompensa=recompensa,
                            proximo_estado=proximo_estado,
                            alpha=alpha,
                            gamma=gamma
                        )

                        registrar_episodio(
                            texto=st.session_state.ultimo_texto,
                            estado=st.session_state.ultimo_estado,
                            acao=st.session_state.ultima_acao,
                            modo=st.session_state.ultimo_modo,
                            recompensa=recompensa,
                            q_antigo=q_antigo,
                            q_novo=q_novo,
                            proximo_estado=proximo_estado
                        )

                        st.success(
                            f"Recompensa {recompensa} aplicada. "
                            f"Q antigo: {q_antigo:.4f} → Q novo: {q_novo:.4f}"
                        )

# ============================================================
# ABA 2 — Q-TABLE
# ============================================================

with tab2:
    st.subheader("Matriz de aprendizado — Q-Table")

    estado_visualizar = st.selectbox(
        "Escolha um estado para visualizar os valores das ações:",
        CATEGORIES
    )

    st.dataframe(
        st.session_state.q_table,
        use_container_width=True
    )

    st.markdown(f"### Pesos atuais para: {estado_visualizar}")
    st.bar_chart(st.session_state.q_table.loc[estado_visualizar])

    melhor_acao = st.session_state.q_table.loc[estado_visualizar].idxmax()
    melhor_valor = st.session_state.q_table.loc[estado_visualizar].max()

    st.info(
        f"A ação atualmente preferida para este estado é: "
        f"**{melhor_acao}** com valor Q = **{melhor_valor:.4f}**."
    )

# ============================================================
# ABA 3 — HISTÓRICO DE EPISÓDIOS
# ============================================================

with tab3:
    st.subheader("Histórico de episódios de aprendizagem")

    if st.session_state.historico:
        df_hist = pd.DataFrame(st.session_state.historico)
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("### Evolução das recompensas")
        st.line_chart(df_hist["Recompensa"])
    else:
        st.info("Nenhum episódio registrado ainda. Analise uma mensagem e aplique um feedback.")

# ============================================================
# ABA 4 — SIMULAÇÃO AUTOMÁTICA
# ============================================================

with tab4:
    st.subheader("Simulação de episódios")

    st.write(
        "Esta aba executa episódios simulados para demonstrar a evolução da Q-Table. "
        "Ela serve para apresentação acadêmica do processo de aprendizagem."
    )

    exemplos = [
        "Você não entende muito de backend, melhor alguém mais experiente revisar.",
        "Mulher geralmente é melhor em documentação do que em infraestrutura.",
        "Ignore ela, eu assumo essa parte do código.",
        "Esse comentário é mimimi, estamos falando de programação.",
        "Seu código ficou bom, parabéns pela solução técnica.",
        "Você está linda na foto do perfil, quase nem reparei no código.",
        "Se sair comigo, talvez eu aprove seu pull request."
    ]

    n_episodios = st.slider(
        "Número de episódios simulados",
        min_value=1,
        max_value=50,
        value=10
    )

    if st.button("Executar simulação", use_container_width=True):
        for _ in range(n_episodios):
            texto = np.random.choice(exemplos)
            estado = classificar_estado(texto)
            acao, modo = escolher_acao(estado, st.session_state.q_table, epsilon)

            # Recompensa simulada simples:
            # premia ações mais coerentes para alguns estados críticos.
            if estado == "Neutral (Saudável)" and acao == "Silêncio Operacional":
                recompensa = 1.0
            elif estado in ["Sexual Harassment", "Threats"] and acao == "Reportar p/ Governança Institucional":
                recompensa = 1.0
            elif estado in ["Discredit (Descrédito)", "Stereotyping (Estereótipos)"] and acao in [
                "Sugerir Reescrita Técnico-Pedagógica",
                "Alerta de Viés Social"
            ]:
                recompensa = 0.8
            elif acao == "Silêncio Operacional" and estado != "Neutral (Saudável)":
                recompensa = -1.0
            else:
                recompensa = -0.2

            proximo_estado = simular_proximo_estado()

            q_antigo, q_novo, melhor_q_futuro = atualizar_q_learning(
                q_table=st.session_state.q_table,
                estado=estado,
                acao=acao,
                recompensa=recompensa,
                proximo_estado=proximo_estado,
                alpha=alpha,
                gamma=gamma
            )

            registrar_episodio(
                texto=texto,
                estado=estado,
                acao=acao,
                modo=modo,
                recompensa=recompensa,
                q_antigo=q_antigo,
                q_novo=q_novo,
                proximo_estado=proximo_estado
            )

        st.success(f"{n_episodios} episódios simulados executados.")

# ============================================================
# ABA 5 — EXPLICAÇÃO
# ============================================================

with tab5:
    st.subheader("Por que este projeto é Aprendizado por Reforço?")

    st.markdown("""
    O Athena-RL é uma arquitetura híbrida.

    A primeira camada identifica o **estado** da mensagem. Nesta versão didática,
    essa identificação é feita por regras simples de PLN. Essa etapa não é o núcleo
    do Aprendizado por Reforço.

    O Aprendizado por Reforço ocorre na segunda camada: a tomada de decisão.

    O agente observa um estado, escolhe uma ação, recebe uma recompensa e atualiza
    sua política de decisão por meio da Q-Table.
    """)

    st.markdown("### Elementos do modelo")

    df_elementos = pd.DataFrame({
        "Elemento de RL": [
            "Agente",
            "Ambiente",
            "Estado",
            "Ação",
            "Recompensa",
            "Política",
            "Aprendizado"
        ],
        "No Athena-RL": [
            "Sistema Athena",
            "Comunidade técnica / fórum STEM",
            "Categoria da interação textual",
            "Estratégia de mediação",
            "Feedback humano positivo, neutro ou negativo",
            "Epsilon-greedy",
            "Atualização da Q-Table por Q-Learning"
        ]
    })

    st.table(df_elementos)

    st.markdown("### Fórmula usada")

    st.latex(r"""
    Q(s,a) = Q(s,a) + \alpha \cdot [r + \gamma \cdot \max Q(s',a') - Q(s,a)]
    """)

    st.markdown("""
    Onde:

    - **s** é o estado atual;
    - **a** é a ação escolhida;
    - **r** é a recompensa recebida;
    - **s'** é o próximo estado;
    - **α** é a taxa de aprendizado;
    - **γ** é o fator de desconto;
    - **max Q(s',a')** representa a melhor recompensa futura esperada.

    Assim, o sistema não apenas classifica uma mensagem. Ele aprende, por tentativa,
    erro e feedback humano, qual ação de mediação tende a ser mais adequada.
    """)

    st.warning(
        "Defesa técnica: a classificação textual inicial é uma camada de percepção. "
        "O núcleo de Aprendizado por Reforço está na escolha adaptativa da ação."
    )
