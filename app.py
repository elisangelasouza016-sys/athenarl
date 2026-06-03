import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Athena-RL | Apoio à Permanência Feminina em STEM",
    layout="wide",
    page_icon="🛡️"
)

# =========================
# ESTILO
# =========================

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #F4EFF7; }

.athena-title {
    color: #6B5B95;
    font-weight: bold;
}

.athena-subtitle {
    color: #FF6F61;
    font-size: 1.1rem;
}

.card {
    padding: 18px;
    border-radius: 12px;
    background-color: #FAFAFA;
    border-left: 6px solid #6B5B95;
    margin-bottom: 15px;
}

.alert-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #FFECEA;
    border-left: 6px solid #FF6F61;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# ESTADOS E AÇÕES
# =========================

STATES = [
    "Descrédito técnico",
    "Silenciamento",
    "Estereótipo de gênero",
    "Isolamento",
    "Assédio",
    "Comentário sobre aparência",
    "Síndrome do impostor",
    "Ambiente saudável"
]

ACTIONS = [
    "Acolhimento individual",
    "Resposta sugerida",
    "Material educativo",
    "Mentoria",
    "Grupo de apoio",
    "Encaminhamento institucional",
    "Nenhuma intervenção"
]


ACTION_MESSAGES = {
    "Acolhimento individual": (
        "Você não está sozinha. O que você relatou pode impactar sua confiança "
        "e seu sentimento de pertencimento. O primeiro passo é reconhecer que essa situação merece atenção."
    ),
    "Resposta sugerida": (
        "Uma resposta possível seria: 'Gostaria que a análise fosse feita sobre a situação ou solução apresentada, "
        "sem pressupor minha capacidade ou meu lugar nesse ambiente.'"
    ),
    "Material educativo": (
        "O sistema recomenda apresentar um conteúdo educativo sobre como esse tipo de comportamento afeta "
        "a permanência de mulheres em STEM."
    ),
    "Mentoria": (
        "O sistema recomenda buscar uma pessoa mentora ou referência de confiança para apoiar sua permanência "
        "e fortalecer sua trajetória."
    ),
    "Grupo de apoio": (
        "O sistema recomenda contato com um grupo de apoio, rede de mulheres em tecnologia ou coletivo institucional."
    ),
    "Encaminhamento institucional": (
        "O sistema recomenda encaminhar a situação para uma instância responsável, como coordenação, professor, RH, "
        "ou canal formal de denúncia."
    ),
    "Nenhuma intervenção": (
        "O sistema entende que, neste caso, talvez não seja necessária uma intervenção imediata, apenas registro ou observação."
    )
}


# =========================
# HIPERPARÂMETROS
# =========================

DEFAULT_ALPHA = 0.30
DEFAULT_GAMMA = 0.80
DEFAULT_EPSILON = 0.20


# =========================
# FUNÇÕES
# =========================

def inicializar_q_table():
    np.random.seed(42)

    q_data = np.random.uniform(
        low=0.1,
        high=0.5,
        size=(len(STATES), len(ACTIONS))
    )

    q_table = pd.DataFrame(
        q_data,
        index=STATES,
        columns=ACTIONS
    )

    # Valores iniciais apenas para guiar a demonstração
    q_table.loc["Descrédito técnico", "Resposta sugerida"] = 1.0
    q_table.loc["Silenciamento", "Mentoria"] = 1.0
    q_table.loc["Estereótipo de gênero", "Material educativo"] = 1.0
    q_table.loc["Isolamento", "Grupo de apoio"] = 1.0
    q_table.loc["Assédio", "Encaminhamento institucional"] = 1.2
    q_table.loc["Síndrome do impostor", "Acolhimento individual"] = 1.0
    q_table.loc["Ambiente saudável", "Nenhuma intervenção"] = 1.0

    return q_table


def escolher_acao(estado, q_table, epsilon):
    """
    Política epsilon-greedy:
    - Exploração: testa ação aleatória.
    - Explotação: escolhe a ação com maior valor Q.
    """
    if np.random.rand() < epsilon:
        acao = np.random.choice(ACTIONS)
        politica = "Exploração"
    else:
        acao = q_table.loc[estado].idxmax()
        politica = "Explotação"

    return acao, politica


def calcular_recompensa_impacto(
    sentiu_acolhimento=False,
    entendeu_situacao=False,
    pretende_continuar=False,
    buscou_apoio=False,
    indicaria_athena=False,
    sentiu_exposicao=False,
    achou_inutil=False,
    pretende_desistir=False
):
    recompensa = 0

    if sentiu_acolhimento:
        recompensa += 2

    if entendeu_situacao:
        recompensa += 1

    if pretende_continuar:
        recompensa += 3

    if buscou_apoio:
        recompensa += 2

    if indicaria_athena:
        recompensa += 1

    if sentiu_exposicao:
        recompensa -= 2

    if achou_inutil:
        recompensa -= 2

    if pretende_desistir:
        recompensa -= 5

    return recompensa


def atualizar_q_learning(q_table, estado, acao, recompensa_total, proximo_estado, alpha, gamma):
    q_antigo = q_table.loc[estado, acao]
    melhor_q_futuro = q_table.loc[proximo_estado].max()

    q_novo = q_antigo + alpha * (
        recompensa_total + gamma * melhor_q_futuro - q_antigo
    )

    q_table.loc[estado, acao] = q_novo

    return q_antigo, q_novo, melhor_q_futuro


def registrar_episodio(
    relato,
    estado,
    acao,
    politica,
    recompensa_usuario,
    recompensa_impacto,
    recompensa_total,
    q_antigo,
    q_novo,
    proximo_estado
):
    st.session_state.historico.append({
        "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Relato": relato,
        "Estado": estado,
        "Ação": acao,
        "Política": politica,
        "Recompensa da usuária": recompensa_usuario,
        "Recompensa de impacto": recompensa_impacto,
        "Recompensa total": recompensa_total,
        "Q antigo": round(q_antigo, 4),
        "Q novo": round(q_novo, 4),
        "Próximo estado": proximo_estado
    })


def simular_proximo_estado():
    return np.random.choice(STATES)


# =========================
# SESSION STATE
# =========================

if "q_table" not in st.session_state:
    st.session_state.q_table = inicializar_q_table()

if "historico" not in st.session_state:
    st.session_state.historico = []

if "ultimo_relato" not in st.session_state:
    st.session_state.ultimo_relato = ""

if "ultimo_estado" not in st.session_state:
    st.session_state.ultimo_estado = None

if "ultima_acao" not in st.session_state:
    st.session_state.ultima_acao = None

if "ultima_politica" not in st.session_state:
    st.session_state.ultima_politica = None


# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## 🛡️ Athena-RL")
st.sidebar.markdown("**Apoio à permanência feminina em STEM**")

st.sidebar.divider()

alpha = st.sidebar.slider(
    "Alpha — taxa de aprendizado",
    0.01,
    1.00,
    DEFAULT_ALPHA,
    0.01
)

gamma = st.sidebar.slider(
    "Gamma — fator de desconto",
    0.00,
    1.00,
    DEFAULT_GAMMA,
    0.01
)

epsilon = st.sidebar.slider(
    "Epsilon — exploração",
    0.00,
    1.00,
    DEFAULT_EPSILON,
    0.01
)

st.sidebar.info(
    "Alpha define quanto o agente aprende. "
    "Gamma considera recompensas futuras. "
    "Epsilon permite testar ações diferentes."
)

if st.sidebar.button("🔄 Reiniciar aprendizado"):
    st.session_state.q_table = inicializar_q_table()
    st.session_state.historico = []
    st.session_state.ultimo_relato = ""
    st.session_state.ultimo_estado = None
    st.session_state.ultima_acao = None
    st.session_state.ultima_politica = None
    st.rerun()


# =========================
# INTERFACE
# =========================

st.markdown(
    "<h1 class='athena-title'>🛡️ Athena-RL</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='athena-subtitle'>Sistema Inteligente de Apoio à Permanência Feminina em STEM baseado em Aprendizado por Reforço</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Relato",
    "⭐ Feedback",
    "📊 Q-Table",
    "🧠 Histórico",
    "📚 Explicação"
])


# =========================
# ABA 1 — RELATO
# =========================

with tab1:
    st.subheader("1. Registro da situação")

    st.markdown("""
    Nesta versão, a usuária escolhe a categoria da situação vivenciada.  
    O sistema **não aprende a classificar frases**.  
    O Aprendizado por Reforço ocorre na escolha da melhor intervenção.
    """)

    relato = st.text_area(
        "Descreva brevemente a situação vivenciada:",
        height=150,
        placeholder="Exemplo: Durante uma reunião técnica, minha sugestão foi ignorada. Depois, um colega repetiu a mesma ideia e foi elogiado."
    )

    estado = st.selectbox(
        "Qual categoria melhor descreve a situação?",
        STATES
    )

    if st.button("Gerar intervenção do Athena", use_container_width=True):
        acao, politica = escolher_acao(
            estado=estado,
            q_table=st.session_state.q_table,
            epsilon=epsilon
        )

        st.session_state.ultimo_relato = relato
        st.session_state.ultimo_estado = estado
        st.session_state.ultima_acao = acao
        st.session_state.ultima_politica = politica

    if st.session_state.ultimo_estado:
        st.markdown("### Intervenção sugerida")

        st.markdown(
            f"""
            <div class='card'>
            <b>Estado informado pela usuária:</b> {st.session_state.ultimo_estado}<br>
            <b>Ação escolhida pelo agente:</b> {st.session_state.ultima_acao}<br>
            <b>Política usada:</b> {st.session_state.ultima_politica}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(ACTION_MESSAGES[st.session_state.ultima_acao])


# =========================
# ABA 2 — FEEDBACK
# =========================

with tab2:
    st.subheader("2. Feedback da usuária e métrica de impacto")

    if not st.session_state.ultimo_estado:
        st.warning("Primeiro gere uma intervenção na aba Relato.")
    else:
        st.markdown("### A intervenção ajudou?")

        avaliacao = st.radio(
            "Como você avalia a intervenção sugerida?",
            [
                "Muito útil (+2)",
                "Útil (+1)",
                "Neutra (0)",
                "Pouco útil (-1)",
                "Inútil (-2)"
            ]
        )

        avaliacao_map = {
            "Muito útil (+2)": 2,
            "Útil (+1)": 1,
            "Neutra (0)": 0,
            "Pouco útil (-1)": -1,
            "Inútil (-2)": -2
        }

        recompensa_usuario = avaliacao_map[avaliacao]

        st.markdown("### Impactos observados")

        col1, col2 = st.columns(2)

        with col1:
            sentiu_acolhimento = st.checkbox("Senti acolhimento com a orientação (+2)")
            entendeu_situacao = st.checkbox("Entendi melhor a situação vivida (+1)")
            pretende_continuar = st.checkbox("Sinto que consigo continuar em STEM (+3)")
            buscou_apoio = st.checkbox("Pretendo buscar apoio ou mentoria (+2)")

        with col2:
            indicaria_athena = st.checkbox("Indicaria o Athena para outra mulher (+1)")
            sentiu_exposicao = st.checkbox("Senti exposição ou desconforto (-2)")
            achou_inutil = st.checkbox("A intervenção não ajudou (-2)")
            pretende_desistir = st.checkbox("Ainda penso em desistir ou me afastar (-5)")

        recompensa_impacto = calcular_recompensa_impacto(
            sentiu_acolhimento=sentiu_acolhimento,
            entendeu_situacao=entendeu_situacao,
            pretende_continuar=pretende_continuar,
            buscou_apoio=buscou_apoio,
            indicaria_athena=indicaria_athena,
            sentiu_exposicao=sentiu_exposicao,
            achou_inutil=achou_inutil,
            pretende_desistir=pretende_desistir
        )

        recompensa_total = recompensa_usuario + recompensa_impacto

        st.markdown("### Recompensa calculada")

        st.write(f"Recompensa da usuária: **{recompensa_usuario}**")
        st.write(f"Recompensa de impacto: **{recompensa_impacto}**")
        st.write(f"Recompensa total: **{recompensa_total}**")

        if st.button("Aplicar recompensa e atualizar aprendizado", use_container_width=True):
            proximo_estado = simular_proximo_estado()

            q_antigo, q_novo, melhor_q_futuro = atualizar_q_learning(
                q_table=st.session_state.q_table,
                estado=st.session_state.ultimo_estado,
                acao=st.session_state.ultima_acao,
                recompensa_total=recompensa_total,
                proximo_estado=proximo_estado,
                alpha=alpha,
                gamma=gamma
            )

            registrar_episodio(
                relato=st.session_state.ultimo_relato,
                estado=st.session_state.ultimo_estado,
                acao=st.session_state.ultima_acao,
                politica=st.session_state.ultima_politica,
                recompensa_usuario=recompensa_usuario,
                recompensa_impacto=recompensa_impacto,
                recompensa_total=recompensa_total,
                q_antigo=q_antigo,
                q_novo=q_novo,
                proximo_estado=proximo_estado
            )

            st.success(
                f"Aprendizado atualizado: Q antigo {q_antigo:.4f} → Q novo {q_novo:.4f}"
            )


# =========================
# ABA 3 — Q-TABLE
# =========================

with tab3:
    st.subheader("3. Q-Table")

    st.dataframe(st.session_state.q_table, use_container_width=True)

    estado_visualizar = st.selectbox(
        "Visualizar ações para o estado:",
        STATES
    )

    st.bar_chart(st.session_state.q_table.loc[estado_visualizar])

    melhor_acao = st.session_state.q_table.loc[estado_visualizar].idxmax()
    melhor_valor = st.session_state.q_table.loc[estado_visualizar].max()

    st.info(
        f"Para **{estado_visualizar}**, a ação atualmente preferida é "
        f"**{melhor_acao}**, com Q = **{melhor_valor:.4f}**."
    )


# =========================
# ABA 4 — HISTÓRICO
# =========================

with tab4:
    st.subheader("4. Histórico de episódios")

    if st.session_state.historico:
        df_hist = pd.DataFrame(st.session_state.historico)
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("### Evolução da recompensa total")
        st.line_chart(df_hist["Recompensa total"])

        st.markdown("### Evolução do valor Q")
        st.line_chart(df_hist["Q novo"])
    else:
        st.info("Nenhum episódio registrado ainda.")


# =========================
# ABA 5 — EXPLICAÇÃO
# =========================

with tab5:
    st.subheader("5. Explicação do modelo")

    st.markdown("""
    O Athena-RL é um sistema de apoio à permanência feminina em STEM.

    Nesta versão, o sistema não tenta classificar automaticamente o relato da usuária.
    A própria usuária informa qual categoria representa melhor a situação vivenciada.

    O Aprendizado por Reforço acontece na escolha da intervenção.

    O agente observa o estado, escolhe uma ação, recebe uma recompensa e atualiza sua Q-Table.
    """)

    df_rl = pd.DataFrame({
        "Elemento": [
            "Agente",
            "Ambiente",
            "Estado",
            "Ação",
            "Recompensa",
            "Política",
            "Episódio",
            "Aprendizado"
        ],
        "No Athena-RL": [
            "Sistema Athena",
            "Ambiente de apoio à permanência feminina em STEM",
            "Categoria informada pela usuária",
            "Intervenção sugerida pelo sistema",
            "Avaliação da usuária + impacto percebido",
            "Epsilon-greedy",
            "Um relato, uma intervenção e um feedback",
            "Atualização da Q-Table por Q-Learning"
        ]
    })

    st.table(df_rl)

    st.markdown("### Fórmula de Q-Learning")

    st.latex(r"""
    Q(s,a) = Q(s,a) + \alpha \cdot [r + \gamma \cdot \max Q(s',a') - Q(s,a)]
    """)

    st.markdown("""
    A defesa central do projeto é:

    **O Athena não usa Aprendizado por Reforço para classificar textos.  
    Ele usa Aprendizado por Reforço para aprender quais intervenções ajudam mais mulheres
    a permanecerem em ambientes STEM.**
    """)

    st.markdown("### Recompensa total")

    st.latex(r"""
    R_{total} = R_{usuaria} + R_{impacto}
    """)
