import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Athena | Apoio à Permanência Feminina em STEM",
    layout="wide",
    page_icon="🌷"
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

.soft-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #FFF7FA;
    border-left: 6px solid #FF6F61;
    margin-bottom: 15px;
}

.small-text {
    font-size: 0.9rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ESTADOS INTERNOS E AÇÕES
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

OPCOES_USUARIA = [
    "💬 Minha opinião ou contribuição foi desconsiderada",
    "🧠 Minha capacidade técnica foi questionada",
    "🚫 Senti-me excluída ou isolada",
    "⚖️ Vivi uma situação de preconceito ou estereótipo",
    "⚠️ Passei por uma situação de assédio ou constrangimento",
    "🌱 Estou enfrentando insegurança ou síndrome do impostor",
    "🤝 Gostaria apenas de orientação para minha trajetória",
    "✨ Quero compartilhar uma experiência positiva"
]

MAPA_ESTADOS = {
    "💬 Minha opinião ou contribuição foi desconsiderada": "Silenciamento",
    "🧠 Minha capacidade técnica foi questionada": "Descrédito técnico",
    "🚫 Senti-me excluída ou isolada": "Isolamento",
    "⚖️ Vivi uma situação de preconceito ou estereótipo": "Estereótipo de gênero",
    "⚠️ Passei por uma situação de assédio ou constrangimento": "Assédio",
    "🌱 Estou enfrentando insegurança ou síndrome do impostor": "Síndrome do impostor",
    "🤝 Gostaria apenas de orientação para minha trajetória": "Ambiente saudável",
    "✨ Quero compartilhar uma experiência positiva": "Ambiente saudável"
}

ACTION_MESSAGES = {
    "Acolhimento individual": (
        "O que você relatou merece atenção. Situações assim podem afetar a confiança, "
        "o bem-estar e o sentimento de pertencimento. Reconhecer o impacto disso já é "
        "um passo importante para buscar apoio e preservar sua trajetória."
    ),
    "Resposta sugerida": (
        "Uma resposta possível seria: “Gostaria que minha contribuição fosse analisada "
        "pelo conteúdo técnico apresentado, sem pressupor minha capacidade ou meu lugar "
        "nesse ambiente.”"
    ),
    "Material educativo": (
        "Pode ser útil compartilhar ou construir um material educativo sobre como esse "
        "tipo de comportamento afeta a permanência de mulheres em STEM. A conscientização "
        "ajuda o grupo a reconhecer padrões que muitas vezes são naturalizados."
    ),
    "Mentoria": (
        "Conversar com uma mentora, professora, colega experiente ou profissional de "
        "referência pode ajudar a organizar estratégias, fortalecer sua confiança e "
        "apoiar sua continuidade na área."
    ),
    "Grupo de apoio": (
        "Buscar uma rede de mulheres em tecnologia, grupo de estudos, coletivo acadêmico "
        "ou comunidade de apoio pode reduzir o isolamento e fortalecer o sentimento de pertencimento."
    ),
    "Encaminhamento institucional": (
        "Pela natureza da situação, pode ser importante procurar um canal institucional "
        "de apoio, como coordenação, professor responsável, setor de acolhimento, RH, "
        "ou canal formal de denúncia."
    ),
    "Nenhuma intervenção": (
        "Neste momento, talvez o mais adequado seja apenas registrar a experiência, observar "
        "a evolução da situação e preservar seu bem-estar. Nem toda situação exige uma ação imediata."
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

    q_table = pd.DataFrame(q_data, index=STATES, columns=ACTIONS)

    q_table.loc["Descrédito técnico", "Resposta sugerida"] = 1.0
    q_table.loc["Silenciamento", "Mentoria"] = 1.0
    q_table.loc["Estereótipo de gênero", "Material educativo"] = 1.0
    q_table.loc["Isolamento", "Grupo de apoio"] = 1.0
    q_table.loc["Assédio", "Encaminhamento institucional"] = 1.2
    q_table.loc["Síndrome do impostor", "Acolhimento individual"] = 1.0
    q_table.loc["Ambiente saudável", "Nenhuma intervenção"] = 1.0

    return q_table


def escolher_acao(estado, q_table, epsilon):
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
        "Intervenção": acao,
        "Política": politica,
        "Avaliação da usuária": recompensa_usuario,
        "Impacto percebido": recompensa_impacto,
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

if "ultima_opcao_usuario" not in st.session_state:
    st.session_state.ultima_opcao_usuario = None

if "ultima_acao" not in st.session_state:
    st.session_state.ultima_acao = None

if "ultima_politica" not in st.session_state:
    st.session_state.ultima_politica = None


# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## 🌷 Athena")
st.sidebar.markdown("**Apoio à permanência feminina em STEM**")

st.sidebar.divider()

st.sidebar.markdown("### 🔬 Modo acadêmico")

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

st.sidebar.caption(
    "Esses parâmetros são exibidos para fins acadêmicos. "
    "Eles controlam como o agente aprende a escolher intervenções."
)

if st.sidebar.button("🔄 Reiniciar aprendizado"):
    st.session_state.q_table = inicializar_q_table()
    st.session_state.historico = []
    st.session_state.ultimo_relato = ""
    st.session_state.ultimo_estado = None
    st.session_state.ultima_opcao_usuario = None
    st.session_state.ultima_acao = None
    st.session_state.ultima_politica = None
    st.rerun()


# =========================
# CABEÇALHO
# =========================

st.markdown(
    "<h1 class='athena-title'>🌷 Athena</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='athena-subtitle'>Apoio à Permanência Feminina em STEM</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌷 Apoio",
    "⭐ Avaliação",
    "📈 Evolução",
    "📚 Recursos",
    "🔬 Painel Acadêmico"
])


# =========================
# ABA 1 — APOIO
# =========================

with tab1:
    st.markdown("""
    ## 🌷 Bem-vinda à Athena

    Um espaço seguro de apoio para mulheres em STEM.

    Compartilhe uma situação vivenciada em sua trajetória acadêmica ou profissional
    e receba orientações, recursos e estratégias de apoio que podem ajudar no seu
    desenvolvimento, bem-estar e permanência na área.
    """)

    st.divider()

    relato = st.text_area(
        "Conte um pouco sobre o que aconteceu:",
        height=150,
        placeholder=(
            "Você pode compartilhar uma situação, desafio, dúvida ou experiência "
            "vivenciada durante sua trajetória acadêmica ou profissional."
        )
    )

    opcao_escolhida = st.selectbox(
        "Qual situação mais se aproxima da sua experiência?",
        OPCOES_USUARIA
    )

    estado = MAPA_ESTADOS[opcao_escolhida]

    if st.button("💜 Receber orientação da Athena", use_container_width=True):
        acao, politica = escolher_acao(
            estado=estado,
            q_table=st.session_state.q_table,
            epsilon=epsilon
        )

        st.session_state.ultimo_relato = relato
        st.session_state.ultimo_estado = estado
        st.session_state.ultima_opcao_usuario = opcao_escolhida
        st.session_state.ultima_acao = acao
        st.session_state.ultima_politica = politica

    if st.session_state.ultimo_estado:
        st.markdown("### 💜 Sugestão da Athena")

        st.markdown(
            f"""
            <div class='soft-card'>
            {ACTION_MESSAGES[st.session_state.ultima_acao]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "As orientações da Athena são sugestões de apoio e não substituem "
            "acompanhamento institucional, psicológico ou jurídico quando necessário."
        )


# =========================
# ABA 2 — AVALIAÇÃO
# =========================

with tab2:
    st.subheader("⭐ Como essa orientação ajudou?")

    if not st.session_state.ultimo_estado:
        st.warning("Primeiro receba uma orientação na aba Apoio.")
    else:
        st.markdown("""
        Sua avaliação ajuda a Athena a melhorar as próximas orientações.
        """)

        avaliacao = st.radio(
            "Como você avalia a orientação recebida?",
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

        st.markdown("### Impacto percebido")

        col1, col2 = st.columns(2)

        with col1:
            sentiu_acolhimento = st.checkbox("Senti acolhimento com a orientação")
            entendeu_situacao = st.checkbox("Entendi melhor a situação vivida")
            pretende_continuar = st.checkbox("Sinto que consigo continuar em STEM")
            buscou_apoio = st.checkbox("Pretendo buscar apoio ou mentoria")

        with col2:
            indicaria_athena = st.checkbox("Indicaria a Athena para outra mulher")
            sentiu_exposicao = st.checkbox("Senti exposição ou desconforto")
            achou_inutil = st.checkbox("A orientação não ajudou")
            pretende_desistir = st.checkbox("Ainda penso em desistir ou me afastar")

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

        if st.button("Enviar avaliação", use_container_width=True):
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

            st.success("Obrigada. Sua avaliação ajuda a Athena a oferecer apoios cada vez mais adequados.")


# =========================
# ABA 3 — EVOLUÇÃO
# =========================

with tab3:
    st.subheader("📈 Evolução das orientações")

    st.markdown("""
    Este painel mostra como a Athena registra as avaliações e melhora suas sugestões ao longo do tempo.
    """)

    if st.session_state.historico:
        df_hist = pd.DataFrame(st.session_state.historico)

        st.dataframe(
            df_hist[
                [
                    "Data/Hora",
                    "Estado",
                    "Intervenção",
                    "Avaliação da usuária",
                    "Impacto percebido",
                    "Recompensa total"
                ]
            ],
            use_container_width=True
        )

        st.markdown("### Evolução da recompensa total")
        st.line_chart(df_hist["Recompensa total"])
    else:
        st.info("Ainda não há avaliações registradas.")


# =========================
# ABA 4 — RECURSOS
# =========================

with tab4:
    st.subheader("📚 Recursos de apoio")

    st.markdown("""
    A Athena pode apoiar mulheres em STEM com diferentes caminhos:

    - **Acolhimento individual:** reconhecer a experiência vivida e reduzir o isolamento.
    - **Resposta sugerida:** ajudar a formular uma resposta cuidadosa e assertiva.
    - **Material educativo:** explicar por que certos comportamentos prejudicam a permanência feminina.
    - **Mentoria:** incentivar contato com pessoas de referência.
    - **Grupo de apoio:** fortalecer redes de pertencimento.
    - **Encaminhamento institucional:** orientar busca por canais formais quando necessário.
    """)

    st.info(
        "Em situações graves, como assédio, ameaça ou violência, a Athena deve ser vista "
        "como ferramenta complementar. O encaminhamento institucional e a rede de proteção "
        "devem ser priorizados."
    )


# =========================
# ABA 5 — PAINEL ACADÊMICO
# =========================

with tab5:
    st.subheader("🔬 Painel Acadêmico — Aprendizado por Reforço")

    st.markdown("""
    Esta aba explica o funcionamento técnico do projeto para fins acadêmicos.

    A Athena não utiliza Aprendizado por Reforço para classificar textos.
    A usuária informa a situação que mais se aproxima de sua experiência.

    O Aprendizado por Reforço ocorre na escolha da intervenção mais adequada.
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
        "No Athena": [
            "Sistema Athena",
            "Ambiente de apoio à permanência feminina em STEM",
            "Categoria informada pela usuária",
            "Intervenção sugerida pelo sistema",
            "Avaliação da usuária + impacto percebido",
            "Epsilon-greedy",
            "Um relato, uma intervenção e uma avaliação",
            "Atualização da Q-Table por Q-Learning"
        ]
    })

    st.table(df_rl)

    st.markdown("### Q-Table")

    st.dataframe(st.session_state.q_table, use_container_width=True)

    estado_visualizar = st.selectbox(
        "Visualizar valores Q para o estado:",
        STATES
    )

    st.bar_chart(st.session_state.q_table.loc[estado_visualizar])

    melhor_acao = st.session_state.q_table.loc[estado_visualizar].idxmax()
    melhor_valor = st.session_state.q_table.loc[estado_visualizar].max()

    st.info(
        f"Para **{estado_visualizar}**, a intervenção atualmente preferida é "
        f"**{melhor_acao}**, com Q = **{melhor_valor:.4f}**."
    )

    st.markdown("### Fórmula de Q-Learning")

    st.latex(r"""
    Q(s,a) = Q(s,a) + \alpha \cdot [r + \gamma \cdot \max Q(s',a') - Q(s,a)]
    """)

    st.markdown("### Recompensa total")

    st.latex(r"""
    R_{total} = R_{usuaria} + R_{impacto}
    """)

    if st.session_state.historico:
        st.markdown("### Histórico completo dos episódios")
        st.dataframe(pd.DataFrame(st.session_state.historico), use_container_width=True)
