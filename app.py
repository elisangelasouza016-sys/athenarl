import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Athena-RL | Q-Learning com Métrica de Impacto",
    layout="wide",
    page_icon="🛡️"
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #F4EFF7; }
.athena-title {
    color: #6B5B95;
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: bold;
}
.athena-subtitle {
    color: #FF6F61;
    font-size: 1.1rem;
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
# ESTADOS E AÇÕES
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

DEFAULT_ALPHA = 0.30
DEFAULT_GAMMA = 0.80
DEFAULT_EPSILON = 0.20

# ============================================================
# FUNÇÕES
# ============================================================

def inicializar_q_table():
    np.random.seed(42)
    q_data = np.random.uniform(
        low=0.1,
        high=0.5,
        size=(len(CATEGORIES), len(ACTIONS))
    )

    q_table = pd.DataFrame(q_data, index=CATEGORIES, columns=ACTIONS)

    q_table.loc["Neutral (Saudável)", "Silêncio Operacional"] = 1.20
    q_table.loc["Discredit (Descrédito)", "Sugerir Reescrita Técnico-Pedagógica"] = 1.00
    q_table.loc["Stereotyping (Estereótipos)", "Alerta de Viés Social"] = 1.00
    q_table.loc["Dominance (Dominação)", "Mediação Direta (Humano)"] = 1.00
    q_table.loc["Sexual Harassment", "Reportar p/ Governança Institucional"] = 1.30
    q_table.loc["Threats", "Reportar p/ Governança Institucional"] = 1.40

    return q_table


def classificar_estado(texto):
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

    if any(p in texto for p in ["gostosa", "delícia", "objeto"]):
        return "Objectification"

    if any(p in texto for p in ["lgbt", "gay", "lésbica", "trans"]):
        return "Anti-LGBTQ+"

    if any(p in texto for p in ["imoral", "sem valor", "vergonha", "não presta"]):
        return "Moral Condemnation"

    return "Neutral (Saudável)"


def escolher_acao(estado, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS), "Exploração"
    return q_table.loc[estado].idxmax(), "Explotação"


def calcular_impacto(
    comentario_editado=False,
    discussao_saudavel=False,
    pr_aceito=False,
    nova_denuncia=False,
    reincidencia=False,
    abandono_discussao=False,
    pessoa_afetada_continua=False,
    novo_pr_pessoa_afetada=False
):
    impacto = 0

    if comentario_editado:
        impacto += 2

    if discussao_saudavel:
        impacto += 1

    if pr_aceito:
        impacto += 2

    if pessoa_afetada_continua:
        impacto += 3

    if novo_pr_pessoa_afetada:
        impacto += 5

    if nova_denuncia:
        impacto -= 3

    if reincidencia:
        impacto -= 2

    if abandono_discussao:
        impacto -= 5

    return impacto


def atualizar_q_learning(q_table, estado, acao, recompensa_total, proximo_estado, alpha, gamma):
    q_antigo = q_table.loc[estado, acao]
    melhor_q_futuro = q_table.loc[proximo_estado].max()

    q_novo = q_antigo + alpha * (
        recompensa_total + gamma * melhor_q_futuro - q_antigo
    )

    q_table.loc[estado, acao] = q_novo

    return q_antigo, q_novo, melhor_q_futuro


def simular_proximo_estado():
    return np.random.choice(CATEGORIES)


def registrar_episodio(
    texto,
    estado,
    acao,
    modo,
    feedback_humano,
    impacto,
    recompensa_total,
    q_antigo,
    q_novo,
    proximo_estado
):
    st.session_state.historico.append({
        "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Texto": texto,
        "Estado": estado,
        "Ação": acao,
        "Política": modo,
        "Feedback humano": feedback_humano,
        "Impacto observado": impacto,
        "Recompensa total": recompensa_total,
        "Q antigo": round(q_antigo, 4),
        "Q novo": round(q_novo, 4),
        "Próximo estado": proximo_estado
    })


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
st.sidebar.markdown("**Q-Learning + Feedback Humano + Métrica de Impacto**")

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
# INTERFACE
# ============================================================

st.markdown(
    "<h1 class='athena-title'>🛡️ Athena-RL: Agente de Mediação com Aprendizado por Reforço</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='athena-subtitle'>"
    "O agente aprende com feedback humano e com métricas reais de impacto no ambiente."
    "</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Auditoria",
    "🎯 Impacto",
    "📊 Q-Table",
    "🧠 Histórico",
    "📚 Explicação"
])

# ============================================================
# ABA 1 — AUDITORIA
# ============================================================

with tab1:
    st.subheader("1. Auditoria da mensagem")

    comentario = st.text_area(
        "Insira uma mensagem simulada de fórum, issue ou pull request:",
        height=140,
        placeholder="Exemplo: Você não entende muito de backend, melhor alguém mais experiente revisar."
    )

    if st.button("Analisar mensagem", use_container_width=True):
        if comentario.strip():
            estado = classificar_estado(comentario)
            acao, modo = escolher_acao(estado, st.session_state.q_table, epsilon)

            st.session_state.ultimo_estado = estado
            st.session_state.ultima_acao = acao
            st.session_state.ultimo_modo = modo
            st.session_state.ultimo_texto = comentario

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

        st.info("Agora vá para a aba **Impacto** para aplicar feedback humano e métricas observáveis.")

# ============================================================
# ABA 2 — IMPACTO
# ============================================================

with tab2:
    st.subheader("2. Feedback humano + Métrica de impacto")

    if not st.session_state.ultimo_estado:
        st.warning("Analise uma mensagem primeiro na aba Auditoria.")
    else:
        st.markdown("### Feedback humano direto")

        feedback_label = st.radio(
            "Como a pessoa avaliadora julgou a ação do Athena?",
            [
                "Excelente (+1.0)",
                "Adequada (+0.5)",
                "Neutra (0.0)",
                "Exagerada (-0.5)",
                "Prejudicial (-1.0)"
            ]
        )

        feedback_map = {
            "Excelente (+1.0)": 1.0,
            "Adequada (+0.5)": 0.5,
            "Neutra (0.0)": 0.0,
            "Exagerada (-0.5)": -0.5,
            "Prejudicial (-1.0)": -1.0
        }

        feedback_humano = feedback_map[feedback_label]

        st.markdown("### Métricas observáveis do ambiente")

        col1, col2 = st.columns(2)

        with col1:
            comentario_editado = st.checkbox("Comentário foi editado/reformulado (+2)")
            discussao_saudavel = st.checkbox("Discussão continuou saudável (+1)")
            pr_aceito = st.checkbox("Pull Request foi aceito (+2)")
            pessoa_afetada_continua = st.checkbox("Pessoa afetada continuou participando (+3)")

        with col2:
            novo_pr_pessoa_afetada = st.checkbox("Pessoa afetada enviou novo PR depois (+5)")
            nova_denuncia = st.checkbox("Houve nova denúncia (-3)")
            reincidencia = st.checkbox("Usuário reincidiu no comportamento (-2)")
            abandono_discussao = st.checkbox("Pessoa afetada abandonou a discussão (-5)")

        impacto = calcular_impacto(
            comentario_editado=comentario_editado,
            discussao_saudavel=discussao_saudavel,
            pr_aceito=pr_aceito,
            nova_denuncia=nova_denuncia,
            reincidencia=reincidencia,
            abandono_discussao=abandono_discussao,
            pessoa_afetada_continua=pessoa_afetada_continua,
            novo_pr_pessoa_afetada=novo_pr_pessoa_afetada
        )

        recompensa_total = feedback_humano + impacto

        st.markdown("### Recompensa calculada")

        st.write(f"Feedback humano: **{feedback_humano}**")
        st.write(f"Impacto observado: **{impacto}**")
        st.write(f"Recompensa total: **{recompensa_total}**")

        if st.button("Aplicar recompensa e atualizar Q-Table", use_container_width=True):
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
                texto=st.session_state.ultimo_texto,
                estado=st.session_state.ultimo_estado,
                acao=st.session_state.ultima_acao,
                modo=st.session_state.ultimo_modo,
                feedback_humano=feedback_humano,
                impacto=impacto,
                recompensa_total=recompensa_total,
                q_antigo=q_antigo,
                q_novo=q_novo,
                proximo_estado=proximo_estado
            )

            st.success(
                f"Q-Table atualizada: Q antigo {q_antigo:.4f} → Q novo {q_novo:.4f}"
            )

# ============================================================
# ABA 3 — Q-TABLE
# ============================================================

with tab3:
    st.subheader("3. Matriz de aprendizado — Q-Table")

    st.dataframe(st.session_state.q_table, use_container_width=True)

    estado_visualizar = st.selectbox(
        "Escolha um estado para visualizar:",
        CATEGORIES
    )

    st.bar_chart(st.session_state.q_table.loc[estado_visualizar])

    melhor_acao = st.session_state.q_table.loc[estado_visualizar].idxmax()
    melhor_valor = st.session_state.q_table.loc[estado_visualizar].max()

    st.info(
        f"Ação preferida para **{estado_visualizar}**: "
        f"**{melhor_acao}** | Q = **{melhor_valor:.4f}**"
    )

# ============================================================
# ABA 4 — HISTÓRICO
# ============================================================

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

# ============================================================
# ABA 5 — EXPLICAÇÃO
# ============================================================

with tab5:
    st.subheader("5. Explicação acadêmica")

    st.markdown("""
    O Athena-RL é uma arquitetura híbrida.

    A primeira camada faz a percepção do texto e transforma a mensagem em um estado.
    Essa etapa pode ser feita por regras, PLN ou modelo supervisionado.

    O Aprendizado por Reforço ocorre na camada de decisão.

    O agente observa o estado, escolhe uma ação, recebe uma recompensa e atualiza sua Q-Table.
    A recompensa não vem apenas do feedback humano direto, mas também de métricas observáveis
    do ambiente, como edição do comentário, continuidade saudável da discussão, reincidência,
    novas denúncias ou abandono da participação.
    """)

    st.markdown("### Elementos do Aprendizado por Reforço")

    df_rl = pd.DataFrame({
        "Elemento": [
            "Agente",
            "Ambiente",
            "Estado",
            "Ação",
            "Recompensa",
            "Política",
            "Aprendizado",
            "Episódio"
        ],
        "No Athena-RL": [
            "Sistema Athena",
            "GitHub, fórum técnico ou comunidade STEM",
            "Categoria da interação textual",
            "Estratégia de mediação",
            "Feedback humano + métrica de impacto",
            "Epsilon-greedy",
            "Atualização da Q-Table",
            "Uma mensagem analisada, uma ação tomada e uma consequência observada"
        ]
    })

    st.table(df_rl)

    st.markdown("### Fórmula de Q-Learning")

    st.latex(r"""
    Q(s,a) = Q(s,a) + \alpha \cdot [r + \gamma \cdot \max Q(s',a') - Q(s,a)]
    """)

    st.markdown("""
    Onde:

    - **s** é o estado atual;
    - **a** é a ação tomada;
    - **r** é a recompensa total;
    - **s'** é o próximo estado;
    - **α** é a taxa de aprendizado;
    - **γ** é o fator de desconto.

    A defesa central é:

    **O Athena não usa Aprendizado por Reforço para classificar o texto.
    Ele usa Aprendizado por Reforço para aprender qual ação de mediação gera melhor impacto
    no ambiente.**
    """)

    st.markdown("### Recompensa total")

    st.latex(r"""
    R_{total} = R_{humano} + R_{impacto}
    """)

    st.markdown("""
    Assim, o sistema não aprende apenas com aprovação ou reprovação humana.
    Ele aprende também com consequências observáveis da comunidade.
    """)
