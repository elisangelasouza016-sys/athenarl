import streamlit as st
import pandas as pd
import numpy as np

# --- Configuração da Página com o Escudo ---
st.set_page_config(
    page_title="Sistema Athena - RLHF & Safe RL", 
    layout="wide", 
    page_icon="🛡️"
)

# --- Customização de Cores via CSS (Paleta Lilás, Rosa e Coral) ---
st.markdown("""
    <style>
    /* Customização da barra lateral (Tom lilás bem suave) */
    [data-testid="stSidebar"] {
        background-color: #F4EFF7;
    }
    /* Estilização dos títulos principais */
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
    /* Estilização das abas */
    .stTabs [data-baseweb="tab"] {
        color: #6B5B95 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F4EFF7 !important;
        border-radius: 5px 5px 0px 0px;
        border-bottom: 3px solid #FF6F61 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATABASE DE CONHECIMENTO (Taxonomia do Artigo 2025) ---
CATEGORIES = [
    "Discredit (Descrédito)", "Stereotyping (Estereótipos)", "Dominance (Dominação)", 
    "Dismissing (Desprezo)", "Sexual Harassment", "Threats", "Maternal Insults", 
    "Objectification", "Anti-LGBTQ+", "Physical Appearance", 
    "Moral Condemnation", "Neutral (Saudável)"
]

ACTIONS = [
    "Silêncio Operacional", 
    "Sugerir Reescrita Técnico-Pedagógica", 
    "Alerta de Viés Social", 
    "Intervenção Educativa Coletiva",
    "Mediação Direta (Humano)",
    "Reportar p/ Governança Institucional"
]

# --- INICIALIZAR A Q-TABLE NA MEMÓRIA DO SISTEMA (Session State) ---
if "q_table" not in st.session_state:
    np.random.seed(42)
    q_data = np.random.uniform(0.5, 1.0, size=(len(CATEGORIES), len(ACTIONS)))
    st.session_state.q_table = pd.DataFrame(q_data, index=CATEGORIES, columns=ACTIONS)
    
    # Ajustes finos éticos iniciais para a demonstração
    st.session_state.q_table.loc["Discredit (Descrédito)", "Sugerir Reescrita Técnico-Pedagógica"] = 1.5
    st.session_state.q_table.loc["Stereotyping (Estereótipos)", "Alerta de Viés Social"] = 1.4
    st.session_state.q_table.loc["Neutral (Saudável)", "Silêncio Operacional"] = 1.8

# --- INTERFACE SIDEBAR ---
st.sidebar.markdown("<h2 style='color: #6B5B95; margin-bottom: 0;'>🛡️ Linha Athena</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #FF6F61; font-weight: bold; margin-top: 0;'>Escudo de Proteção em STEM</p>", unsafe_allow_html=True)
st.sidebar.info("Modo Atual: RLHF com Restrições Rígidas de Segurança (Safe RL).")

st.sidebar.markdown("---")
st.sidebar.markdown("<b style='color: #6B5B95;'>Status do Ambiente</b>", unsafe_allow_html=True)
st.sidebar.write("🟢 Monitoramento Ativo")
st.sidebar.write("🔒 Protocolo de Proteção: Ativado")

if st.sidebar.button("🔄 Reiniciar Memória do Agente"):
    del st.session_state.q_table
    st.rerun()

# --- CONTEÚDO PRINCIPAL ---
st.markdown("<h1 class='athena-title'>🛡️ Painel Athena: Governança com Feedback Humano & Safe RL</h1>", unsafe_allow_html=True)
st.markdown("<p class='athena-subtitle'>Auditoria algorítmica viva que combina aprendizado dinâmico com travas éticas inegociáveis para proteção de gênero.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Auditoria & Feedback Ativo", "📊 Matriz de Aprendizado (Q-Table)", "📚 Base Científica"])

with tab1:
    st.subheader("Análise de Interações e Validação da Usuária")
    col
