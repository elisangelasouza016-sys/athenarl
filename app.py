import streamlit as st
import pandas as pd
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Sistema Athena - RLHF", layout="wide", page_icon="🛡️")

# --- Customização de Cores via CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #F4EFF7; }
    .athena-title { color: #6B5B95; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; margin-bottom: 5px; }
    .athena-subtitle { color: #FF6F61; font-size: 1.1rem; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# --- DATABASE DE CONHECIMENTO ---
CATEGORIES = [
    "Discredit (Descrédito)", "Stereotyping (Estereótipos)", "Dominance (Dominação)", 
    "Dismissing (Desprezo)", "Sexual Harassment", "Threats", "Maternal Insults", 
    "Objectification", "Anti-LGBTQ+", "Physical Appearance", 
    "Moral Condemnation", "Neutral (Saudável)"
]

ACTIONS = [
    "Silêncio Operacional", "Sugerir Reescrita Técnico-Pedagógica", 
    "Alerta de Viés Social", "Intervenção Educativa Coletiva",
    "Mediação Direta (Humano)", "Reportar p/ Governança Institucional"
]

# --- INICIALIZAR A Q-TABLE NA MEMÓRIA DO SISTEMA (Session State) ---
# Isso garante que quando clicarmos nos botões, a IA realmente altere seus valores e aprenda
if "q_table" not in st.session_state:
    np.random.seed(42)
    q_data = np.random.uniform(0.5, 1.0, size=(len(CATEGORIES), len(ACTIONS)))
    st.session_state.q_table = pd.DataFrame(q_data, index=CATEGORIES, columns=ACTIONS)
    
    # Valores base iniciais para a demonstração
    st.session_state.q_table.loc["Sexual Harassment", "Reportar p/ Governança Institucional"] = 1.6
    st.session_state.q_table.loc["Discredit (Descrédito)", "Sugerir Reescrita Técnico-Pedagógica"] = 1.4
    st.session_state.q_table.loc["Neutral (Saudável)", "Silêncio Operacional"] = 1.8

# --- INTERFACE SIDEBAR ---
st.sidebar.markdown("<h2 style='color: #6B5B95; margin-bottom: 0;'>🛡️ Linha Athena</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #FF6F61; font-weight: bold; margin-top: 0;'>Escudo de Proteção em STEM</p>", unsafe_allow_html=True)
st.sidebar.info("Modo Atual: Aprendizado por Reforço com Feedback Humano (RLHF).")

if st.sidebar.button("🔄 Reiniciar Aprendizado da IA"):
    del st.session_state.q_table
    st.rerun()

# --- CONTEÚDO PRINCIPAL ---
st.markdown("<h1 class='athena-title'>🛡️ Painel Athena: Governança com Feedback Humano (RLHF)</h1>", unsafe_allow_html=True)
st.markdown("<p class='athena-subtitle'>O agente de IA aprende e adapta suas ações de mediação de acordo com o julgamento das usuárias.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Auditoria & Feedback Ativo", "📊 Matriz de Aprendizado (Q-Table)", "📚 Base Científica"])

with tab1:
    st.subheader("Análise de Interações e Validação da Usuária")
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        comentario = st.text_area(
            "Insira a mensagem do fórum para auditar:", 
            height=120,
            placeholder="Cole um exemplo de teste aqui..."
        )
        # Armazena o comentário para processamento estável
        if comentario:
            texto_low = comentario.lower()
            if any(word in texto_low for word in ["sair comigo", "contrato", "encontrar"]): estado_atual = "Sexual Harassment"
            elif any(word in texto_low for word in ["batom", "bonita", "foto", "linda"]): estado_atual = "Physical Appearance"
            elif any(word in texto_low for word in ["mulher", "homem", "menina", "geralmente"]): estado_atual = "Stereotyping (Estereótipos)"
            elif any(word in texto_low for word in ["amador", "amadora", "não entende"]): estado_atual = "Discredit (Descrédito)"
            elif any(word in texto_low for word in ["ignore", "cale", "assumo", "meu jeito"]): estado_atual = "Dominance (Dominação)"
            else: estado_atual = "Neutral (Saudável)"
            
            # Escolhe a melhor ação baseada na Q-Table atualizada na memória
            melhor_acao = st.session_state.q_table.loc[estado_atual].idxmax()
            valor_q = st.session_state.q_table.loc[estado_atual].max()

    with col_output:
        if comentario:
            st.markdown(f"📊 **Categoria Identificada:** <span style='color:#6B5B95; font-weight:bold;'>{estado_atual}</span>", unsafe_allow_html=True)
            
            if estado_atual == "Neutral (Saudável)":
                st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#EAF2F8; border-left:6px solid #6B5B95; color:#2980B9;'>✅ <b>Ação Atual do Agente:</b> {melhor_acao}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#FFECEA; border-left:6px solid #FF6F61; color:#C0392B;'>⚠️ <b>Ação Atual do Agente:</b> {melhor_acao}</div>", unsafe_allow_html=True)
            
            # --- SISTEMA DE RECOMPENSA VIA BOTÕES (O REFORÇO VIVO) ---
            st.write("")
            st.markdown("### 🗣️ A usuária concorda com a ação da IA?")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("👍 Sim, foi justa e protegeu o ambiente", use_container_width=True):
                    # Recompensa Positiva: aumenta o valor daquela ação para aquele estado
                    st.session_state.q_table.loc[estado_atual, melhor_acao] += 0.4
                    st.success("Recompensa (+0.4) enviada ao agente! Ele aprendeu que tomou a decisão certa.")
                    st.balloons()
            
            with col_btn2:
                if st.button("👎 Não, ação inadequada / injusta", use_container_width=True):
                    # Punição (Recompensa Negativa): reduz drasticamente o valor dessa ação
                    st.session_state.q_table.loc[estado_atual, melhor_acao] -= 0.6
                    st.error("Punição (-0.6) enviada ao agente! O peso dessa ação caiu. Teste novamente para ver a nova escolha.")

with tab2:
    st.subheader("Gráfico da Q-Table em Tempo Real")
    st.write("Veja como os pesos mudam instantaneamente quando as usuárias interagem através dos botões:")
    
    if 'estado_atual' in locals():
        cat_visualizar = estado_atual
        st.info(f"Visualizando a mente da IA para o estado atual: **{cat_visualizar}**")
    else:
        cat_visualizar = st.selectbox("Escolha uma categoria para auditar os pesos da IA:", CATEGORIES)
    
    st.bar_chart(st.session_state.q_table.loc[cat_visualizar], color="#FF6F61")
    st.caption("A ação com a maior barra será a escolhida pela IA. Se você clicar em 'Não', a barra correspondente vai diminuir ao vivo!")

with tab3:
    st.subheader("Taxonomia Científica Base: Beyond Binary Moderation (2025)")
    df_taxonomia = pd.DataFrame({"Categoria de Auditoria": CATEGORIES, "Descrição": ["Foco em STEM e comportamentos de exclusão..."] * 12})
    st.table(df_taxonomia)
