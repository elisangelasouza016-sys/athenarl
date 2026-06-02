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
    # Inicializa com valores estáveis entre 0.5 e 1.0
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
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        comentario = st.text_area(
            "Insira a mensagem do chat ou fórum para auditar:", 
            height=140,
            placeholder="Cole aqui um exemplo (Ex: 'Seu código está amador, deixe que eu assumo o backend.')"
        )
        
        # Mapeamento linguístico de estados para a simulação
        if comentario:
            texto_low = comentario.lower()
            if any(word in texto_low for word in ["sair comigo", "contrato", "encontrar", "assédio"]):
                estado_atual = "Sexual Harassment"
            elif any(word in texto_low for word in ["ameaça", "bater", "processar", "ferrar"]):
                estado_atual = "Threats"
            elif any(word in texto_low for word in ["batom", "bonita", "foto", "linda", "aparência"]):
                estado_atual = "Physical Appearance"
            elif any(word in texto_low for word in ["mulher", "homem", "menina", "geralmente"]):
                estado_atual = "Stereotyping (Estereótipos)"
            elif any(word in texto_low for word in ["amador", "amadora", "não entende", "incompetente"]):
                estado_atual = "Discredit (Descrédito)"
            elif any(word in texto_low for word in ["ignore", "cale", "assumo", "meu jeito", "apenas aceite"]):
                estado_atual = "Dominance (Dominação)"
            else:
                estado_atual = "Neutral (Saudável)"

    with col_output:
        if comentario:
            st.markdown(f"📊 **Categoria Identificada:** <span style='color:#6B5B95; font-weight:bold; font-size:1.1rem;'>{estado_atual}</span>", unsafe_allow_html=True)
            
            # --- CAMADA SAFE RL: Hard Constraints para Casos Críticos ---
            if estado_atual in ["Sexual Harassment", "Threats"]:
                melhor_acao = "Reportar p/ Governança Institucional"
                
                st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#FFECEA; border-left:6px solid #FF4B4B; color:#C0392B; margin-top:10px;'>🚨 <b>PROTOCOLO DE EMERGÊNCIA ATIVADO:</b> {melhor_acao}</div>", unsafe_allow_html=True)
                st.write("")
                st.warning("⚠️ **Bloqueio de Segurança:** Devido à gravidade legal e ética desta infração, o caso foi congelado e enviado imediatamente para a Ouvidoria e Comissões de Direitos Humanos. O feedback público foi desativado nesta instância para evitar manipulações do algoritmo.")
                
            else:
                # Fluxo normal de Aprendizado por Reforço para as demais categorias
                melhor_acao = st.session_state.q_table.loc[estado_atual].idxmax()
                valor_q = st.session_state.q_table.loc[estado_atual].max()
                
                if estado_atual == "Neutral (Saudável)":
                    st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#EAF2F8; border-left:6px solid #6B5B95; color:#2980B9; margin-top:10px;'>✅ <b>Ação Recomendada pela IA:</b> {melhor_acao}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#FFF5EC; border-left:6px solid #FF6F61; color:#D35400; margin-top:10px;'>⚠️ <b>Ação Recomendada pela IA:</b> {melhor_acao}</div>", unsafe_allow_html=True)
                
                # Interface de Feedback Ativo (RLHF)
                st.write("")
                st.markdown("### 🗣️ O feedback da comunidade valida esta ação?")
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("👍 Sim, ação justa e protetiva", use_container_width=True):
                        st.session_state.q_table.loc[estado_atual, melhor_acao] += 0.4
                        st.success("Recompensa (+0.4) aplicada à Q-Table! O agente reforçou esta política.")
                        st.balloons()
                
                with col_btn2:
                    if st.button("👎 Não, ação inadequada ou ineficiente", use_container_width=True):
                        st.session_state.q_table.loc[estado_atual, melhor_acao] -= 0.6
                        st.error("Punição (-0.6) aplicada! O peso desta ação caiu. Dê um espaço no texto para atualizar e ver a nova escolha do agente.")

with tab2:
    st.subheader("Visualização da Q-Table em Tempo Real")
    st.write("Abaixo é possível auditar como os pesos da matriz de utilidade mudam instantaneamente após as interações:")
    
    if 'estado_atual' in locals():
        cat_visualizar = estado_atual
        st.info(f"Visualizando a 'mente' do agente para o estado sob análise: **{cat_visualizar}**")
    else:
        cat_visualizar = st.selectbox("Selecione uma categoria da taxonomia para visualizar a estratégia do agente:", CATEGORIES)
    
    # Renderização dinâmica do gráfico de barras da Q-Table
    st.bar_chart(st.session_state.q_table.loc[cat_visualizar], color="#FF6F61")
    st.caption("A ação com a barra mais alta é a escolhida de forma autônoma pela IA. Punições (cliques em 'Não') reduzem a barra ao vivo.")

with tab3:
    st.subheader("Taxonomia Científica Base: Beyond Binary Moderation (2025)")
    st.markdown("O sistema categoriza nuances sociolinguísticas que impactam diretamente a permanência feminina em STEM:")
    
    df_taxonomia = pd.DataFrame({
        "Categoria de Auditoria": CATEGORIES,
        "Indicadores Comportamentais (Foco em STEM)": [
            "Atacar ou desqualificar a competência técnica sem fundamentos sólidos.",
            "Generalizações de gênero que limitam a atuação feminina a papéis secundários.",
            "Tentativas de silenciamento, interrupções ou controle impositivo do fluxo de trabalho.",
            "Ignorar propositalmente contribuições válidas em repositórios ou debates.",
            "Comentários, insinuações ou intimidações de cunho sexual.",
            "Ameaças explícitas ou implícitas direcionadas à integridade ou carreira.",
            "Insultos ou cobranças baseadas em papéis reprodutivos e familiares.",
            "Tratar a profissional como elemento visual ou objeto, ignorando sua entrega técnica.",
            "Manifestações de preconceito contra a diversidade e identidades sexuais.",
            "Desviar o foco da discussão técnica para avaliar atributos de aparência física.",
            "Julgamentos morais ou policiamento sobre a conduta e vida pessoal fora do trabalho.",
            "Comunicação técnica sadia, colaborativa, profissional e inclusiva."
        ]
    })
    st.table(df_taxonomia)

# Rodapé
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8E44AD;'><b>Linha de Pesquisa Athena</b> — Tecnologia para a Permanência e Salvaguarda da Mulher em STEM.</p>", unsafe_allow_html=True)
