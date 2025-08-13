import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o aplicativo Streamlit
st.title("Aplicativo Interativo de Aprendizado de Estatística")
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha um Tópico", [
    "Tipos de Variáveis", 
    "Tipos de Dados", 
    "Medidas de Tendência Central", 
    "Medidas de Dispersão"
])

# Página 1: Tipos de Variáveis
if page == "Tipos de Variáveis":
    st.header("Tipos de Variáveis")
    st.write("""
    Variáveis são características ou atributos que podem ser medidos ou observados. 
    Elas são classificadas em três tipos principais:
    """)
    
    st.subheader("1. Variáveis Contínuas")
    st.write("""
    - **Definição**: Variáveis que podem assumir qualquer valor dentro de um intervalo.
    - **Exemplos**: Altura, peso, temperatura.
    - **Visualização**: Histograma ou gráfico de linha.
    """)
    st.write("**Exemplo Interativo**")
    mean = st.slider("Média da distribuição", 0.0, 100.0, 50.0)
    std_dev = st.slider("Desvio Padrão", 0.1, 20.0, 10.0)
    continuous_data = np.random.normal(mean, std_dev, 10000)
    st.write(pd.Series(continuous_data).describe())
    fig, ax = plt.subplots()
    sns.histplot(continuous_data, kde=True, ax=ax)
    ax.set_xlabel("Valores")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

    st.subheader("2. Variáveis Categóricas")
    st.write("""
    - **Definição**: Variáveis que representam categorias ou grupos.
    - **Exemplos**: Gênero, cor, tipo de carro.
    - **Visualização**: Gráfico de barras ou pizza.
    """)
    st.write("**Exemplo Interativo**")
    categories = st.text_input("Digite categorias (separadas por vírgula)", "Vermelho,Azul,Verde")
    categories = [cat.strip() for cat in categories.split(",")]
    categorical_data = np.random.choice(categories, 100)
    st.write(pd.Series(categorical_data).value_counts())
    
    # Definir cores para as categorias
    color_map = {
        'Vermelho': 'red',
        'Azul': 'blue', 
        'Verde': 'green',
        'Amarelo': 'yellow',
        'Roxo': 'purple',
        'Laranja': 'orange',
        'Rosa': 'pink',
        'Marrom': 'brown',
        'Cinza': 'gray',
        'Preto': 'black'
    }
    
    fig, ax = plt.subplots()
    unique_categories = pd.Series(categorical_data).value_counts().index
    colors = [color_map.get(cat.lower().capitalize(), 'steelblue') for cat in unique_categories]
    
    sns.countplot(x=categorical_data, order=unique_categories, palette=colors, ax=ax)
    ax.set_xlabel("Categorias")
    ax.set_ylabel("Contagem")
    st.pyplot(fig)

    st.subheader("3. Variáveis Ordinais")
    st.write("""
    - **Definição**: Variáveis categóricas com uma ordenação ou classificação clara.
    - **Exemplos**: Nível de educação (Ensino Médio, Graduação, Mestrado, Doutorado), 
    classificação de satisfação (Baixo, Médio, Alto).
    - **Visualização**: Gráfico de barras ordenado.
    """)
    st.write("**Exemplo Interativo**")
    levels = st.text_input("Digite níveis ordinais (separados por vírgula)", "Baixo,Médio,Alto")
    levels = [level.strip() for level in levels.split(",")]
    probabilities = st.text_input("Digite probabilidades para cada nível (separadas por vírgula)", "0.2,0.5,0.3")
    probabilities = [float(p.strip()) for p in probabilities.split(",")]
    ordinal_data = np.random.choice(levels, 100, p=probabilities)
    st.write(pd.Series(ordinal_data).value_counts())
    
    # Definir cores diferentes para níveis ordinais
    ordinal_colors = {
        'Baixo': '#ff6b6b',      # Vermelho claro
        'Médio': '#4ecdc4',      # Verde azulado
        'Alto': '#45b7d1',       # Azul claro
        'Muito Baixo': '#ff9ff3', # Rosa
        'Muito Alto': '#96ceb4',  # Verde claro
        'Extremo': '#feca57'      # Amarelo
    }
    
    fig, ax = plt.subplots()
    colors = [ordinal_colors.get(level, '#steelblue') for level in levels]
    
    sns.countplot(x=ordinal_data, order=levels, palette=colors, ax=ax)
    ax.set_xlabel("Níveis")
    ax.set_ylabel("Contagem")
    st.pyplot(fig)

# Página 2: Tipos de Dados
elif page == "Tipos de Dados":
    st.header("Tipos de Dados")
    st.write("""
    Os dados podem ser classificados em três tipos principais baseados em como são coletados:
    """)
    
    st.subheader("1. Dados Transversais (Cross-Sectional)")
    st.write("""
    - **Definição**: Dados coletados em um único ponto no tempo.
    - **Exemplos**: Dados de pesquisa, dados de censo.
    - **Visualização**: Gráfico de dispersão ou barras.
    """)
    st.write("**Exemplo Interativo**")
    num_points = st.slider("Número de pontos de dados", 10, 1000, 100)
    cross_sectional_data = pd.DataFrame({
        "Idade": np.random.randint(18, 65, num_points),
        "Renda": np.random.randint(20000, 100000, num_points)
    })
    st.write(cross_sectional_data.head())
    fig, ax = plt.subplots()
    sns.scatterplot(x="Idade", y="Renda", data=cross_sectional_data, ax=ax)
    st.pyplot(fig)

    st.subheader("2. Dados de Séries Temporais")
    st.write("""
    - **Definição**: Dados coletados ao longo do tempo.
    - **Exemplos**: Preços de ações, leituras de temperatura.
    - **Visualização**: Gráfico de linha.
    """)
    st.write("**Exemplo Interativo**")
    start_date = st.date_input("Data de início", pd.to_datetime("2023-01-01"))
    num_days = st.slider("Número de dias", 10, 365, 100)
    time_series_data = pd.DataFrame({
        "Data": pd.date_range(start=start_date, periods=num_days, freq="D"),
        "Preço": np.cumsum(np.random.randn(num_days)) + 100
    })
    st.write(time_series_data.head())
    fig, ax = plt.subplots()
    sns.lineplot(x="Data", y="Preço", data=time_series_data, ax=ax)
    st.pyplot(fig)

    st.subheader("3. Dados em Painel")
    st.write("""
    - **Definição**: Dados coletados ao longo do tempo para múltiplas entidades.
    - **Exemplos**: PIB de países ao longo dos anos, dados de vendas para múltiplas lojas.
    - **Visualização**: Gráfico de linhas facetado ou mapa de calor.
    """)
    st.write("**Exemplo Interativo**")
    num_entities = st.slider("Número de entidades", 2, 10, 3)
    num_days_panel = st.slider("Número de dias (painel)", 10, 365, 100)
    panel_data = pd.DataFrame({
        "País": np.repeat([f"País {i+1}" for i in range(num_entities)], num_days_panel),
        "Data": pd.date_range(start="2023-01-01", periods=num_days_panel, freq="D").tolist() * num_entities,
        "PIB": np.random.randn(num_entities * num_days_panel).cumsum() + 100
    })
    st.write(panel_data.head())
    fig, ax = plt.subplots()
    sns.lineplot(x="Data", y="PIB", hue="País", data=panel_data, ax=ax)
    st.pyplot(fig)

# Página 3: Medidas de Tendência Central
elif page == "Medidas de Tendência Central":
    st.header("Medidas de Tendência Central")
    st.write("""
    As medidas de tendência central descrevem o centro ou valor típico de um conjunto de dados.
    """)
    
    st.subheader("Exemplo Interativo")
    data_input = st.text_input("Digite uma lista de números (separados por vírgula)", "10,20,30,40,50,60,70,80,90,10")
    data = [float(x.strip()) for x in data_input.split(",")]
    st.write("**Dados**:", data)

    st.subheader("1. Média")
    st.write(f"Média: {np.mean(data):.2f}")
    st.write("""
    A média é a soma de todos os valores dividida pelo número de observações. 
    É sensível a valores extremos (outliers).
    """)

    st.subheader("2. Mediana")
    st.write(f"Mediana: {np.median(data):.2f}")
    st.write("""
    A mediana é o valor que divide o conjunto de dados ao meio quando ordenado. 
    É mais resistente a outliers que a média.
    """)

    st.subheader("3. Moda")
    mode_result = pd.Series(data).mode()
    if not mode_result.empty:
        st.write(f"Moda: {mode_result[0]}")
    else:
        st.write("Não há moda única encontrada.")
    st.write("""
    A moda é o valor que aparece com maior frequência no conjunto de dados.
    """)

    # Novo tópico: Quarteto de Anscombe
    st.header("O Quarteto de Anscombe")
    st.write("""
    O Quarteto de Anscombe é um conjunto famoso de quatro conjuntos de dados que demonstra 
    a importância da visualização de dados. Todos os quatro conjuntos têm praticamente 
    as mesmas propriedades estatísticas básicas, mas são muito diferentes quando visualizados.
    """)
    
    st.subheader("Demonstração Interativa")
    
    # Dados do Quarteto de Anscombe
    anscombe_data = {
        'Conjunto 1': {
            'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        },
        'Conjunto 2': {
            'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
        },
        'Conjunto 3': {
            'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
        },
        'Conjunto 4': {
            'x': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
        }
    }
    
    # Escolha do conjunto
    selected_set = st.selectbox("Escolha um conjunto do Quarteto de Anscombe:", 
                               list(anscombe_data.keys()))
    
    x_data = anscombe_data[selected_set]['x']
    y_data = anscombe_data[selected_set]['y']
    
    # Estatísticas
    st.write("**Estatísticas do conjunto selecionado:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Média de X: {np.mean(x_data):.2f}")
        st.write(f"Média de Y: {np.mean(y_data):.2f}")
        st.write(f"Desvio padrão de X: {np.std(x_data, ddof=1):.2f}")
        st.write(f"Desvio padrão de Y: {np.std(y_data, ddof=1):.2f}")
    
    with col2:
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        st.write(f"Correlação: {correlation:.3f}")
        
        # Regressão linear simples
        slope, intercept = np.polyfit(x_data, y_data, 1)
        st.write(f"Equação da reta: y = {slope:.2f}x + {intercept:.2f}")
    
    # Visualização
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_data, y_data, s=80, alpha=0.7)
    
    # Linha de regressão
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', alpha=0.8, label=f'y = {slope:.2f}x + {intercept:.2f}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{selected_set} - Quarteto de Anscombe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Comparação de todos os conjuntos
    if st.checkbox("Mostrar todos os conjuntos simultaneamente"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, data) in enumerate(anscombe_data.items()):
            ax = axes[i]
            x_vals = data['x']
            y_vals = data['y']
            
            ax.scatter(x_vals, y_vals, s=60, alpha=0.7)
            
            # Linha de regressão
            slope_i, intercept_i = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            y_line = slope_i * x_line + intercept_i
            ax.plot(x_line, y_line, 'r--', alpha=0.8)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(2, 20)
            ax.set_ylim(2, 14)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        **Conclusão**: Apesar de terem estatísticas quase idênticas (média, desvio padrão, 
        correlação e equação de regressão), os quatro conjuntos mostram padrões completamente 
        diferentes quando visualizados. Isso demonstra a importância crucial da visualização 
        de dados na análise estatística!
        """)

# Página 4: Medidas de Dispersão
elif page == "Medidas de Dispersão":
    st.header("Medidas de Dispersão")
    st.write("""
    As medidas de dispersão descrevem o quão espalhados os dados estão.
    """)
    
    st.subheader("Exemplo Interativo")
    data_input = st.text_input("Digite uma lista de números (separados por vírgula)", "10,20,30,40,50,60,70,80,90,10")
    data = [float(x.strip()) for x in data_input.split(",")]
    st.write("**Dados**:", data)

    st.subheader("1. Variância")
    st.write(f"Variância: {np.var(data, ddof=1):.2f}")
    st.write("""
    A variância mede a dispersão dos dados em relação à média. 
    É calculada como a média dos quadrados das diferenças em relação à média.
    """)

    st.subheader("2. Desvio Padrão")
    st.write(f"Desvio Padrão: {np.std(data, ddof=1):.2f}")
    st.write("""
    O desvio padrão é a raiz quadrada da variância. 
    É expresso na mesma unidade dos dados originais.
    """)

    st.subheader("3. Covariância")
    st.write("Digite outra lista de números para calcular a covariância:")
    data_input2 = st.text_input("Segunda lista (separada por vírgula)", "15,25,35,45,55,65,75,85,95,15")
    data2 = [float(x.strip()) for x in data_input2.split(",")]
    if len(data) == len(data2):
        st.write(f"Covariância: {np.cov(data, data2)[0, 1]:.2f}")
        st.write("""
        A covariância mede como duas variáveis variam juntas. 
        Valores positivos indicam que as variáveis tendem a aumentar juntas.
        """)
        fig, ax = plt.subplots()
        sns.scatterplot(x=data, y=data2, ax=ax)
        ax.set_xlabel("Primeira variável")
        ax.set_ylabel("Segunda variável")
        st.pyplot(fig)
    else:
        st.write("Ambas as listas devem ter o mesmo comprimento para calcular a covariância.")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.write("Criado por José Américo")