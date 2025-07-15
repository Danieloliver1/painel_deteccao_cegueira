import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import gdown # Biblioteca para download de arquivos do Google Drive

# A biblioteca 'requests' foi removida pois não é mais necessária
import plotly.express as px

# --- FUNÇÕES DO APLICATIVO ---


@st.cache_resource(show_spinner="Carregando modelo...")
def carrega_modelo():
    #url = 'https://drive.google.com/uc?id=109cgIdRWfFVAWkhT3WWQlrSPQKl2tYjB'
    url = 'https://drive.google.com/uc?id=1pHKPIVO4IJmIFrlAqpgMAv3kbRzdD_f4'
    
    gdown.download(url,'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    
    return interpreter


def carrega_imagem_usuario():
    """
    Cria a interface para o usuário fazer upload de uma imagem de retina
    e a pré-processa para o modelo.
    """
    uploaded_file = st.file_uploader(
        'Arraste e solte um exame de retina ou clique para selecionar',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        # Lê e exibe a imagem
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB') # Garante 3 canais de cor

        st.image(image, caption="Exame de Retina Carregado", use_container_width=True)

        # Pré-processamento da imagem para o modelo
        # Ajuste o 'target_size' para o tamanho que seu modelo espera (ex: 224x224)
        target_size = (224, 224)
        image_resized = image.resize(target_size)
        
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = image_array / 255.0  # Normalização
        image_array = np.expand_dims(image_array, axis=0) # Adiciona dimensão do batch

        return image_array
    return None


def realizar_previsao(interpreter, image):
    """
    Executa a inferência do modelo na imagem e exibe os resultados.
    """
    if interpreter is None:
        st.warning("Modelo não carregado. Não é possível realizar a previsão.")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define o tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], image)

    # Executa a previsão
    interpreter.invoke()

    # Obtém o resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Mapeia as classes de acordo com a competição APTOS 2019
    classes = [
        '0 - Sem Retinopatia',
        '1 - Leve',
        '2 - Moderada',
        '3 - Severa',
        '4 - Proliferativa'
    ]

    # Cria um DataFrame para o gráfico
    df = pd.DataFrame({
        'classes': classes,
        'probabilidades': output_data[0] * 100
    })
    df = df.sort_values(by='probabilidades', ascending=True)

    # Cria o gráfico de barras com Plotly
    fig = px.bar(
        df,
        y='classes',
        x='probabilidades',
        orientation='h',
        text=df['probabilidades'].apply(lambda x: f'{x:.2f}%'),
        title='Diagnóstico de Retinopatia Diabética',
        labels={'classes': 'Nível de Severidade', 'probabilidades': 'Probabilidade (%)'}
    )
    fig.update_layout(
        xaxis_title="Confiança da IA (%)",
        yaxis_title="Diagnóstico Possível",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Exibe o diagnóstico mais provável
    diagnostico_final = df.iloc[-1]
    st.success(f"**Diagnóstico Mais Provável:** {diagnostico_final['classes']} com {diagnostico_final['probabilidades']:.2f}% de confiança.")


def main():
    """
    Função principal que organiza a interface do Streamlit.
    """
    st.set_page_config(
        page_title='Detector de Retinopatia Diabética',
        page_icon='👁️',
        layout='wide'
    )

    st.title('👁️ Detector de Retinopatia Diabética')
    st.markdown("""
    **Ajude a prevenir a cegueira antes que seja tarde demais.**
    
    Esta ferramenta utiliza Inteligência Artificial para analisar exames de fundo de olho e classificar o nível de severidade da retinopatia diabética, a principal causa de cegueira em adultos em idade produtiva.
    
    *Baseado no desafio [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection).*
    """)
    st.warning("**Atenção:** Este é um projeto de demonstração. Os resultados não substituem uma avaliação médica profissional.")

    # Colunas para organizar o layout
    col1, col2 = st.columns(2)

    with col1:
        # Carrega o modelo de IA
        interpreter = carrega_modelo()
        # Interface para carregar a imagem
        image = carrega_imagem_usuario()

    with col2:
        # Realiza a previsão se uma imagem foi carregada
        if image is not None and interpreter is not None:
            realizar_previsao(interpreter, image)
        elif interpreter is not None:
            st.info("Aguardando o upload de um exame de retina para análise.")


if __name__ == "__main__":
    main()
