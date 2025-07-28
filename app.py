import streamlit as st
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import gdown # Biblioteca para download de arquivos do Google Drive
import plotly.express as px

# --- FUNÇÕES DE CARREGAMENTO ---

@st.cache_resource(show_spinner="Carregando modelo de IA...")
def carrega_modelo():
    # Descomente as linhas abaixo se precisar baixar o modelo do Google Drive
    
    url = 'https://drive.google.com/uc?id=1819W4_0_PfSNVIloaehc_mNnYjRwXu8s' # Novo de 32 bits
    model_escolhido = 'modelo_quantizado_float32.tflite'
    gdown.download(url,model_escolhido)
    
    #url = 'https://drive.google.com//uc?id=1c8fR3xGwiMqduV2frmQYi-UHrhLjvvJm' # Novo de 16 bits
    #model_escolhido = 'modelo_quantizado_float16.tflite'
    #gdown.download(url,model_escolhido)
    
    #url = 'https://drive.google.com/uc?id=1pHKPIVO4IJmIFrlAqpgMAv3kbRzdD_f4' # Antigo de 16 bits  
    #model_escolhido = 'modelo_quantizado16bits.tflite'
    #gdown.download(url, model_escolhido)
    
    # Certifique-se que o modelo está na mesma pasta do script
    try:
        interpreter = tf.lite.Interpreter(model_path=model_escolhido)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Erro ao carregar o modelo {model_escolhido}. Verifique se o arquivo está na pasta correta.")
        st.error(f"Detalhe do erro: {e}")
        return None


@st.cache_data(show_spinner="Carregando gabarito de diagnósticos...")
def carrega_csv(caminho_csv):
    """
    Carrega o arquivo CSV com os diagnósticos de referência e o prepara para busca rápida.
    """
    try:
        df = pd.read_csv(caminho_csv)
        # Define 'id_code' como índice para facilitar a busca pelo nome da imagem
        df.set_index('id_code', inplace=True)
        return df
    except FileNotFoundError:
        st.warning(f"Arquivo de gabarito não encontrado em '{caminho_csv}'. A comparação de resultados não estará disponível.")
        return None


# --- FUNÇÕES DE INTERFACE E PREVISÃO ---

def carrega_imagem_usuario():
    """
    Cria a interface para o usuário fazer upload de uma imagem de retina,
    a pré-processa e retorna a imagem e seu ID.
    """
    uploaded_file = st.file_uploader(
        'Arraste e solte um exame de retina ou clique para selecionar',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        # Extrai o ID da imagem (nome do arquivo sem extensão)
        id_code = uploaded_file.name.split('.')[0]

        # Lê e exibe a imagem
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        st.image(image, caption=f"Exame Carregado (ID: {id_code})", width=400)

        # Pré-processamento da imagem para o modelo
        target_size = (224, 224)
        image_resized = image.resize(target_size)
        
        image_array = np.array(image_resized, dtype=np.float32)
        #image_array = image_array / 255.0   # Normalização
        image_array = np.expand_dims(image_array, axis=0) # Adiciona dimensão do batch

        return image_array, id_code
    
    return None, None # Retorna None para ambos se nenhum arquivo for carregado


def realizar_previsao(interpreter, image):
    """
    Executa a inferência do modelo na imagem, exibe os resultados
    e retorna a classe com maior probabilidade.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    classes = [
        '0 - Sem Retinopatia', '1 - Leve', '2 - Moderada', 
        '3 - Severa', '4 - Proliferativa'
    ]

    df = pd.DataFrame({
        'classes': classes,
        'probabilidades': output_data[0] * 100
    })
    df = df.sort_values(by='probabilidades', ascending=True)

    fig = px.bar(
        df, y='classes', x='probabilidades', orientation='h',
        text=df['probabilidades'].apply(lambda x: f'{x:.2f}%'),
        title='Diagnóstico de Retinopatia Diabética',
        labels={'classes': 'Nível de Severidade', 'probabilidades': 'Probabilidade (%)'}
    )
    fig.update_layout(
        xaxis_title="Confiança da IA (%)", yaxis_title="Diagnóstico Possível",
        uniformtext_minsize=8, uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    diagnostico_final = df.iloc[-1]
    st.success(f"**Diagnóstico Mais Provável:** {diagnostico_final['classes']} com {diagnostico_final['probabilidades']:.2f}% de confiança.")
    
    # Extrai o NÚMERO da classe da string (ex: '0' de '0 - Sem Retinopatia')
    predicao_modelo = int(diagnostico_final['classes'].split(' ')[0])
    
    return predicao_modelo


def exibir_comparacao_diagnostico(df_gabarito, id_code, predicao_modelo):
    """
    Compara o resultado da IA com o gabarito do CSV e exibe o resultado.
    """
    st.markdown("---")
    st.subheader("✔️ Verificação do Resultado")

    if df_gabarito is None:
        # O aviso já foi mostrado na função carrega_csv
        return

    try:
        # Busca o diagnóstico real no DataFrame usando o id_code como índice
        diagnostico_real = int(df_gabarito.loc[id_code, 'diagnosis'])
        
        st.write(f"**ID da Imagem:** `{id_code}`")
        
        col1, col2 = st.columns(2)
        col1.metric(label="Diagnóstico da IA", value=predicao_modelo)
        col2.metric(label="Diagnóstico de Referência (CSV)", value=diagnostico_real)

        if predicao_modelo == diagnostico_real:
            st.success("✅ **Excelente!** O diagnóstico da IA corresponde ao gabarito.", icon="✅")
        else:
            st.error(f"❌ **Atenção!** O diagnóstico da IA ({predicao_modelo}) diverge do gabarito ({diagnostico_real}).", icon="❌")

    except KeyError:
        st.warning(f"O ID da imagem `{id_code}` não foi encontrado no arquivo de gabarito. Não é possível comparar.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao comparar os diagnósticos: {e}")


def main():
    """
    Função principal que organiza a interface do Streamlit.
    """
    st.set_page_config(
        page_title='Detector de Retinopatia Diabética',
        page_icon='👁️',
        layout='wide'
    )

    st.title('👁️ Detector de Retinopatia Diabética com Verificação')
    
        # --- NOVA BARRA LATERAL (SIDEBAR) À ESQUERDA ---
    with st.sidebar:
        st.header("Autor do Projeto")
        # Substitua com seus dados
        st.markdown("**Nome:**")
        st.text("DANIEL OLIVEIRA")  # st.text cria um texto simples
        st.text("RENAN DE CASTRO") 
        st.text("THALLES ROBSON")
        st.markdown("**Matrícula:**")
        st.text("[564307][564258][564322]")
        st.markdown("---")
        st.info("Aplicação de IA para análise de Retinopatia Diabética.")
    st.sidebar.markdown("---")
    st.markdown("""
    **Ajude a prevenir a cegueira antes que seja tarde demais.**
    
    Esta ferramenta utiliza Inteligência Artificial para analisar exames de fundo de olho e classificar o nível de severidade da retinopatia diabética. Faça o upload de uma imagem do dataset para comparar o resultado da IA com o diagnóstico real.
    
    *Baseado no desafio [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection).*
    """)
    st.warning("**Atenção:** Este é um projeto de demonstração. Os resultados não substituem uma avaliação médica profissional.")

    # --- CARREGAMENTO DOS DADOS ---
    # Coloque seu arquivo CSV em uma pasta chamada 'dados_csv'
    #df_gabarito = carrega_csv('dados_csv/dados_concatenados.csv')
    df_gabarito = carrega_csv('https://raw.githubusercontent.com/Danieloliver1/painel_deteccao_cegueira/refs/heads/main/dados_csv/dados_concatenados.csv')
    
    interpreter = carrega_modelo()
    
    # --- LAYOUT DA PÁGINA ---
    col1, col2 = st.columns(2)

    with col1:
        image, id_code = carrega_imagem_usuario()

    with col2:
        if image is not None and interpreter is not None:
            predicao_modelo = realizar_previsao(interpreter, image)
            # A nova função é chamada aqui para exibir a comparação
            exibir_comparacao_diagnostico(df_gabarito, id_code, predicao_modelo)
        elif interpreter is not None:
            st.info("Aguardando o upload de um exame de retina para análise.")
        else:
            st.error("O aplicativo não pode continuar pois o modelo de IA não foi carregado.")


if __name__ == "__main__":
    main()