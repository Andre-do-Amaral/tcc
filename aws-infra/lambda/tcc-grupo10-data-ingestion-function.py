import boto3
import csv
import logging
import urllib.parse

from datetime import datetime

logging.getLogger().setLevel(logging.INFO)

# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import requests
import pandas as pd
from datetime import datetime
import time
import urllib3
import boto3
import logging
import io
from botocore.exceptions import ClientError
# Desativa avisos de segurança para requisições não verificadas
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Configuração de logging (opcional, mas recomendado para Lambda)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Parâmetros do S3
S3_BUCKET_NAME = "tcc-grupo10-mba-raw"
S3_KEY = "dados_sabesp_cantareira_historico.csv"
LOCAL_FILE_PATH = f"/tmp/{S3_KEY}" # Caminho obrigatório no ambiente Lambda
# ==============================================================================
# 2. PARÂMETROS DA CONSULTA (Mantidos do seu código)
# ==============================================================================
BASE_URL = "https://mananciais.sabesp.com.br/api/v4/elementos"
RESERVATORIO_ID = 64 # Sistema Cantareira
VARIAVEIS = {
    #1: 'Chuva (mm)', 2: 'Nível (m)', 3: 'Vazão (M³/s)', 4: 'Volume (hm³)', 
    684: 'Volume Útil Armazenado (%)',
    #685: 'Volume Útil Armazenado (hm³)', 
    #686: 'Volume Total Armazenado (hm³)', 687: 'Vazão Captada (m³/s)', 
    #688: 'Vazão Produzida (m³/s)',
    689: 'Vazão Jusante (m³/s)', 
    690: 'Vazão Natural (m³/s)',
    #691: 'Vazão Afluente (m³/s)', 
    #692: 'Variação do Volume Útil (%)', 693: 'Chuva Acumulada no Mês (mm)', 
    #694: 'Chuva Média Historica (mm)', 695: 'Vazão Jusante no mês (m³/s)', 
    #696: 'Vazão Jusante Média Histórica (m³/s)', 697: 'Vazão Natural no Mês (m³/s)', 
    #698: 'Vazão Natural Media Historica (m³/s)', 699: 'Vazão Produzida no Mês (m³/s)', 
    #700: 'Vazão Retirada no Mês (m³/s)'
}
# ==============================================================================
# 3. HANDLER PRINCIPAL DO LAMBDA
# ==============================================================================
def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    # 3.1. BUSCA O ARQUIVO EXISTENTE NO S3 E DEFINE DATA_INICIO
    try:
        logger.info(f"Tentando baixar {S3_KEY} do S3...")
        
        # 1. Baixa o arquivo do S3 para o disco temporário do Lambda
        s3_client.download_file(S3_BUCKET_NAME, S3_KEY, LOCAL_FILE_PATH)
        
        # 2. Lê o arquivo local para determinar a última data
        dados_existentes = pd.read_csv(
            LOCAL_FILE_PATH, 
            sep=';', decimal=',', encoding='utf-8-sig', 
            parse_dates=['Data'], dayfirst=True
        )
        
        # Garante que a coluna 'Data' seja o índice
        if 'Data' in dados_existentes.columns:
            data_maxima = pd.to_datetime(dados_existentes['Data']).max()
            DATA_INICIO = data_maxima.strftime('%Y-%m-%d')
        else:
            raise ValueError("Coluna 'Data' não encontrada no arquivo CSV existente.")
            
        logger.info(f"Arquivo existente baixado. Última data: {DATA_INICIO}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # Caso de primeira execução
            logger.warning("Arquivo não encontrado no S3. Usando data de início histórica (2000-01-01).")
            DATA_INICIO = "2000-01-01"
            dados_existentes = pd.DataFrame()
        else:
            logger.error(f"Erro ao baixar arquivo do S3: {e}")
            raise e
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo CSV: {e}")
        raise e
    # 3.2. DEFINE A DATA FINAL DA CONSULTA
    DATA_FIM = datetime.today().strftime('%Y-%m-%d')
    
    if DATA_INICIO == DATA_FIM:
        logger.info("As datas de início e fim são iguais. Não há novos dados a buscar.")
        return {
            'statusCode': 200,
            'body': 'Nenhuma nova informação baixada, a data de início é a mesma de hoje.'
        }
        
    # 3.3. EXECUÇÃO DA EXTRAÇÃO DA API
    lista_de_dataframes = []
    logger.info(f"Iniciando busca de dados para o período: de {DATA_INICIO} a {DATA_FIM}")
    for var_id, nome_coluna in VARIAVEIS.items():
        url = f"{BASE_URL}/{RESERVATORIO_ID}/dados/{var_id}/diario/{DATA_INICIO}/{DATA_FIM}"
        
        try:
            response = requests.get(url, timeout=90, verify=False)
            response.raise_for_status()
            dados_json = response.json()
            
            if dados_json.get("succeeded") and dados_json.get("data"):
                df_temp = pd.DataFrame(dados_json['data'])
                df_temp['Data'] = pd.to_datetime(df_temp['date']).dt.date
                df_temp.set_index('Data', inplace=True)
                df_temp = df_temp[['value']].rename(columns={'value': nome_coluna})
                lista_de_dataframes.append(df_temp)
                logger.info(f"Sucesso ao buscar {nome_coluna}. Registros: {len(df_temp)}")
            else:
                logger.warning(f"API não retornou dados para '{nome_coluna}'.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição para '{nome_coluna}': {e}")
        except Exception as e:
            logger.error(f"Erro inesperado ao processar '{nome_coluna}': {e}")
            
        time.sleep(1) # Pausa de 1 segundo (reduzido de 2s para otimizar o tempo de execução do Lambda)
    # 3.4. CONSOLIDAÇÃO, EXPORTAÇÃO LOCAL E UPLOAD PARA S3
    if lista_de_dataframes:
        logger.info("Consolidando dados...")
        
        # 1. Concatena os DataFrames das variáveis novas
        df_novo = pd.concat(lista_de_dataframes, axis=1)
        df_novo.index = pd.to_datetime(df_novo.index)
        df_novo.sort_index(inplace=True)
        #df_novo.index.name = 'Data'
        #df_novo.reset_index(inplace=True)
        
        # 2. Concatena os dados existentes com os novos, se houver dados existentes
        if not dados_existentes.empty:
            dados_existentes['Data'] = pd.to_datetime(dados_existentes['Data'], dayfirst=True)
            dados_existentes.set_index('Data', inplace=True)
            df_final = pd.concat([dados_existentes, df_novo])
            # Remove duplicatas pela data, mantendo o mais novo ('last')
            df_final = df_final[~df_final.index.duplicated(keep='last')]
            #df_final.drop_duplicates(subset=['Data'], keep='last', inplace=True)
        else:
            # Primeira execução, apenas os novos dados são o resultado final
            df_final = df_novo
        # 3. Exportação para CSV local no /tmp
        df_final = df_final[['Volume Útil Armazenado (%)', 'Vazão Jusante (m³/s)', 'Vazão Natural (m³/s)']]
        df_final.to_csv(
            LOCAL_FILE_PATH,
            index=True, # Não queremos o índice do DataFrame como coluna
            index_label='Data',
            header=True,
            sep=';',
            decimal=',',
            date_format='%d/%m/%Y',
            encoding='utf-8-sig'
        )
        logger.info(f"Dados consolidados e salvos localmente em {LOCAL_FILE_PATH}.")
        # 4. Upload para S3
        s3_client.upload_file(LOCAL_FILE_PATH, S3_BUCKET_NAME, S3_KEY)
        logger.info(f"Arquivo atualizado enviado com sucesso para s3://{S3_BUCKET_NAME}/{S3_KEY}")
        
        return {
            'statusCode': 200,
            'body': f'Processo finalizado com sucesso. Dados de {DATA_INICIO} a {DATA_FIM} atualizados. {len(df_final)} registros totais.'
        }
    else:
        logger.info("Nenhuma informação foi baixada. Verifique as URLs, a conexão ou o período.")
        return {
            'statusCode': 200,
            'body': 'Nenhuma nova informação baixada. Verifique os logs.'
        }