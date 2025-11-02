# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import requests
import pandas as pd
from datetime import datetime
import time
import urllib3

# A API pode usar um certificado SSL que falha na verificação.
# O código abaixo desativa os avisos de segurança ao usar `verify=False`.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("✅ Bibliotecas importadas e configuração concluída.")

# ==============================================================================
# 2. PARÂMETROS DA CONSULTA
# ==============================================================================

# URL base da nova API
BASE_URL = "https://mananciais.sabesp.com.br/api/v4/elementos"

# ID do reservatório (64 = Sistema Cantareira)
RESERVATORIO_ID = 64

# Pegando data de inicio (ultima data do arquivo salvo)
dados = pd.read_csv('dados_sabesp_cantareira_historico.csv', sep=';', decimal=',', encoding='utf-8-sig', parse_dates=['Data'], dayfirst=True)
data = pd.to_datetime(dados['Data']).max()
DATA_INICIO = data.strftime('%Y-%m-%d')

# Datas para a consulta
# DATA_INICIO = "2000-01-01"
DATA_FIM = datetime.today().strftime('%Y-%m-%d')

# Dicionário com o ID de cada variável e o nome da coluna correspondente
# Nota: O nome 'Vazão Produzida no Mês (m³/s)' aparece duplicado para IDs 698 e 699.
# Adicionei um sufixo "_2" para diferenciar a segunda ocorrência no DataFrame.
VARIAVEIS = {
    1: 'Chuva (mm)',
    2: 'Nível (m)',
    3: 'Vazão (M³/s)',
    4: 'Volume (hm³)',
    684: 'Volume Útil Armazenado (%)',
    685: 'Volume Útil Armazenado (hm³)',
    686: 'Volume Total Armazenado (hm³)',
    687: 'Vazão Captada (m³/s)',
    688: 'Vazão Produzida (m³/s)',
    689: 'Vazão Jusante (m³/s)',
    690: 'Vazão Natural (m³/s)',
    691: 'Vazão Afluente (m³/s)',
    692: 'Variação do Volume Útil (%)',
    693: 'Chuva Acumulada no Mês (mm)',
    694: 'Chuva Média Historica (mm)',
    695: 'Vazão Jusante no mês (m³/s)',
    696: 'Vazão Jusante Média Histórica (m³/s)',
    697: 'Vazão Natural no Mês (m³/s)',
    698: 'Vazão Natural Media Historica (m³/s)',
    699: 'Vazão Produzida no Mês (m³/s)',
    700: 'Vazão Retirada no Mês (m³/s)'
}

# ==============================================================================
# 3. EXECUÇÃO DA EXTRAÇÃO
# ==============================================================================

# Lista para armazenar os DataFrames de cada variável
lista_de_dataframes = []

print(f"\n🚀 Iniciando a busca de dados para o Sistema Cantareira (ID: {RESERVATORIO_ID})")
print(f"Período da consulta: de {DATA_INICIO} a {DATA_FIM}")

# Loop para buscar os dados de cada variável
for var_id, nome_coluna in VARIAVEIS.items():
    print(f"\nBuscando dados para: '{nome_coluna}' (ID: {var_id})...")
    
    # Monta a URL completa para a requisição
    url = f"{BASE_URL}/{RESERVATORIO_ID}/dados/{var_id}/diario/{DATA_INICIO}/{DATA_FIM}"
    print(f"  -> URL: {url}")

    try:
        # Faz a requisição GET
        response = requests.get(url, timeout=90, verify=False)
        
        # Lança um erro para respostas com status 4xx ou 5xx (ex: 404 Not Found)
        response.raise_for_status()
        
        # Converte a resposta para JSON
        dados_json = response.json()
        
        # Verifica se a requisição foi bem-sucedida e se há dados
        if dados_json.get("succeeded") and dados_json.get("data"):
            
            # Cria um DataFrame temporário com os dados da variável
            df_temp = pd.DataFrame(dados_json['data'])
            
            # Converte a coluna 'date' para o formato de data, removendo as horas
            df_temp['Data'] = pd.to_datetime(df_temp['date']).dt.date
            
            # Define a data como o índice do DataFrame
            df_temp.set_index('Data', inplace=True)
            
            # Seleciona apenas a coluna 'value' e a renomeia com o nome correto
            df_temp = df_temp[['value']].rename(columns={'value': nome_coluna})
            
            # Adiciona o DataFrame processado à nossa lista
            lista_de_dataframes.append(df_temp)
            
            print(f"  -> ✅ Sucesso! {len(df_temp)} registros encontrados.")
            
        else:
            # Caso a API retorne sucesso mas sem dados, ou com a flag "succeeded": false
            print(f"  -> ⚠️ Aviso: A API não retornou dados para '{nome_coluna}'. Pulando.")

    except requests.exceptions.RequestException as e:
        print(f"  -> ❌ Erro de requisição para '{nome_coluna}': {e}")
    except Exception as e:
        print(f"  -> ❌ Ocorreu um erro inesperado ao processar '{nome_coluna}': {e}")
        
    # Pausa de 2 segundos para não sobrecarregar o servidor da Sabesp
    time.sleep(2)

# ==============================================================================
# 4. CONSOLIDAÇÃO E EXPORTAÇÃO
# ==============================================================================

if lista_de_dataframes:
    print("\nConsolidando todos os dados em um único DataFrame...")
    
    # Concatena todos os DataFrames da lista em um só, alinhando pelo índice (data)
    df_final = pd.concat(lista_de_dataframes, axis=1)
    
    # Ordena o índice por data, caso haja alguma desordem
    df_final.sort_index(inplace=True)
    df_final = pd.concat([dados, df_final.reset_index()]).reset_index().drop('index', axis = 1)  # Adiciona os dados já existentes
    
    print("  -> Consolidação concluída!")
    
    # Exportação para CSV
    nome_arquivo = "dados_sabesp_cantareira_historico.csv"
    
    df_final.to_csv(
        nome_arquivo,
        index=True,
        index_label='Data',
        header=True,
        sep=';',
        decimal=',',
        date_format='%d/%m/%Y',
        encoding='utf-8-sig' # 'utf-8-sig' garante compatibilidade com Excel
    )
    
    print(f"\n\n🎉 BRABA! Processo finalizado com sucesso!")
    print(f"Arquivo salvo em: ./{nome_arquivo}")
    print("\nVisualização das 5 primeiras linhas do resultado:")
    print(df_final.head())

else:
    print("\n\n❌ Nenhuma informação foi baixada. Verifique as URLs e a conexão.")