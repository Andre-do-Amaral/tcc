import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

def gerar_periodos_mensais(data_inicial, data_final):
    """
    Gera uma lista de tuplas (inicio_mes, fim_mes) para um intervalo de datas.
    Respeita a regra de buscar um mês de cada vez.
    """
    periodos = []
    data_corrente = data_inicial
    
    while data_corrente < data_final:
        # O início do período é o primeiro dia do mês corrente
        #inicio_periodo = data_corrente.replace(day=1)
        inicio_periodo = data_corrente
        
        # O fim do período é o último dia do mesmo mês
        fim_periodo = inicio_periodo + relativedelta(months=1) - relativedelta(days=1)
        
        # Garante que o fim do período não ultrapasse a data final geral
        if fim_periodo > data_final:
            fim_periodo = data_final
            
        periodos.append((inicio_periodo, fim_periodo))
        
        # Avança para o próximo mês
        data_corrente = inicio_periodo + relativedelta(months=1)
        
    return periodos

def buscar_e_converter_dados(reservatorio_id, data_inicial, data_final):
    """
    Busca os dados de um reservatório para um período específico,
    analisa o HTML e retorna um DataFrame do Pandas.
    """
    base_url = "https://www.ana.gov.br/sar0/MedicaoCantareira"
    
    # Formata as datas para o formato esperado pela URL (dd/mm/yyyy)
    params = {
        'dropDownListReservatorios': reservatorio_id,
        'dataInicial': data_inicial.strftime('%d/%m/%Y'),
        'dataFinal': data_final.strftime('%d/%m/%Y')
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()  # Lança exceção para erros HTTP (4xx ou 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Erro ao fazer a requisição para o período {params['dataInicial']} - {params['dataFinal']}: {e}")
        return pd.DataFrame() # Retorna DataFrame vazio em caso de erro

    # Analisa o HTML com BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontra a tabela de dados
    table = soup.find('table', class_='table-striped')

    # Se não houver tabela na página, retorna um DataFrame vazio
    if not table:
        return pd.DataFrame()

    # Extrai os cabeçalhos
    headers = [header.get_text(strip=True) for header in table.find('thead').find_all('th')]
    
    # Extrai as linhas de dados
    rows = []
    for row in table.find('tbody').find_all('tr'):
        cols = [col.get_text(strip=True) for col in row.find_all('td')]
        if cols: # Garante que a linha não está vazia
            rows.append(cols)

    if not rows:
        return pd.DataFrame()

    # Cria o DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # --- Limpeza e conversão de tipos de dados ---
    numeric_cols = [
        'Cota (m)', 'Volume Útil (hm³)', 'Volume Útil (%)',
        'Afluência (m³/s)', 'Defluência (m³/s)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

    df['Data da Medição'] = pd.to_datetime(df['Data da Medição'], format='%d/%m/%Y')
    df['Código do Reservatório'] = df['Código do Reservatório'].astype(int)
    df['Reservatório'] = df['Reservatório'].str.strip()
    
    return df

def main():
    """
    Função principal para orquestrar o scraping de dados históricos.
    """
    

    year_final = datetime.today().year
    mes_final = datetime.today().month
    dia_final = datetime.today().day

    dados = pd.read_csv('dados_reservatorio.csv')
    dados['Data da Medição'] = pd.to_datetime(dados['Data da Medição'], format='%Y-%m-%d')
    ultima_data = dados['Data da Medição'].max() + timedelta(days=1)
    ano = ultima_data.year
    mes = ultima_data.month
    dia = ultima_data.day
    RESERVATORIO_ID = 29002  # Cachoeira
    #DATA_INICIAL_GERAL = datetime(2000, 1, 1)
    #DATA_FINAL_GERAL = datetime(2025, 9, 10)
    DATA_INICIAL_GERAL = datetime(ano, mes, dia)
    DATA_FINAL_GERAL = datetime(year_final, mes_final, dia_final)

    print(f"Iniciando coleta de dados para o reservatório ID {RESERVATORIO_ID}")
    print(f"Período total: de {DATA_INICIAL_GERAL.strftime('%d/%m/%Y')} a {DATA_FINAL_GERAL.strftime('%d/%m/%Y')}\n")

    # 1. Gera todos os períodos mensais a serem consultados
    periodos_de_busca = gerar_periodos_mensais(DATA_INICIAL_GERAL, DATA_FINAL_GERAL)
    
    lista_de_dataframes = []

    # 2. Itera sobre cada período, busca os dados e armazena
    for inicio, fim in periodos_de_busca:
        print(f"Buscando dados de: {inicio.strftime('%d/%m/%Y')} a {fim.strftime('%d/%m/%Y')}...")
        
        df_mensal = buscar_e_converter_dados(RESERVATORIO_ID, inicio, fim)
        if not df_mensal.empty:
            lista_de_dataframes.append(df_mensal)
            print(f"-> {len(df_mensal)} registros encontrados.")
        else:
            print("-> Nenhum registro encontrado para este período.")
            
        # Pausa para ser gentil com o servidor
        time.sleep(0.5) 

    # 3. Concatena todos os DataFrames em um só
    if lista_de_dataframes:
        df_final = pd.concat(lista_de_dataframes, ignore_index=True)
        print("\n\n--- Coleta Finalizada com Sucesso! ---")
        df_final.to_csv("./dados_reservatorio_append.csv", index=False)
        print(f"Total de registros coletados: {len(df_final)}")

        
        # Opcional: Salvar os dados em um arquivo CSV para uso futuro
        # nome_arquivo = f"dados_reservatorio_{RESERVATORIO_ID}.csv"
        # df_final.to_csv(nome_arquivo, index=False)
        # print(f"\nDados salvos em '{nome_arquivo}'")

    else:
        print("\nNenhum dado foi coletado no período especificado.")
    
    dados = pd.read_csv('dados_reservatorio.csv')
    dados_append = pd.read_csv('dados_reservatorio_append.csv')
    dados_final = pd.concat([dados, dados_append], ignore_index=True)
    dados_final.to_csv('dados_reservatorio.csv', index=False)
main()