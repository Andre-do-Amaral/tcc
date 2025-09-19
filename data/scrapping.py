# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES
# ==============================================================================
import os
import json
import time
import zlib  # Essencial para descompactar a resposta da API
import requests
import pandas as pd
from datetime import date, datetime, timedelta
import calendar

# A nova API usa um certificado SSL que pode falhar na verificação.
# O código abaixo desativa os avisos de segurança ao usar `verify=False`.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("Bibliotecas importadas e configuração concluída.")
year_final = datetime.today().year
month_final = datetime.today().month
day_final = datetime.today().day

dados = pd.read_csv('./tab_cantareira.csv', sep=';', parse_dates=['Data'], dayfirst=True)

ultima_data = dados['Data'].max() + timedelta(days=1)
year = ultima_data.year
month = ultima_data.month
day = ultima_data.day

# ==============================================================================
# 2. PARÂMETROS DA CONSULTA
# ==============================================================================
#DATA_INICIO = date(2000, 1, 1)
DATA_INICIO = date(year, month, day)
DATA_FIM = date(year_final, month_final, day_final)
SISTEMA_ID = 0  # 0 = Sistema Cantareira

# ==============================================================================
# 3. FUNÇÕES DA NOVA DOCUMENTAÇÃO (Refatoradas para maior clareza)
# ==============================================================================

def get_json(url):
    """
    Busca os dados na API e retorna o JSON.
    A API agora envia JSON puro, então a descompressão com zlib não é mais necessária.
    """
    try:
        print(f"  Fazendo requisição para: {url}")
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()  # Verifica se houve erro na requisição (4xx ou 5xx)
        
        # A MUDANÇA ESTÁ AQUI: Usamos o método .json() direto da resposta
        data = r.json() 
        
        return json.dumps(data) # json.dumps ainda é útil para manter a estrutura
    except requests.exceptions.RequestException as e:
        print(f"  !!! Erro de requisição: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  !!! Erro ao decodificar JSON. A resposta não parece ser um JSON válido: {e}")
        print(f"      Conteúdo recebido: {r.text[:200]}") # Mostra o início da resposta
        return None

def json2df(jsn):
    df = pd.read_json(jsn)
    return df.drop(['FlagHasError', 'Message'], axis=1, errors='ignore')

def rename_field(x):
    return str(x).replace('/', '-').replace(' (', '-').replace('-', '_').replace(')', '').replace('Cesp', 'CESP').replace('Represa ', '').replace(' ', '')

def list_represas(df):
    lst = df.loc['ListaRepresas']['ReturnObj']
    df_represas = pd.json_normalize(lst)
    return df_represas.drop(['temChuva','temNivel', 'temQjus', 'temQnat', 'temVolume'], axis=1, errors='ignore')

def safe_merge(df_full, df_new):
    """Função auxiliar para fazer o merge de forma segura, tratando o primeiro caso."""
    if df_full.empty:
        return df_new
    else:
        # Usar how='outer' para não perder dados se as datas não alinharem perfeitamente
        return pd.merge(df_full, df_new, on='Data', how='outer')

def list_volumes(df):
    lst = df.loc['ListaDados']['ReturnObj']
    
    # --- NOVA CORREÇÃO APLICADA AQUI ---
    # Iteramos manualmente para criar uma lista "limpa" de registros,
    # ignorando qualquer valor 'None' que apareça dentro da lista 'Dados'.
    records_limpos = []
    for item_diario in lst:
        # Pula dias que não têm a lista 'Dados'
        if not isinstance(item_diario.get('Dados'), list):
            continue
        # Pega apenas os dicionários válidos de dentro da lista 'Dados'
        for record in item_diario['Dados']:
            if isinstance(record, dict):
                records_limpos.append(record)

    # Se não houver nenhum registro válido, retorna um DataFrame vazio
    if not records_limpos:
        return pd.DataFrame()
    
    # Cria o DataFrame a partir da lista de registros já limpa e plana
    df_normalized = pd.DataFrame(records_limpos)
    # --- FIM DA CORREÇÃO ---

    fields = sorted(list(set(df_normalized['Nome'])))
    df_full = pd.DataFrame()

    for field_name in fields:
        j = rename_field(field_name)
        temp_df = df_normalized[df_normalized['Nome'] == field_name].copy()
        temp_df.drop(['FlagConsolidado', 'NAMaxMax', 'NAMinMin', 'QJusanteMax', 'QJusanteMin', 'NivelUltimoDia', 'SistemaId', 'ComponenteId', 'UltimoDia', 'VazaoJusantePrincipal', 'VazaoJusanteSecundaria', 'VolumeOperacionalUltimoDia', 'VolumePorcentagemUltimoDia', 'VolumeTotalUltimoDia', 'Nome'], axis=1, errors='ignore', inplace=True)
        temp_df.columns = [col if col == 'Data' else f"{j}_{col}" for col in temp_df.columns]
        temp_df['Data'] = pd.to_datetime(temp_df['Data'])
        df_full = safe_merge(df_full, temp_df)
    
    if not df_full.empty:
        df_full.set_index('Data', inplace=True)
    return df_full

def list_vazao(df):
    df_represas = list_represas(df)
    lst = df.loc['ListaDados']['ReturnObj']

    # --- NOVA CORREÇÃO APLICADA AQUI ---
    # Lógica idêntica à de list_volumes, mas para a chave 'Qnat'
    records_limpos = []
    for item_diario in lst:
        if not isinstance(item_diario.get('Qnat'), list):
            continue
        for record in item_diario['Qnat']:
            if isinstance(record, dict):
                records_limpos.append(record)

    if not records_limpos:
        return pd.DataFrame()
        
    df_normalized = pd.DataFrame(records_limpos)
    # --- FIM DA CORREÇÃO ---
    
    df_merged = pd.merge(df_normalized, df_represas, on='ComponenteId', how='outer')
    fields = sorted(list(set(df_merged['Nome'].dropna())))
    df_full = pd.DataFrame()

    for field_name in fields:
        j = rename_field(field_name)
        temp_df = df_merged[df_merged['Nome'] == field_name].copy()
        temp_df.drop(['ComponenteId', 'Nome', 'VazaoAfluenteMax', 'VazaoAfluenteMin', 'VazaoNaturalMax', 'VazaoNaturalMin'], axis=1, errors='ignore', inplace=True)
        temp_df.columns = [col if col == 'Data' else f"{j}_{col}" for col in temp_df.columns]
        temp_df['Data'] = pd.to_datetime(temp_df['Data'])
        df_full = safe_merge(df_full, temp_df)
        
    if not df_full.empty:
        df_full.set_index('Data', inplace=True)
    return df_full

def list_SE(df):
    lst = df.loc['ListaDados']['ReturnObj']
    df_se = pd.json_normalize(lst)
    df_se.drop(['Dados', 'Data', 'Qnat'], axis=1, errors='ignore', inplace=True)
    col = [f"SE_{c}" for c in df_se.columns]
    col = [c.replace('SistemaEquivalente.', '').replace('SE_Data', 'Data') for c in col]
    df_se.columns = col
    df_se['Data'] = pd.to_datetime(df_se['Data'])
    df_se.set_index('Data', inplace=True)
    return df_se

def list_SC(df):
    lst = df.loc['ListaDadosSistema']['ReturnObj']
    df_sc = pd.json_normalize(lst)
    df_sc.drop(['objSistema.SistemaId', 'objQETA', 'objSistema.Data'], axis=1, errors='ignore', inplace=True)
    col = [f"SC_{c}".replace('objSistema', '').replace('.', '') for c in df_sc.columns]
    col = [c.replace('SC_Data', 'Data') for c in col]
    df_sc.columns = col
    df_sc['Data'] = pd.to_datetime(df_sc['Data'])
    df_sc.set_index('Data', inplace=True)
    return df_sc

def list_vazaoestruturas(df):
    lst = df.loc['ListaDadosLocais']['ReturnObj']
    list_d = [parte2 for parte1 in lst for parte2 in parte1['Dados'] if isinstance(parte2, dict)]
    if not list_d: return pd.DataFrame()
    df_normalized = pd.DataFrame(list_d)
    fields = sorted(list(set(df_normalized['Abreviatura'])))
    df_full = pd.DataFrame()

    for field_name in fields:
        j = rename_field(field_name)
        temp_df = df_normalized[df_normalized['Abreviatura'] == field_name].copy()
        temp_df.drop(['Maximo', 'Minimo', 'Dia', 'Abreviatura', 'ComponenteId', 'LocalMedicaoId', 'Nome', 'SistemaId'], axis=1, errors='ignore', inplace=True)
        temp_df.columns = [col if col == 'Data' else f"{j}_{col}" for col in temp_df.columns]
        temp_df['Data'] = pd.to_datetime(temp_df['Data'])
        df_full = safe_merge(df_full, temp_df)
        
    if not df_full.empty:
        df_full.set_index('Data', inplace=True)
    return df_full

# ==============================================================================
# 4. EXECUÇÃO DO LOOP PRINCIPAL (MÊS A MÊS)
# ==============================================================================
datas_loop = pd.date_range(start=DATA_INICIO, end=DATA_FIM, freq='D')
dfs_list = []
print(datas_loop)

print(f"\nIniciando a busca de dados de {DATA_INICIO.strftime('%Y-%m-%d')} até {DATA_FIM.strftime('%Y-%m-%d')}...")

for start_of_month in datas_loop:
    _, last_day_of_month = calendar.monthrange(start_of_month.year, start_of_month.month)
    end_of_month = date(start_of_month.year, start_of_month.month, last_day_of_month)
    
    # Garante que a data final do chunk não ultrapasse a DATA_FIM geral
    if end_of_month > DATA_FIM:
        end_of_month = DATA_FIM

    print(f"\nBuscando dados para o período: {start_of_month.strftime('%Y-%m-%d')} a {end_of_month.strftime('%Y-%m-%d')}")
    
    url = f"http://mananciais.sabesp.com.br/api/Mananciais/RepresasSistemasNivel/{start_of_month.strftime('%Y-%m-%d')}/{end_of_month.strftime('%Y-%m-%d')}/{SISTEMA_ID}"
    jsn = get_json(url)

    if jsn is None:
        print("  !!! Falha ao obter JSON. Pulando para o próximo mês.")
        continue

    df_base = json2df(jsn)

    # Extrai todas as tabelas de dados
    df_volumes = list_volumes(df_base)
    df_vazao = list_vazao(df_base)
    df_SE = list_SE(df_base)
    df_SC = list_SC(df_base)
    df_vazaoestruturas = list_vazaoestruturas(df_base)

    # Junta todas as tabelas do mês em uma só
    df_mes = pd.concat([df_volumes, df_vazao, df_SE, df_SC, df_vazaoestruturas], axis=1)
    dfs_list.append(df_mes)

    time.sleep(3) # Pausa para não sobrecarregar o servidor

# ==============================================================================
# 5. CONSOLIDAÇÃO FINAL E EXPORTAÇÃO
# ==============================================================================
if dfs_list:
    
    print("\nConsolidando todos os dados baixados...")
    df_final = pd.concat(dfs_list)
    
    # Remove linhas duplicadas (de sobreposição de meses, se houver) e ordena
    df_final = df_final[~df_final.index.duplicated(keep='last')]
    df_final.sort_index(inplace=True)

    # Garante que o DataFrame final cubra todo o período solicitado
    date_index = pd.date_range(start=DATA_INICIO, end=DATA_FIM, freq='D')
    print(date_index)
    df_final = df_final.reindex(date_index)

    # Exportação para CSV

    filename = f"./tab_Cantareira_append.csv"
    
    df_final.to_csv(
        filename,
        index=True,
        index_label='Data',
        header=True,
        sep=';',
        decimal=',',
        date_format='%d/%m/%Y',
        encoding='utf-8-sig'
    )
    
    print(f"\n✅ BRABA! Processo finalizado com sucesso!")
    print(f"Arquivo salvo em: {filepath}")
    print("\nVisualização das 5 primeiras linhas do resultado:")

else:
    print("\n❌ Nenhuma informação foi baixada. Verifique os parâmetros e a conexão.")
    df_final = pd.DataFrame()  # Garante que df_final exista para concatenação

df_1 = pd.read_csv('./tab_cantareira.csv', sep=';', parse_dates=['Data'], dayfirst=True)
df = pd.concat([df_1, df_final.reset_index().rename(columns={'index':'Data'})], axis=0, ignore_index=True)

df.to_csv('./tab_cantareira.csv', 
        index=False,
        header=True,
        sep=';',
        decimal=',',
        date_format='%d/%m/%Y',
        encoding='utf-8-sig')