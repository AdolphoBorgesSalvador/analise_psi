import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from typing import Tuple, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

CAMINHO_PSI = os.getenv("CAMINHO_PSI")
CAMINHO_MAPA = os.getenv("CAMINHO_MAPA")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

def conectar_postgres():
    """
    Estabelece conexão com o banco de dados PostgreSQL.

    Returns:
        engine: SQLAlchemy engine object ou None em caso de erro
    """
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        return create_engine(connection_string)
    except Exception as e:
        print(f"Erro ao conectar ao PostgreSQL: {e}")
        return None   

def carregar_dados(
    caminho_psi: str, caminho_mapa: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega dados dos arquivos PSI e Mapa.
    
    Args:
        caminho_psi: Caminho para o arquivo PSI
        caminho_mapa: Caminho para o arquivo Mapa

    Returns:
        Tuple contendo DataFrames do PSI e Mapa

    Como Usar:
        psi, mapa = carregar_dados("caminho/para/psi.xlsx", "caminho/para/mapa.xlsx")
    """
    psi = pd.read_excel(caminho_psi, skiprows=7, sheet_name="Summary")
    mapa = pd.read_excel(caminho_mapa)
    return psi, mapa

def primeiro_calculo(psi,mapa):
    """
    Como Usar:
        mapa_processado = primeiro_calculo(psi, mapa)
    """

    lista = psi["Code"].tolist()

    mapa = mapa[[
        'Material','Descrição',
        '07/2024', '08/2024', '09/2024',
        '10/2024', '11/2024', '12/2024', '01/2025', '02/2025', '03/2025',
        '04/2025', '05/2025', '06/2025',
        'Quant estoque 06/2025','Backorder total', 'P.O. 07/2025', 'P.O. 08/2025',
        'P.O. 09/2025', 'P.O. 10/2025', 'P.O. 11/2025', 'P.O. 12/2025',
        'Total Mov.'
        ]]

    mapa = mapa.rename(columns={
        '07/2024': 'Jul/24', '08/2024': 'Ago/24', '09/2024': 'Set/24',
        '10/2024': 'Out/24', '11/2024': 'Nov/24', '12/2024': 'Dez/24',
        '01/2025': 'Jan/25', '02/2025': 'Fev/25', '03/2025': 'Mar/25',
        '04/2025': 'Abr/25', '05/2025': 'Mai/25', '06/2025': 'Jun/25',
        'Quant estoque 06/2025': 'Estoque Jun/25','Backorder total' : 'Pendencia',
        'P.O. 07/2025': 'IMP 07/25', 'P.O. 08/2025': 'IMP 08/25',
        'P.O. 09/2025': 'IMP 09/25', 'P.O. 10/2025': 'IMP 10/25',
        'P.O. 11/2025': 'IMP 11/25', 'P.O. 12/2025': 'IMP 12/25','Total Mov.':'Total Mov'
    })

    mapa = mapa[mapa["Material"].isin(lista)]


    mapa["media_12m"]  = mapa[['Jul/24', 'Ago/24', 'Set/24', 'Out/24', 'Nov/24', 'Dez/24',
                            'Jan/25','Fev/25','Mar/25','Abr/25','Mai/25','Jun/25',]].mean(axis=1)

    mapa["media_6m"]  = mapa[['Jan/25','Fev/25','Mar/25','Abr/25','Mai/25','Jun/25',]].mean(axis=1)


    mapa["media_3m"]  = mapa[['Abr/25','Mai/25','Jun/25',]].mean(axis=1)

    mapa["mediana_6m"] = mapa[["Jan/25","Fev/25","Mar/25","Abr/25","Mai/25","Jun/25"]].median(axis=1)

    mapa['Total IMP'] = mapa[['IMP 07/25', 'IMP 08/25', 'IMP 09/25', 'IMP 10/25', 'IMP 11/25', 'IMP 12/25']].sum(axis=1)

    mapa["Mos_Com_Imp"] = (mapa["Estoque Jun/25"] + mapa["Total IMP"]) / mapa["media_6m"].where(mapa["media_6m"] != 0, 1)

    mapa["Mos_Sem_Imp"] = mapa["Estoque Jun/25"] / mapa["media_6m"].where(mapa["media_6m"] != 0, 1)
    return mapa


def classificar_abc(percentual):
    """
    Classifica categoria ABC com base no percentual acumulado.
    """
    if percentual <= 80:
        return "A"
    elif percentual <= 95:
        return "B"
    else:
        return "C"

def classificar_xyz(cv):
    """
    Classifica categoria XYZ com base no coeficiente de variação.
    """
    if cv <= 0.5:
        return "X"
    elif cv <= 1.0:
        return "Y"
    else:
        return "Z"

def classificar(mapa):
    """
    Executa análise ABC/XYZ e retorna o DataFrame classificado.
    
    Args:
        mapa (DataFrame): DataFrame com colunas de consumo mensal e Total Mov.
    
    Returns:
        DataFrame com colunas de classificação ABC/XYZ.
    """

    mapa = mapa.sort_values("Total Mov", ascending=False).reset_index(drop=True)

    mapa["Acumulado"] = mapa["Total Mov"].cumsum()
    mapa["Percentual"] = 100 * mapa["Acumulado"] / mapa["Total Mov"].sum()

    mapa["Classe_ABC"] = mapa["Percentual"].apply(classificar_abc)

    colunas_meses = [
        'Jul/24', 'Ago/24', 'Set/24', 'Out/24', 'Nov/24', 'Dez/24',
        'Jan/25', 'Fev/25', 'Mar/25', 'Abr/25', 'Mai/25', 'Jun/25'
    ]

    mapa["Media_Mensal"] = mapa[colunas_meses].mean(axis=1)
    mapa["Desvio_Mensal"] = mapa[colunas_meses].std(axis=1)

    mapa["CV"] = mapa["Desvio_Mensal"] / mapa["Media_Mensal"].replace(0, 1)

    mapa["Classe_XYZ"] = mapa["CV"].apply(classificar_xyz)

    mapa["Classe_ABC_XYZ"] = mapa["Classe_ABC"] + mapa["Classe_XYZ"]

    return mapa


# a
psi, mapa = carregar_dados(CAMINHO_PSI, CAMINHO_MAPA)
mapa_processado = primeiro_calculo(psi, mapa)
mapa_classificado = classificar(mapa_processado)

mapa_classificado = mapa_classificado[['Material', 'Descrição', 'Jul/24', 'Ago/24', 'Set/24', 'Out/24',
       'Nov/24', 'Dez/24', 'Jan/25', 'Fev/25', 'Mar/25', 'Abr/25', 'Mai/25',
       'Jun/25', 'Estoque Jun/25', 'Pendencia', 'IMP 07/25', 'IMP 08/25',
       'IMP 09/25', 'IMP 10/25', 'IMP 11/25', 'IMP 12/25', 'Total Mov',
       'media_12m', 'media_6m', 'media_3m', 'mediana_6m', 'Total IMP',
       'Mos_Sem_Imp','Classe_ABC_XYZ']]

mapa_classificado


mapa_classificado.to_json(
    os.getenv("CAMINHO_JSON_CLASSIFICADO"),
    orient="records",
    force_ascii=False,
    indent=4
)


