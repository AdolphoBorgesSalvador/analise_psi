import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from typing import Tuple, List, Optional
from dotenv import load_dotenv
import os

# Carrega variáveis do .env
load_dotenv()

# Configurações dos caminhos dos arquivos
CAMINHO_PSI = os.getenv("CAMINHO_PSI")
CAMINHO_MAPA = os.getenv("CAMINHO_MAPA")
CAMINHO_DISPON = os.getenv("CAMINHO_DISPON")

# Configurações do banco de dados
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
    """
    psi = pd.read_excel(caminho_psi, skiprows=7, sheet_name="Summary")
    mapa = pd.read_excel(caminho_mapa)
    return psi, mapa


def tratar_dados(psi: pd.DataFrame, mapa: pd.DataFrame) -> pd.DataFrame:
    """
    Trata e combina dados do PSI e Mapa.

    Args:
        psi: DataFrame com dados do PSI
        mapa: DataFrame com dados do Mapa

    Returns:
        DataFrame filtrado e tratado
    """
    psi_df = psi[["Code"]].dropna().rename(columns={"Code": "Material"})
    mapa_df = mapa[["Material", "Total Mov.", "Quant estoque 03/2025"]]
    df_merged = psi_df.merge(mapa_df, on="Material", how="left")
    df_filtrado = df_merged.dropna(subset=["Total Mov."])
    return df_filtrado[df_filtrado["Total Mov."] > 0]


def gerar_pareto(
    df: pd.DataFrame, coluna: str = "Total Mov.", top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Gera análise de Pareto para um DataFrame.

    Args:
        df: DataFrame de entrada
        coluna: Nome da coluna para análise
        top_n: Número de itens a retornar (opcional)

    Returns:
        DataFrame com análise de Pareto
    """
    df = df.copy()
    df["Total Mov."] = df[coluna]
    df = df.sort_values(by="Total Mov.", ascending=False)
    df["Cumulative Sum"] = df["Total Mov."].cumsum()
    df["Cumulative %"] = 100 * df["Cumulative Sum"] / df["Total Mov."].sum()

    return df.head(top_n) if top_n else df


def plotar_pareto_customizado(
    df: pd.DataFrame, coluna: str = "Total Mov.", top_n: int = 100
) -> None:
    """
    Plota gráfico de Pareto customizado.

    Args:
        df: DataFrame com dados para plotagem
        coluna: Nome da coluna para análise
        top_n: Número de itens a mostrar
    """
    df = gerar_pareto(df, coluna=coluna, top_n=top_n)

    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Gráfico de barras (frequência)
    ax1.bar(df["Material"], df["Total Mov."], color="skyblue")
    ax1.set_ylabel(coluna)
    ax1.set_xlabel("Material")
    ax1.tick_params(axis="x", rotation=90)

    # Linha de % acumulado
    ax2 = ax1.twinx()
    ax2.plot(
        df["Material"], df["Cumulative %"], color="red", marker="o", linestyle="--"
    )
    ax2.set_ylabel("% Acumulado")
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=1)

    plt.title(f"Gráfico de Pareto - Top {top_n} materiais")
    plt.tight_layout()
    plt.show()


def plotar_pareto_top_n(
    df, top_n=50, coluna_valor="Total Mov.", coluna_material="Material"
):

    df_top = df.head(top_n)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(df_top[coluna_material], df_top[coluna_valor], color="skyblue")
    ax1.set_ylabel("Total Movimentado")
    ax1.set_xticks(range(len(df_top)))
    ax1.set_xticklabels(df_top[coluna_material], rotation=45, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        df_top[coluna_material], df_top["Cumulative %"], color="orange", marker="o"
    )
    ax2.set_ylabel("% Acumulado")
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=1, label="80%")

    plt.title(f"Gráfico de Pareto – {coluna_valor} (Top {top_n})")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


def filtrar_por_classes(df, classe_xyz=None, classe_abc=None):

    filtro = pd.Series([True] * len(df))

    if classe_xyz is not None:
        filtro &= df["Classe XYZ"] == classe_xyz

    if classe_abc is not None:
        filtro &= df["Classe ABC"] == classe_abc

    return df[filtro]


def filtrar_estoque_por_materiais(zstok, materiais_disponiveis):
    zstok_pivot = zstok.pivot_table(
        index="Material", values="Estoque Total", aggfunc="sum", fill_value=0
    ).reset_index()

    zstok_filtrado = zstok_pivot[
        zstok_pivot["Material"].astype(str).str.strip().isin(materiais_disponiveis)
    ]
    return zstok_filtrado.set_index("Material")


def tratar_movimentacoes(zmb51, datas_interesse):
    zmb51["Data de lançamento"] = pd.to_datetime(
        zmb51["Data de lançamento"], errors="coerce"
    )
    zmb51["Mês/Ano"] = zmb51["Data de lançamento"].dt.to_period("M").astype(str)

    zmb51_pivot = zmb51.pivot_table(
        index="Material",
        values="Qtd.  UM registro",
        columns="Data de lançamento",
        aggfunc="sum",
        fill_value=0,
    )

    datas_validas = [d for d in datas_interesse if d in zmb51_pivot.columns]
    if not datas_validas:
        raise ValueError(
            "Nenhuma das datas informadas está presente nas colunas do pivot_table."
        )

    zmb51_pivot = zmb51_pivot[datas_validas]
    zmb51_pivot["Média"] = zmb51_pivot.mean(axis=1)

    return zmb51_pivot[["Média"]].reset_index().set_index("Material")


def tratar_fup(fup):
    fup["Nº do Pedido"] = pd.to_numeric(fup["Nº do Pedido"], errors="coerce")
    fup["TipoMat"] = fup["TipoMat"].astype(str).str.strip().str.upper()

    fup_filtrado = fup[
        (fup["Nº do Pedido"] > 5_000_000_000) & (fup["TipoMat"] == "PCCC")
    ].copy()

    fup_filtrado["Data de Remessa"] = pd.to_datetime(
        fup_filtrado["Data de Remessa"], errors="coerce"
    )

    fup_pivot = fup_filtrado.pivot_table(
        index="Material",
        values="Qtde Pedido",
        columns="Data de Remessa",
        aggfunc="sum",
        fill_value=0,
    )

    abril_cols = fup_pivot.columns[fup_pivot.columns.month == 4]
    fup_pivot["Total Abril"] = fup_pivot[abril_cols].sum(axis=1)

    return fup_pivot[["Total Abril"]].reset_index().set_index("Material")


def anexar_fup_ao_pareto(dados_pareto: pd.DataFrame, fup: pd.DataFrame) -> pd.DataFrame:
    """
    Anexa dados do FUP ao DataFrame de Pareto.

    Args:
        dados_pareto: DataFrame com análise de Pareto
        fup: DataFrame com dados do FUP

    Returns:
        DataFrame com dados do FUP anexados
    """
    fup["Data Prev.Entrada"] = pd.to_datetime(fup["Data Prev.Entrada"])
    fup["Mês/Ano"] = fup["Data Prev.Entrada"].dt.strftime("%m/%Y")

    fup_pivotado = fup.pivot_table(
        index="Material",
        columns="Mês/Ano",
        values="Qtde Pedido",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    if "Material" not in dados_pareto.columns:
        dados_pareto = dados_pareto.reset_index()

    return dados_pareto.merge(fup_pivotado, on="Material", how="left")


def anexar_mos_ao_pareto(
    dados_pareto: pd.DataFrame, mos_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Anexa dados do MOS ao DataFrame de Pareto.

    Args:
        dados_pareto: DataFrame com análise de Pareto
        mos_df: DataFrame com dados do MOS

    Returns:
        DataFrame com dados do MOS anexados
    """
    return dados_pareto.merge(mos_df, on="Material", how="left")


def calcular_mos(
    dispon: np.ndarray,
    zstok_pivot: pd.DataFrame,
    zmb51_pivot: pd.DataFrame,
    fup_pivot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula o MOS (Months of Stock) para os materiais.

    Args:
        dispon: Array com materiais disponíveis
        zstok_pivot: DataFrame pivotado do ZSTOK
        zmb51_pivot: DataFrame pivotado do ZMB51
        fup_pivot: DataFrame pivotado do FUP

    Returns:
        DataFrame com cálculo do MOS
    """
    dispon_df = pd.DataFrame(dispon, columns=["Material"])

    mos = (
        dispon_df.merge(zstok_pivot, on="Material", how="left")
        .merge(zmb51_pivot, on="Material", how="left")
        .merge(fup_pivot, on="Material", how="left")
    )

    mos = mos.dropna(subset=["Estoque Total", "Média", "Total Abril"], how="all")
    mos["Estoque Total"] = mos["Estoque Total"].fillna(0)
    mos["Total Abril"] = mos["Total Abril"].fillna(0)
    mos["Soma Total"] = mos["Estoque Total"] + mos["Total Abril"]

    mos["Mos"] = np.where(
        mos["Média"] == 0, mos["Soma Total"], mos["Soma Total"] / mos["Média"]
    )

    df_resultado = (
        mos[mos["Média"] != 0][["Material", "Mos"]]
        .query("Mos != 0")
        .dropna()
        .sort_values(by="Mos", ascending=True)
    )

    return df_resultado


def carregar_dados_dispon_zstok_zmb51_fup(
    caminho_dispon: str,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega dados do PostgreSQL e arquivo Excel de disponibilidade.

    Args:
        caminho_dispon: Caminho para o arquivo de disponibilidade

    Returns:
        Tuple contendo dados de disponibilidade, ZSTOK, ZMB51 e FUP

    Raises:
        Exception: Se não for possível conectar ao PostgreSQL
    """
    engine = conectar_postgres()
    if engine is None:
        raise Exception("Não foi possível conectar ao PostgreSQL")

    dispon = pd.read_excel(caminho_dispon, skiprows=2)
    dispon = dispon["material"].dropna().astype(str).str.strip().unique()

    zstok = pd.read_sql("SELECT * FROM zstok", engine)
    zmb51 = pd.read_sql("SELECT * FROM zmb51", engine)
    fup = pd.read_sql("SELECT * FROM fup", engine)

    return dispon, zstok, zmb51, fup


def classificar_abc(pct_acumulado: float) -> str:
    """
    Classifica itens em categorias ABC com base no percentual acumulado.

    Args:
        pct_acumulado: Percentual acumulado do item

    Returns:
        Classificação A, B ou C
    """
    if pct_acumulado <= 80:
        return "A"
    elif pct_acumulado <= 95:
        return "B"
    return "C"


def classificar_xyz(cv: float) -> str:
    """
    Classifica itens em categorias XYZ com base no coeficiente de variação.

    Args:
        cv: Coeficiente de variação

    Returns:
        Classificação X, Y ou Z
    """
    if cv <= 0.5:
        return "X"
    elif cv <= 1.0:
        return "Y"
    return "Z"


def calcular_classe_xyz(mapa: pd.DataFrame, dados_pareto: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula classificação XYZ para os materiais.

    Args:
        mapa: DataFrame com dados do mapa
        dados_pareto: DataFrame com análise de Pareto

    Returns:
        DataFrame com classificação XYZ adicionada
    """
    colunas_saidas = [
        "04/2024",
        "05/2024",
        "06/2024",
        "07/2024",
        "08/2024",
        "09/2024",
        "10/2024",
        "11/2024",
        "12/2024",
        "01/2025",
        "02/2025",
        "03/2025",
    ]

    colunas_existentes = [col for col in colunas_saidas if col in mapa.columns]

    mapa["Média"] = mapa[colunas_existentes].mean(axis=1)
    mapa["Desvio Padrão"] = mapa[colunas_existentes].std(axis=1)
    mapa["CV"] = mapa["Desvio Padrão"] / mapa["Média"]

    return dados_pareto.merge(
        mapa[["Material", "Média", "Desvio Padrão", "CV"]], on="Material", how="left"
    ).assign(Classe_XYZ=lambda x: x["CV"].apply(classificar_xyz))


def plotar_materiais_com_baixo_mos(
    df_resultado: pd.DataFrame, limite: float = 4
) -> None:
    """
    Plota gráfico de barras para materiais com MOS abaixo do limite.

    Args:
        df_resultado: DataFrame com resultados do MOS
        limite: Limite superior para filtrar materiais
    """
    df_filtrado = df_resultado[df_resultado["Mos"] < limite]

    plt.figure(figsize=(20, 10))
    bars = plt.bar(df_filtrado["Material"], df_filtrado["Mos"])

    plt.xlabel("Material")
    plt.ylabel("Mos")
    plt.title(f"Materiais com Mos abaixo de {limite}")
    plt.xticks(rotation=45)

    for bar in bars:
        altura = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            altura + 0.05,
            f"{altura:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def main():
    """
    Função principal que executa a análise completa.
    """
    # Carrega dados iniciais
    dispon, zstok, zmb51, fup = carregar_dados_dispon_zstok_zmb51_fup(CAMINHO_DISPON)
    psi, mapa = carregar_dados(CAMINHO_PSI, CAMINHO_MAPA)

    # Processa dados para análise ABC/XYZ
    dados_tratados = tratar_dados(psi, mapa)
    dados_pareto = gerar_pareto(dados_tratados)
    dados_pareto["Classe ABC"] = dados_pareto["Cumulative %"].apply(classificar_abc)
    dados_pareto = calcular_classe_xyz(mapa, dados_pareto)

    # Visualização dos resultados
    plotar_pareto_customizado(dados_pareto)
    dados_pareto = anexar_fup_ao_pareto(dados_pareto, fup)

    # Análise por classe
    classe_xa = filtrar_por_classes(dados_pareto, classe_xyz="X", classe_abc="A")
    classe_ya = filtrar_por_classes(dados_pareto, classe_xyz="Y", classe_abc="A")
    classe_b = filtrar_por_classes(dados_pareto, classe_abc="B")

    # Análise MOS
    zstok_filtrado = filtrar_estoque_por_materiais(zstok, dispon)
    datas_interesse = (
        pd.to_datetime(zmb51["Data de lançamento"], errors="coerce")
        .dropna()
        .sort_values(ascending=False)
        .unique()[:6]
    )
    datas_interesse = sorted(datas_interesse)

    zmb51_tratado = tratar_movimentacoes(zmb51, datas_interesse)
    fup_tratado = tratar_fup(fup)

    mos_df = calcular_mos(
        dispon, zstok_filtrado.reset_index(), zmb51_tratado.reset_index(), fup_tratado
    )

    dados_pareto = anexar_mos_ao_pareto(dados_pareto, mos_df)
    plotar_materiais_com_baixo_mos(mos_df)
    plotar_pareto_top_n(dados_pareto, top_n=50)

    # Salva resultados
    dados_pareto.to_excel("data/dados_pareto.xlsx")

    return dados_pareto, classe_xa, classe_ya, classe_b


if __name__ == "__main__":
    dados_pareto, classe_xa, classe_ya, classe_b = main()
