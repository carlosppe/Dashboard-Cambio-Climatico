# -*- coding: utf-8 -*-
"""
Dashboard Interactivo - Ejecución Presupuestal (Versión 3 - Optimizado)
Polars + DuckDB persistente + caché agresivo + WebGL rendering.
"""

import streamlit as st
import polars as pl
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard Presupuestal - DEA (v3)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size:2rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:1rem}
    .metric-card {background-color:#f0f2f6;border-radius:10px;padding:15px;text-align:center}
    .stDataFrame {font-size:12px}
    div[data-testid="stMetricValue"] {font-size:1.5rem}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================
PARQUET_PATH = "https://huggingface.co/datasets/carlospurizaca/PresupuestoCambioClimatico/resolve/main/Cambio_Climatico_Unificado.parquet?download=1"
# Token de Hugging Face (lo obtienes desde Streamlit Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Headers para autenticación si el dataset es privado
HF_HEADERS = (
    f"Authorization: Bearer {HF_TOKEN}"
    if HF_TOKEN else None
)

COLS_NECESARIAS = [
    'ANO_EJE', 'NIVEL_GOBIERNO_NOMBRE',
    'SECTOR_NOMBRE', 'PLIEGO_NOMBRE',
    'DEPARTAMENTO_EJECUTORA_NOMBRE', 'PROVINCIA_EJECUTORA_NOMBRE',
    'DISTRITO_EJECUTORA_NOMBRE',
    'TIPO_RECURSO_NOMBRE', 'CATEGORIA_GASTO_NOMBRE', 'TIPO_ACT_PROY_NOMBRE',
    'PRODUCTO_PROYECTO_NOMBRE', 'FUNCION_NOMBRE',
    'DIVISION_FUNCIONAL_NOMBRE', 'GRUPO_FUNCIONAL_NOMBRE',
    'MONTO_PIA', 'MONTO_PIM', 'MONTO_DEVENGADO', 'MONTO_CERTIFICADO',
    'MONTO_COMPROMETIDO', 'MONTO_COMPROMETIDO_ANUAL', 'MONTO_GIRADO',
    'Pr_Py_Nombre2',
]

COLS_MONTOS = [
    'MONTO_PIA', 'MONTO_PIM', 'MONTO_DEVENGADO', 'MONTO_CERTIFICADO',
    'MONTO_COMPROMETIDO', 'MONTO_COMPROMETIDO_ANUAL', 'MONTO_GIRADO',
]

TABLE_FORMAT = {
    'NumProy': '{:,.0f}', 'NumProy+': '{:,.0f}', 'NumProy-': '{:,.0f}',
    'M_PIA': '{:,.2f}', 'M_PIM': '{:,.2f}',
    'Suma de MONTO_DEVENGADO': '{:,.2f}', 'M_Devengado': '{:,.2f}',
    'M_Comprometido_Anual': '{:,.2f}', 'M_Certificado': '{:,.2f}',
    'R_Dev-PIM': '{:.2f}%', 'R_Dev-Certif': '{:.2f}%',
    'R_Dev-CompAnual': '{:.2f}%', 'R_PIM-PIA': '{:.2f}',
}

COLORES_BOX = [
    '#F8766D', '#00BA38', '#619CFF', '#C77CFF', '#FF9F1C',
    '#2EC4B6', '#E63946', '#457B9D', '#F4A261', '#6A4C93',
    '#1982C4', '#8AC926',
]
SIMBOLOS_BOX = [
    'circle', 'triangle-up', 'square', 'diamond', 'cross',
    'x', 'star', 'pentagon', 'hexagon', 'circle-open',
    'triangle-up-open', 'square-open',
]


# =============================================================================
# CAPA DE DATOS – Conexión DuckDB persistente (se crea UNA sola vez)
# =============================================================================
@st.cache_data
def _cargar_parquet():
    """Carga Parquet desde Hugging Face o ruta local, con columnas necesarias."""

    # --- Caso 1: archivo remoto (URL Hugging Face) ---
    if isinstance(PARQUET_PATH, str) and PARQUET_PATH.startswith("http"):
        lf = pl.scan_parquet(
            PARQUET_PATH,
            storage_options={"headers": HF_HEADERS} if HF_HEADERS else None
        )
        cols = [c for c in COLS_NECESARIAS if c in lf.columns]
        lf = lf.select(cols)

    # --- Caso 2: archivo local (solo desarrollo) ---
    else:
        p = Path(PARQUET_PATH)
        if not p.exists():
            st.error("Archivo Parquet no encontrado en ruta local.")
            return pl.DataFrame()

        lf = pl.scan_parquet(p)
        cols = [c for c in COLS_NECESARIAS if c in lf.columns]
        lf = lf.select(cols)

    # --- Casting de columnas ---
    cast = []
    for c in COLS_MONTOS:
        if c in cols:
            cast.append(pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0))

    if 'ANO_EJE' in cols:
        cast.append(pl.col('ANO_EJE').cast(pl.Int32, strict=False))

    if cast:
        lf = lf.with_columns(cast)

    return lf.collect()

@st.cache_resource
def _init_db():
    """Conexión DuckDB persistente con datos registrados UNA vez."""
    df = _cargar_parquet()
    if df is None:
        return None, None, 0
    con = duckdb.connect()
    con.register('datos', df)
    # Retornar df para mantener referencia viva (evitar GC) 
    return con, df, df.shape[0]


def _sql(query: str):
    """Ejecuta SQL sobre el DuckDB persistente."""
    con, _, _ = _init_db()
    return con.sql(query)


# =============================================================================
# HELPERS SQL – construcción segura de cláusulas WHERE
# =============================================================================
def _esc(val) -> str:
    """Escape de comillas simples para literales SQL."""
    return str(val).replace("'", "''") if val is not None else ''


def _esc_like(val) -> str:
    """Escape para patrones ILIKE."""
    s = str(val).replace("'", "''")
    s = s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return s


def _build_where(parts: list) -> str:
    """Convierte lista de condiciones en cláusula WHERE."""
    return " AND ".join(parts) if parts else "1=1"


def _opciones(col: str, where: str, all_label: str = 'Todas') -> list:
    """Valores únicos vía DuckDB – mucho más rápido que cargar a Polars."""
    rows = _sql(
        f'SELECT DISTINCT "{col}" FROM datos '
        f'WHERE {where} AND "{col}" IS NOT NULL ORDER BY "{col}"'
    ).fetchall()
    return [all_label] + [r[0] for r in rows]


# =============================================================================
# FUNCIONES DE CÓMPUTO (todas con @st.cache_data, clave = where string)
# =============================================================================
@st.cache_data(ttl=300)
def calcular_medidas(where: str) -> dict:
    """KPIs principales – una sola query DuckDB."""
    row = _sql(f"""
        SELECT
            SUM(MONTO_PIA), SUM(MONTO_PIM), SUM(MONTO_DEVENGADO),
            SUM(MONTO_CERTIFICADO), SUM(MONTO_COMPROMETIDO_ANUAL),
            SUM(MONTO_GIRADO),
            COUNT(DISTINCT CASE WHEN MONTO_PIA>0 OR MONTO_PIM>0 OR MONTO_DEVENGADO>0
                  THEN Pr_Py_Nombre2 END),
            COUNT(DISTINCT CASE WHEN MONTO_DEVENGADO>0 THEN Pr_Py_Nombre2 END)
        FROM datos WHERE {where}
    """).fetchone()
    vals = [v or 0 for v in row]
    M_PIA, M_PIM, M_Dev, M_Cert, M_CA, M_Gir, NumProy, NumProyP = vals
    return {
        'NumProy': NumProy, 'NumProy+': NumProyP, 'NumProy-': NumProy - NumProyP,
        'M_PIA': M_PIA, 'M_PIM': M_PIM,
        'Suma de MONTO_DEVENGADO': M_Dev, 'M_Devengado': M_Dev,
        'M_Comprometido_Anual': M_CA, 'M_Certificado': M_Cert,
        'R_Dev-PIM': (M_Dev / M_PIM * 100) if M_PIM > 0 else 0,
        'R_Dev-Certif': (M_Dev / M_Cert * 100) if M_Cert > 0 else 0,
        'R_Dev-CompAnual': (M_Dev / M_CA * 100) if M_CA > 0 else 0,
        'R_PIM-PIA': (M_PIM / M_PIA) if M_PIA > 0 else 0,
    }


@st.cache_data(ttl=300)
def crear_tabla_agrupada(where: str, grupo_col: str) -> pd.DataFrame:
    """Tabla agrupada con DuckDB – una query GROUP BY + fila Total."""
    df_g = _sql(f"""
        SELECT
            "{grupo_col}",
            COUNT(DISTINCT CASE WHEN MONTO_PIA>0 OR MONTO_PIM>0 OR MONTO_DEVENGADO>0
                  THEN Pr_Py_Nombre2 END) AS "NumProy",
            COUNT(DISTINCT CASE WHEN MONTO_DEVENGADO>0
                  THEN Pr_Py_Nombre2 END) AS "NumProy+",
            COUNT(DISTINCT CASE WHEN (MONTO_PIA>0 OR MONTO_PIM>0 OR MONTO_DEVENGADO>0)
                  AND MONTO_DEVENGADO<=0 THEN Pr_Py_Nombre2 END) AS "NumProy-",
            SUM(MONTO_PIA) AS "M_PIA",
            SUM(MONTO_PIM) AS "M_PIM",
            SUM(MONTO_DEVENGADO) AS "Suma de MONTO_DEVENGADO",
            SUM(MONTO_DEVENGADO) AS "M_Devengado",
            SUM(MONTO_COMPROMETIDO_ANUAL) AS "M_Comprometido_Anual",
            SUM(MONTO_CERTIFICADO) AS "M_Certificado",
            CASE WHEN SUM(MONTO_PIM)>0 THEN SUM(MONTO_DEVENGADO)/SUM(MONTO_PIM)*100 ELSE 0 END AS "R_Dev-PIM",
            CASE WHEN SUM(MONTO_CERTIFICADO)>0 THEN SUM(MONTO_DEVENGADO)/SUM(MONTO_CERTIFICADO)*100 ELSE 0 END AS "R_Dev-Certif",
            CASE WHEN SUM(MONTO_COMPROMETIDO_ANUAL)>0 THEN SUM(MONTO_DEVENGADO)/SUM(MONTO_COMPROMETIDO_ANUAL)*100 ELSE 0 END AS "R_Dev-CompAnual",
            CASE WHEN SUM(MONTO_PIA)>0 THEN SUM(MONTO_PIM)/SUM(MONTO_PIA) ELSE 0 END AS "R_PIM-PIA"
        FROM datos WHERE {where}
        GROUP BY "{grupo_col}"
    """).df()
    total = calcular_medidas(where)
    total[grupo_col] = 'Total'
    df_all = pd.concat([df_g, pd.DataFrame([total])], ignore_index=True)
    cols_ord = [grupo_col, 'NumProy', 'NumProy+', 'NumProy-', 'M_PIA', 'M_PIM',
                'Suma de MONTO_DEVENGADO', 'M_Devengado', 'M_Comprometido_Anual',
                'M_Certificado', 'R_Dev-PIM', 'R_Dev-Certif', 'R_Dev-CompAnual', 'R_PIM-PIA']
    cols_d = [c for c in cols_ord if c in df_all.columns]
    return df_all[cols_d].sort_values('M_Devengado', ascending=False)


@st.cache_data(ttl=300)
def obtener_evolucion_anual(where: str) -> pd.DataFrame:
    return _sql(f"""
        SELECT ANO_EJE,
            COUNT(DISTINCT CASE WHEN MONTO_PIA>0 OR MONTO_PIM>0 OR MONTO_DEVENGADO>0
                  THEN Pr_Py_Nombre2 END) AS NumProy,
            COUNT(DISTINCT CASE WHEN MONTO_DEVENGADO>0 THEN Pr_Py_Nombre2 END) AS "NumProy+",
            SUM(MONTO_PIA) AS M_PIA,
            SUM(MONTO_PIM) AS M_PIM,
            SUM(MONTO_DEVENGADO) AS M_Devengado,
            CASE WHEN SUM(MONTO_PIM)>0 THEN SUM(MONTO_DEVENGADO)/SUM(MONTO_PIM)*100 ELSE 0 END AS "R_Dev-PIM"
        FROM datos WHERE {where} AND ANO_EJE BETWEEN 2000 AND 2100
        GROUP BY ANO_EJE ORDER BY ANO_EJE
    """).df()


@st.cache_data(ttl=300)
def obtener_heatmap_depto(where: str) -> pd.DataFrame:
    return _sql(f"""
        SELECT DEPARTAMENTO_EJECUTORA_NOMBRE, ANO_EJE, SUM(MONTO_DEVENGADO) AS DEV
        FROM datos WHERE {where} GROUP BY 1, 2
    """).df()


@st.cache_data(ttl=300)
def _obtener_base_proyectos(where: str) -> pl.DataFrame:
    """Base año-proyecto (consistente con NumProy) – se reutiliza para stats y boxplot."""
    cols_m = COLS_MONTOS
    sums = ', '.join([f'SUM(f."{c}") AS "{c}"' for c in cols_m])
    return _sql(f"""
        WITH filtered AS (
            SELECT * FROM datos WHERE {where}
        ),
        valid AS (
            SELECT DISTINCT ANO_EJE, Pr_Py_Nombre2
            FROM filtered
            WHERE (MONTO_PIA>0 OR MONTO_PIM>0 OR MONTO_DEVENGADO>0)
              AND ANO_EJE BETWEEN 2000 AND 2100
        )
        SELECT f.ANO_EJE, f.Pr_Py_Nombre2, {sums}
        FROM filtered f
        INNER JOIN valid v ON f.ANO_EJE = v.ANO_EJE AND f.Pr_Py_Nombre2 = v.Pr_Py_Nombre2
        WHERE f.ANO_EJE BETWEEN 2000 AND 2100
        GROUP BY f.ANO_EJE, f.Pr_Py_Nombre2
    """).pl()


@st.cache_data(ttl=300)
def _obtener_sumas_raw(where: str) -> dict:
    """Sumas crudas por año (para la columna Suma del panel de stats)."""
    cols_m = COLS_MONTOS
    sums_q = _sql(f"""
        SELECT ANO_EJE, {', '.join([f'SUM("{c}") AS "{c}"' for c in cols_m])}
        FROM datos WHERE {where} AND ANO_EJE BETWEEN 2000 AND 2100
        GROUP BY ANO_EJE
    """).pl()
    return {r['ANO_EJE']: r for r in sums_q.iter_rows(named=True)}


@st.cache_data(ttl=300)
def calcular_estadisticas(where: str) -> pd.DataFrame:
    """Estadísticas descriptivas por año + variable."""
    base = _obtener_base_proyectos(where)
    if base.is_empty():
        return pd.DataFrame()
    sumas_dict = _obtener_sumas_raw(where)

    resultados = []
    for ano in sorted(base['ANO_EJE'].unique().to_list()):
        if not (2000 <= ano <= 2100):
            continue
        bloque = base.filter(pl.col('ANO_EJE') == ano)
        n_proy = bloque['Pr_Py_Nombre2'].n_unique()

        for col in COLS_MONTOS:
            arr = bloque[col].to_numpy().astype(np.float64)
            arr = np.nan_to_num(arr, 0.0)
            no_cero = arr[arr > 0]
            suma = sumas_dict.get(ano, {}).get(col, 0) or 0
            media = float(np.mean(arr)) if len(arr) else 0
            mediana = float(np.median(arr)) if len(arr) else 0
            desv = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
            cv = (desv / media * 100) if media != 0 else np.nan
            if len(arr) >= 4:
                m, s = np.mean(arr), np.std(arr, ddof=1)
                if s > 0:
                    n = len(arr)
                    kurt = (n*(n+1))/((n-1)*(n-2)*(n-3))*np.sum(((arr-m)/s)**4) \
                           - 3*(n-1)**2/((n-2)*(n-3))
                else:
                    kurt = np.nan
            else:
                kurt = np.nan
            q1 = float(np.percentile(arr, 25)) if len(arr) else 0
            q3 = float(np.percentile(arr, 75)) if len(arr) else 0
            mn = float(np.min(arr)) if len(arr) else 0
            mx = float(np.max(arr)) if len(arr) else 0
            resultados.append({
                'Año': int(ano), 'Variable': col.replace('MONTO_', ''),
                'NumProy (base)': n_proy, 'Conteo': n_proy,
                'Conteo No Cero': len(no_cero),
                'Media': media, 'Mediana': mediana,
                'Desv.Est.': desv, 'Coef. Variación': cv, 'Kurtosis': kurt,
                'Mín': mn, 'Q1': q1, 'Q2': mediana, 'Q3': q3, 'Máx': mx,
                'Suma': suma, 'Rango': mx - mn,
            })
    return pd.DataFrame(resultados)


# =============================================================================
# HELPERS UI
# =============================================================================
def formatear_numero(valor, tipo='moneda'):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    if tipo == 'moneda':
        if abs(valor) >= 1e9:
            return f"S/ {valor/1e9:,.2f} B"
        elif abs(valor) >= 1e6:
            return f"S/ {valor/1e6:,.2f} M"
        return f"S/ {valor:,.2f}"
    elif tipo == 'porcentaje':
        return f"{valor:.2f}%"
    elif tipo == 'entero':
        return f"{int(valor):,}"
    return str(valor)


def mostrar_tabla(df_pd: pd.DataFrame, height: int = 500):
    fmt = {k: v for k, v in TABLE_FORMAT.items() if k in df_pd.columns}
    st.dataframe(df_pd.style.format(fmt, na_rep=''), use_container_width=True, height=height)


def chart_bar_top(df_pd: pd.DataFrame, dim_col: str, title: str, top_n: int = 10):
    df_top = df_pd[df_pd[dim_col] != 'Total'].nlargest(top_n, 'M_Devengado')
    fig = px.bar(df_top, x='M_Devengado', y=dim_col, orientation='h',
                 color='R_Dev-PIM', color_continuous_scale='RdYlGn',
                 labels={'M_Devengado': 'Monto Devengado (S/)', dim_col: dim_col.replace('_NOMBRE', '')},
                 title=title)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      height=max(400, top_n * 30))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    st.markdown('<h1 class="main-header">📊 Dashboard de Ejecución Presupuestal (v3)</h1>',
                unsafe_allow_html=True)

    con, _, n_rows = _init_db()
    if con is None:
        st.error("❌ No se encontró el Parquet. Ejecuta primero csv_a_parquet.py")
        return

    st.success(f"✅ Datos cargados: {n_rows:,} registros  (Polars + DuckDB persistente)")

    # =========================================================================
    # FILTROS LATERALES – construyen WHERE clause incrementalmente
    # =========================================================================
    st.sidebar.header("🔍 Filtros")

    wp = ["1=1"]  # where parts

    anos_disp = [r[0] for r in _sql(
        "SELECT DISTINCT ANO_EJE FROM datos WHERE ANO_EJE IS NOT NULL ORDER BY ANO_EJE"
    ).fetchall()]
    anos_sel = st.sidebar.multiselect("ANO_EJE", anos_disp, default=anos_disp,
                                       help="Seleccione los años a visualizar")
    if anos_sel:
        wp.append(f"ANO_EJE IN ({','.join(str(int(a)) for a in anos_sel)})")

    w = _build_where(wp)
    nivel_sel = st.sidebar.selectbox("NIVEL_GOBIERNO_NOMBRE",
                                      _opciones('NIVEL_GOBIERNO_NOMBRE', w))
    if nivel_sel != 'Todas':
        wp.append(f"\"NIVEL_GOBIERNO_NOMBRE\" = '{_esc(nivel_sel)}'")

    w = _build_where(wp)
    depto_sel = st.sidebar.selectbox("DEPARTAMENTO_EJECUTORA",
                                      _opciones('DEPARTAMENTO_EJECUTORA_NOMBRE', w))
    if depto_sel != 'Todas':
        wp.append(f"\"DEPARTAMENTO_EJECUTORA_NOMBRE\" = '{_esc(depto_sel)}'")

    w = _build_where(wp)
    prov_sel = st.sidebar.selectbox("PROVINCIA_EJECUTORA_N",
                                     _opciones('PROVINCIA_EJECUTORA_NOMBRE', w))
    if prov_sel != 'Todas':
        wp.append(f"\"PROVINCIA_EJECUTORA_NOMBRE\" = '{_esc(prov_sel)}'")

    w = _build_where(wp)
    dist_sel = st.sidebar.selectbox("DISTRITO_EJECUTORA_NO",
                                     _opciones('DISTRITO_EJECUTORA_NOMBRE', w))
    if dist_sel != 'Todas':
        wp.append(f"\"DISTRITO_EJECUTORA_NOMBRE\" = '{_esc(dist_sel)}'")

    w = _build_where(wp)
    tr_sel = st.sidebar.selectbox("TIPO_RECURSO_NOM",
                                   _opciones('TIPO_RECURSO_NOMBRE', w))
    if tr_sel != 'Todas':
        wp.append(f"\"TIPO_RECURSO_NOMBRE\" = '{_esc(tr_sel)}'")

    w = _build_where(wp)
    cg_sel = st.sidebar.selectbox("CATEGORIA_GASTO_NOMBRE",
                                   _opciones('CATEGORIA_GASTO_NOMBRE', w))
    if cg_sel != 'Todas':
        wp.append(f"\"CATEGORIA_GASTO_NOMBRE\" = '{_esc(cg_sel)}'")

    w = _build_where(wp)
    tap_sel = st.sidebar.selectbox("TIPO_ACT_PROY_NOMBRE",
                                    _opciones('TIPO_ACT_PROY_NOMBRE', w))
    if tap_sel != 'Todas':
        wp.append(f"\"TIPO_ACT_PROY_NOMBRE\" = '{_esc(tap_sel)}'")

    prod_busq = st.sidebar.text_input("🔎 Buscar PRODUCTO_PROYECTO", "")
    if prod_busq.strip():
        wp.append(f"\"PRODUCTO_PROYECTO_NOMBRE\" ILIKE '%{_esc_like(prod_busq.strip())}%'")

    sidebar_where = _build_where(wp)

    n_filt = _sql(f"SELECT COUNT(*) FROM datos WHERE {sidebar_where}").fetchone()[0]
    st.sidebar.markdown("---")
    st.sidebar.info(f"📝 Registros filtrados: {n_filt:,}")

    # =========================================================================
    # PESTAÑAS PRINCIPALES
    # =========================================================================
    tab_principal, tab_estadisticas = st.tabs(
        ["📊 Dashboard Principal", "📈 Análisis Estadístico"])

    # =========================================================================
    # PESTAÑA 1: DASHBOARD PRINCIPAL
    # =========================================================================
    with tab_principal:
        st.markdown("---")
        st.markdown("### 📈 Indicadores Principales")

        with st.spinner("Calculando indicadores…"):
            medidas = calcular_medidas(sidebar_where)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("NumProy", f"{medidas['NumProy']:,}")
        c2.metric("NumProy+", f"{medidas['NumProy+']:,}")
        c3.metric("NumProy-", f"{medidas['NumProy-']:,}")
        c4.metric("M_PIA", formatear_numero(medidas['M_PIA']))
        c5.metric("M_PIM", formatear_numero(medidas['M_PIM']))
        c6.metric("M_Devengado", formatear_numero(medidas['M_Devengado']))
        c7.metric("R_Dev-PIM", f"{medidas['R_Dev-PIM']:.1f}%")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Por Sector", "📊 Por Pliego", "📈 Evolución Anual",
            "🗺️ Por Departamento", "🔧 Por Grupo Funcional", "🎯 Por Función"])

        # --- Sector ---
        with tab1:
            st.markdown("### Análisis por SECTOR_NOMBRE")
            with st.spinner("Procesando…"):
                df_sec = crear_tabla_agrupada(sidebar_where, 'SECTOR_NOMBRE')
            if not df_sec.empty:
                mostrar_tabla(df_sec)
                st.markdown("#### Top 10 Sectores por Devengado")
                chart_bar_top(df_sec, 'SECTOR_NOMBRE', 'Top 10 Sectores - Monto Devengado', 10)

        # --- Pliego ---
        with tab2:
            st.markdown("### Análisis por PLIEGO_NOMBRE")
            sectores_pli = _opciones('SECTOR_NOMBRE', sidebar_where, 'Todos')
            sector_sel = st.selectbox("Filtrar por Sector:", sectores_pli, key='sector_pliego')
            where_pli = sidebar_where
            if sector_sel != 'Todos':
                where_pli = sidebar_where + f" AND \"SECTOR_NOMBRE\" = '{_esc(sector_sel)}'"
            with st.spinner("Procesando…"):
                df_pliego = crear_tabla_agrupada(where_pli, 'PLIEGO_NOMBRE')
            if not df_pliego.empty:
                mostrar_tabla(df_pliego)

        # --- Evolución Anual ---
        with tab3:
            st.markdown("### Evolución Anual")
            with st.spinner("Procesando…"):
                df_anual = obtener_evolucion_anual(sidebar_where)

            if not df_anual.empty:
                st.dataframe(df_anual.style.format({
                    'ANO_EJE': '{:.0f}', 'NumProy': '{:,.0f}', 'NumProy+': '{:,.0f}',
                    'M_PIA': '{:,.2f}', 'M_PIM': '{:,.2f}', 'M_Devengado': '{:,.2f}',
                    'R_Dev-PIM': '{:.2f}%'
                }, na_rep=''), use_container_width=True)

                x_a = df_anual['ANO_EJE'].astype(int).astype(str)
                ax_cat = dict(type='category', categoryorder='array', categoryarray=x_a.tolist())

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(name='M_PIA', x=x_a, y=df_anual['M_PIA'], marker_color='#636EFA'))
                    fig1.add_trace(go.Bar(name='M_PIM', x=x_a, y=df_anual['M_PIM'], marker_color='#EF553B'))
                    fig1.add_trace(go.Bar(name='M_Devengado', x=x_a, y=df_anual['M_Devengado'], marker_color='#00CC96'))
                    fig1.update_layout(title='PIA, PIM y Devengado por Año', barmode='group',
                                       xaxis_title='Año', yaxis_title='Monto (S/)', xaxis=ax_cat)
                    st.plotly_chart(fig1, use_container_width=True)
                with col_e2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=x_a, y=df_anual['R_Dev-PIM'],
                                              mode='lines+markers', name='R_Dev-PIM',
                                              line=dict(color='#AB63FA', width=3)))
                    fig2.update_layout(title='Ratio Devengado/PIM', xaxis_title='Año',
                                       yaxis_title='%', yaxis=dict(range=[0, 100]), xaxis=ax_cat)
                    st.plotly_chart(fig2, use_container_width=True)

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=x_a, y=df_anual['NumProy'],
                    mode='lines+markers+text', name='NumProy',
                    text=df_anual['NumProy'].apply(lambda v: f'{int(v):,}'),
                    textposition='top center', line=dict(color='#19D3F3', width=2)))
                fig3.update_layout(title='Número de Proyectos por Año',
                                   xaxis_title='Año', yaxis_title='Proyectos', xaxis=ax_cat)
                st.plotly_chart(fig3, use_container_width=True)

        # --- Departamento ---
        with tab4:
            st.markdown("### Análisis por Departamento")
            with st.spinner("Procesando…"):
                df_dep = crear_tabla_agrupada(sidebar_where, 'DEPARTAMENTO_EJECUTORA_NOMBRE')
            if not df_dep.empty:
                mostrar_tabla(df_dep)
                st.markdown("#### Mapa de Calor: Devengado por Departamento y Año")
                with st.spinner("Generando mapa…"):
                    pivot_df = obtener_heatmap_depto(sidebar_where)
                if not pivot_df.empty:
                    pt = pivot_df.pivot(index='DEPARTAMENTO_EJECUTORA_NOMBRE',
                                        columns='ANO_EJE', values='DEV').fillna(0)
                    fig = px.imshow(pt, labels=dict(x="Año", y="Departamento", color="Devengado"),
                                    aspect="auto", color_continuous_scale="YlOrRd")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

        # --- Grupo Funcional ---
        with tab5:
            st.markdown("### Análisis por Grupo Funcional")
            with st.spinner("Procesando…"):
                df_gf = crear_tabla_agrupada(sidebar_where, 'GRUPO_FUNCIONAL_NOMBRE')
            if not df_gf.empty:
                mostrar_tabla(df_gf)
                st.markdown("#### Top 15 Grupos Funcionales por Devengado")
                chart_bar_top(df_gf, 'GRUPO_FUNCIONAL_NOMBRE',
                              'Top 15 Grupos Funcionales - Monto Devengado', 15)

        # --- Función ---
        with tab6:
            st.markdown("### Análisis por Función")
            with st.spinner("Procesando…"):
                df_fn = crear_tabla_agrupada(sidebar_where, 'FUNCION_NOMBRE')
            if not df_fn.empty:
                mostrar_tabla(df_fn)
                st.markdown("#### Top 15 Funciones por Devengado")
                chart_bar_top(df_fn, 'FUNCION_NOMBRE',
                              'Top 15 Funciones - Monto Devengado', 15)

    # =========================================================================
    # PESTAÑA 2: ANÁLISIS ESTADÍSTICO
    # =========================================================================
    with tab_estadisticas:
        st.markdown("### Filtros de Análisis Estadístico")

        # Cascada de filtros — cada dropdown usa DuckDB con WHERE progresivo
        sp = list(wp)  # copia de los filtros del sidebar

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            sector_est = st.selectbox("Sector",
                _opciones('SECTOR_NOMBRE', _build_where(sp), 'Todos'), key='sector_est')
        if sector_est != 'Todos':
            sp.append(f"\"SECTOR_NOMBRE\" = '{_esc(sector_est)}'")

        with col_f2:
            funcion_est = st.selectbox("Función",
                _opciones('FUNCION_NOMBRE', _build_where(sp)), key='func_est')
        if funcion_est != 'Todas':
            sp.append(f"\"FUNCION_NOMBRE\" = '{_esc(funcion_est)}'")

        with col_f3:
            division_est = st.selectbox("División Funcional",
                _opciones('DIVISION_FUNCIONAL_NOMBRE', _build_where(sp)), key='div_func_est')
        if division_est != 'Todas':
            sp.append(f"\"DIVISION_FUNCIONAL_NOMBRE\" = '{_esc(division_est)}'")

        with col_f4:
            grupo_est = st.selectbox("Grupo Funcional",
                _opciones('GRUPO_FUNCIONAL_NOMBRE', _build_where(sp), 'Todos'), key='grupo_est')
        if grupo_est != 'Todos':
            sp.append(f"\"GRUPO_FUNCIONAL_NOMBRE\" = '{_esc(grupo_est)}'")

        stats_where = _build_where(sp)

        st.markdown("---")
        st.markdown("### 📊 Estadísticas Descriptivas por Año")

        with st.spinner("Calculando estadísticas descriptivas…"):
            df_stats = calcular_estadisticas(stats_where)

        if not df_stats.empty:
            variables = sorted(df_stats['Variable'].unique())
            variable_sel = st.selectbox("Seleccionar Variable", variables, key='variable_est')
            df_var = df_stats[df_stats['Variable'] == variable_sel].sort_values('Año')
            df_var = df_var[df_var['Año'].between(2000, 2100)]
            anos_ticks = [str(int(a)) for a in sorted(df_var['Año'].dropna().astype(int).unique())]

            st.markdown(f"#### Estadísticas de {variable_sel}")
            st.caption("ℹ️ **Conteo (NumProy)** = proyectos únicos con PIA>0 ó PIM>0 ó DEVENGADO>0")

            cols_show = ['Año', 'NumProy (base)', 'Conteo No Cero', 'Suma', 'Media', 'Mediana',
                         'Desv.Est.', 'Coef. Variación', 'Kurtosis', 'Mín', 'Q1', 'Q2', 'Q3',
                         'Máx', 'Rango']
            df_disp = df_var[[c for c in cols_show if c in df_var.columns]].copy()
            st.dataframe(df_disp.style.format({
                'NumProy (base)': '{:,.0f}', 'Conteo No Cero': '{:,.0f}',
                'Suma': '{:,.2f}', 'Media': '{:,.2f}', 'Mediana': '{:,.2f}',
                'Desv.Est.': '{:,.2f}', 'Coef. Variación': '{:,.2f}%', 'Kurtosis': '{:,.4f}',
                'Mín': '{:,.2f}', 'Q1': '{:,.2f}', 'Q2': '{:,.2f}', 'Q3': '{:,.2f}',
                'Máx': '{:,.2f}', 'Rango': '{:,.2f}',
            }, na_rep=''), use_container_width=True)

            st.markdown("---")
            ax_cat = dict(type='category', categoryorder='array', categoryarray=anos_ticks)
            x_v = df_var['Año'].astype(int).astype(str)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_mm = go.Figure()
                fig_mm.add_trace(go.Scatter(x=x_v, y=df_var['Media'], mode='lines+markers',
                    name='Media', line=dict(color='#636EFA', width=2), marker=dict(size=8)))
                fig_mm.add_trace(go.Scatter(x=x_v, y=df_var['Mediana'], mode='lines+markers',
                    name='Mediana', line=dict(color='#EF553B', width=2, dash='dash'), marker=dict(size=8)))
                fig_mm.update_layout(title=f'Media y Mediana de {variable_sel}',
                    xaxis_title='Año', yaxis_title='Valor (S/)', hovermode='x unified', xaxis=ax_cat)
                st.plotly_chart(fig_mm, use_container_width=True)
            with col_g2:
                fig_sd = go.Figure()
                fig_sd.add_trace(go.Bar(x=x_v, y=df_var['Desv.Est.'], name='Desv.Est.',
                    marker=dict(color='#00CC96')))
                fig_sd.update_layout(title=f'Desviación Estándar de {variable_sel}',
                    xaxis_title='Año', yaxis_title='Desv. Est. (S/)', xaxis=ax_cat)
                st.plotly_chart(fig_sd, use_container_width=True)

            col_g3, col_g4 = st.columns(2)
            with col_g3:
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(x=x_v, y=df_var['Coef. Variación'],
                    name='CV', marker=dict(color='#FFA600')))
                fig_cv.update_layout(title=f'Coef. Variación de {variable_sel}',
                    xaxis_title='Año', yaxis_title='CV (%)', xaxis=ax_cat)
                st.plotly_chart(fig_cv, use_container_width=True)
            with col_g4:
                fig_k = go.Figure()
                fig_k.add_trace(go.Scatter(x=x_v, y=df_var['Kurtosis'], mode='lines+markers',
                    name='Kurtosis', line=dict(color='#BC5090', width=2), marker=dict(size=8)))
                fig_k.update_layout(title=f'Kurtosis de {variable_sel}',
                    xaxis_title='Año', yaxis_title='Kurtosis', hovermode='x unified', xaxis=ax_cat)
                st.plotly_chart(fig_k, use_container_width=True)

            # ==================================================================
            # BOX PLOT – WebGL (Scattergl) para evitar congelamiento del browser
            # ==================================================================
            st.markdown("#### Gráfico de Cajas (Box Plot) por Año")

            with st.spinner("Preparando datos de distribución…"):
                base_box = _obtener_base_proyectos(stats_where)

            col_var_name = f"MONTO_{variable_sel}"

            if not base_box.is_empty() and col_var_name in base_box.columns:
                anos_box = sorted([int(a) for a in df_var['Año'].dropna().astype(int).unique()
                                   if 2000 <= int(a) <= 2100])

                max_puntos = st.slider("Puntos máximos en Box Plot", 500, 8000, 1500, 500,
                                        key='max_puntos_boxplot')

                # Pre-extraer todos los arrays UNA vez (no 3x por año)
                arrays_by_year = {}
                for ano in anos_box:
                    arr = base_box.filter(pl.col('ANO_EJE') == ano)[col_var_name].to_numpy().astype(np.float64)
                    arrays_by_year[ano] = np.nan_to_num(arr, 0.0)

                fig_box = go.Figure()
                for i, ano in enumerate(anos_box):
                    arr = arrays_by_year[ano]
                    if len(arr) == 0:
                        continue
                    color = COLORES_BOX[i % len(COLORES_BOX)]
                    simbolo = SIMBOLOS_BOX[i % len(SIMBOLOS_BOX)]

                    cuota = max(1, max_puntos // max(len(anos_box), 1))
                    if len(arr) > cuota:
                        display = arr[np.random.default_rng(42).choice(len(arr), cuota, replace=False)]
                    else:
                        display = arr

                    jitter = np.random.default_rng(42 + i).uniform(-0.25, 0.25, len(display))

                    # *** Scattergl = WebGL = NO congela el browser ***
                    fig_box.add_trace(go.Scattergl(
                        x=(i + jitter).tolist(), y=display.tolist(), mode='markers',
                        marker=dict(size=5, color=color, opacity=0.55),
                        name=str(ano), legendgroup=str(ano), showlegend=False,
                        hovertemplate=f'Año: {ano}<br>Valor: %{{y:,.2f}}<extra></extra>'))

                    q1 = float(np.percentile(arr, 25))
                    q2 = float(np.median(arr))
                    q3 = float(np.percentile(arr, 75))
                    iqr = q3 - q1
                    lower = arr[arr >= q1 - 1.5 * iqr]
                    upper = arr[arr <= q3 + 1.5 * iqr]
                    lf = float(lower.min()) if len(lower) else float(arr.min())
                    uf = float(upper.max()) if len(upper) else float(arr.max())

                    fig_box.add_trace(go.Box(
                        x0=i, q1=[q1], median=[q2], q3=[q3],
                        lowerfence=[lf], upperfence=[uf], mean=[float(np.mean(arr))],
                        name=str(ano), legendgroup=str(ano), showlegend=True,
                        boxmean=True, boxpoints=False, width=0.5,
                        line=dict(width=2, color=color), fillcolor='rgba(255,255,255,0.1)'))

                fig_box.update_layout(
                    title=f'Distribución de proyectos - {variable_sel} por Año',
                    xaxis_title='Año', yaxis_title=f'{variable_sel} (S/)',
                    xaxis=dict(tickvals=list(range(len(anos_box))),
                               ticktext=[str(a) for a in anos_box]),
                    legend=dict(title='Año'), height=600)
                st.plotly_chart(fig_box, use_container_width=True)

                # ==============================================================
                # DENSIDAD (reutiliza arrays_by_year)
                # ==============================================================
                st.markdown("#### Gráfico de Densidad por Año")
                fig_dens = go.Figure()
                for i, ano in enumerate(anos_box):
                    arr = arrays_by_year[ano]
                    arr = arr[~np.isnan(arr)]
                    if len(arr) < 2:
                        continue
                    density, bins = np.histogram(arr, bins=50, density=True)
                    centers = (bins[:-1] + bins[1:]) / 2
                    color = COLORES_BOX[i % len(COLORES_BOX)]
                    fig_dens.add_trace(go.Scatter(x=centers, y=density, mode='lines',
                        name=str(ano), line=dict(color=color, width=2),
                        fill='tozeroy', opacity=0.18))
                fig_dens.update_layout(title=f'Densidad de {variable_sel} por Año',
                    xaxis_title=f'{variable_sel} (S/)', yaxis_title='Densidad',
                    hovermode='x unified', height=460)
                st.plotly_chart(fig_dens, use_container_width=True)

                # ==============================================================
                # INTERPRETACIÓN (reutiliza arrays_by_year)
                # ==============================================================
                st.markdown("#### Interpretación Automática de la Densidad")
                interpret_rows = []
                for ano in anos_box:
                    fila = df_var[df_var['Año'] == ano]
                    if fila.empty:
                        continue
                    fila = fila.iloc[0]
                    arr = arrays_by_year[ano]
                    arr_clean = arr[~np.isnan(arr)]
                    if len(arr_clean) == 0:
                        continue

                    dens, bins = np.histogram(arr_clean, bins=60, density=True)
                    pico_x = (bins[:-1] + bins[1:]) / 2
                    pico_principal = float(pico_x[np.argmax(dens)])

                    m = float(np.mean(arr_clean))
                    s = float(np.std(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0
                    if len(arr_clean) > 2 and s > 0:
                        n = len(arr_clean)
                        sk = n / ((n-1)*(n-2)) * np.sum(((arr_clean - m) / s) ** 3)
                    else:
                        sk = np.nan

                    cv_val = float(fila['Coef. Variación']) if pd.notna(fila['Coef. Variación']) else np.nan
                    kurt_val = float(fila['Kurtosis']) if pd.notna(fila['Kurtosis']) else np.nan

                    forma = "No concluyente" if np.isnan(sk) else \
                            "Sesgo a la derecha" if sk > 0.5 else \
                            "Sesgo a la izquierda" if sk < -0.5 else "Aproximadamente simétrica"
                    disp = "No concluyente" if np.isnan(cv_val) else \
                           "Baja dispersión" if cv_val < 30 else \
                           "Dispersión media" if cv_val < 60 else "Alta dispersión"
                    cola = "No concluyente" if np.isnan(kurt_val) else \
                           "Colas pesadas (leptocúrtica)" if kurt_val > 3 else \
                           "Colas ligeras (platicúrtica)" if kurt_val < 3 else "Mesocúrtica"

                    interpret_rows.append({
                        'Año': int(ano), 'Proyectos': int(fila['NumProy (base)']),
                        'Pico principal (S/)': pico_principal,
                        'Media (S/)': float(fila['Media']), 'Mediana (S/)': float(fila['Mediana']),
                        'Q1 (S/)': float(fila['Q1']), 'Q3 (S/)': float(fila['Q3']),
                        'Coef. Variación (%)': cv_val, 'Kurtosis': kurt_val,
                        'Asimetría': sk, 'Forma': forma, 'Dispersión': disp, 'Colas': cola,
                    })

                if interpret_rows:
                    df_interp = pd.DataFrame(interpret_rows).sort_values('Año')
                    st.dataframe(df_interp.style.format({
                        'Pico principal (S/)': '{:,.2f}', 'Media (S/)': '{:,.2f}',
                        'Mediana (S/)': '{:,.2f}', 'Q1 (S/)': '{:,.2f}', 'Q3 (S/)': '{:,.2f}',
                        'Coef. Variación (%)': '{:,.2f}', 'Kurtosis': '{:,.4f}',
                        'Asimetría': '{:,.4f}',
                    }), use_container_width=True)

                    resumen = []
                    if df_interp['Coef. Variación (%)'].notna().any():
                        idx_cv = df_interp['Coef. Variación (%)'].idxmax()
                        resumen.append(
                            f"- Mayor variabilidad relativa: {int(df_interp.loc[idx_cv, 'Año'])} "
                            f"(CV = {float(df_interp.loc[idx_cv, 'Coef. Variación (%)']):,.2f}%).")
                    if df_interp['Pico principal (S/)'].notna().any():
                        idx_p = df_interp['Pico principal (S/)'].idxmax()
                        resumen.append(
                            f"- Mayor concentración en montos altos: "
                            f"{int(df_interp.loc[idx_p, 'Año'])} "
                            f"(pico ≈ S/ {float(df_interp.loc[idx_p, 'Pico principal (S/)']):,.2f}).")
                    if resumen:
                        st.markdown("\n".join(resumen))

            # ==================================================================
            # CUARTILES
            # ==================================================================
            st.markdown("#### Cuartiles, Media y Mediana por Año")
            df_q = df_var[['Año', 'Q1', 'Q2', 'Q3', 'Media', 'Mediana']].copy().sort_values('Año')
            st.dataframe(df_q.style.format({
                'Q1': '{:,.2f}', 'Q2': '{:,.2f}', 'Q3': '{:,.2f}',
                'Media': '{:,.2f}', 'Mediana': '{:,.2f}'
            }), use_container_width=True)

            x_q = df_q['Año'].astype(int).astype(str)
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=x_q, y=df_q['Q1'], mode='lines+markers', name='Q1'))
            fig_q.add_trace(go.Scatter(x=x_q, y=df_q['Q2'], mode='lines+markers', name='Q2 (Mediana)'))
            fig_q.add_trace(go.Scatter(x=x_q, y=df_q['Q3'], mode='lines+markers', name='Q3'))
            fig_q.add_trace(go.Scatter(x=x_q, y=df_q['Media'], mode='lines+markers',
                                        name='Media', line=dict(dash='dash')))
            fig_q.update_layout(title=f'Cuartiles y Media de {variable_sel} por Año',
                xaxis_title='Año', yaxis_title='Valor (S/)', xaxis=ax_cat, hovermode='x unified')
            st.plotly_chart(fig_q, use_container_width=True)

    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        📊 Dashboard de Ejecución Presupuestal | Datos: 2014-2025 |
        v3.0 Optimizado con Polars + DuckDB
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
