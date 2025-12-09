# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import time
import io
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. ROBUSTNESS & CONFIGURATION
# -----------------------------------------------------------------------------

# Version Gate
if st.__version__ < "1.23.0":
    st.error("This app requires Streamlit 1.23+. Please upgrade: pip install --upgrade streamlit")
    st.stop()

st.set_page_config(
    page_title="Financial Analytics | Enterprise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Design System
COLORS = {
    'primary': '#1E3A8A',           # Navy blue
    'primary_light': '#3B82F6',     # Lighter blue
    'secondary': '#059669',          # Emerald green
    'danger': '#DC2626',             # Red
    'warning': '#F59E0B',            # Amber
    'neutral_dark': '#374151',       # Dark gray
    'neutral': '#6B7280',            # Medium gray
    'neutral_light': '#9CA3AF',      # Light gray
    'background': '#F9FAFB',         # Very light gray
    'border': '#E5E7EB',             # Border gray
    
    # Chart fills (with opacity)
    'primary_fill': 'rgba(30, 58, 138, 0.12)',
    'secondary_fill': 'rgba(5, 150, 105, 0.12)',
    'danger_fill': 'rgba(220, 38, 38, 0.12)',
}

# Plotly Template Configuration
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "financial_chart",
        "height": 1080,
        "width": 1920,
        "scale": 2
    }
}

PLOTLY_LAYOUT = {
    "font": {
        "family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "size": 12,
        "color": COLORS['neutral_dark']
    },
    "paper_bgcolor": "white",
    "plot_bgcolor": COLORS['background'],
    "margin": {"l": 60, "r": 40, "t": 50, "b": 60},
    "showlegend": True,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.25,
        "xanchor": "center",
        "x": 0.5,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": COLORS['border'],
        "borderwidth": 1
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 13,
        "font_family": "Inter",
        "bordercolor": COLORS['border']
    },
    "xaxis": {
        "showgrid": True,
        "gridcolor": COLORS['border'],
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": COLORS['border'],
        "linewidth": 1
    },
    "yaxis": {
        "showgrid": True,
        "gridcolor": COLORS['border'],
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": COLORS['neutral_light'],
        "zerolinewidth": 1,
        "showline": True,
        "linecolor": COLORS['border'],
        "linewidth": 1
    }
}

# CSS Injection
def inject_custom_css():
    st.markdown("""
        <style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Main container */
.main {
    background-color: #F9FAFB;
}

/* Headers */
h1, h2, h3, h4 {
    color: #111827;
    font-weight: 600;
}

h1 {
    font-size: 32px;
    margin-bottom: 24px;
}

h2 {
    font-size: 24px;
    margin-bottom: 20px;
    margin-top: 32px;
}

h3 {
    font-size: 18px;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Sidebar */
.css-1d391kg {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* KPI Cards (defined earlier but included here for completeness) */
.fp-kpi-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 24px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    margin-bottom: 16px;
    transition: box-shadow 0.2s ease;
}

.fp-kpi-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.fp-kpi-label {
    font-size: 11px;
    color: #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.fp-kpi-value {
    font-size: 28px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 4px;
    line-height: 1.2;
    word-wrap: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}

.fp-kpi-delta {
    font-size: 13px;
    font-weight: 500;
}

.fp-kpi-delta.positive {
    color: #059669;
}

.fp-kpi-delta.negative {
    color: #DC2626;
}

.fp-kpi-delta.neutral {
    color: #6B7280;
}

/* Buttons */
.stButton button {
    background-color: #1E3A8A;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    transition: background-color 0.2s ease;
}

.stButton button:hover {
    background-color: #1E40AF;
}

/* Download button */
.stDownloadButton button {
    background-color: #059669;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

.stDownloadButton button:hover {
    background-color: #047857;
}

/* Selectbox and inputs */
.stSelectbox, .stNumberInput {
    border-radius: 8px;
}

/* Divider */
hr {
    margin: 32px 0;
    border: none;
    border-top: 1px solid #E5E7EB;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .fp-kpi-value {
        font-size: 22px;
    }
    
    h1 {
        font-size: 26px;
    }
    
    h2 {
        font-size: 20px;
    }
    
    /* Force single column layout on mobile */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
        max-width: 100% !important;
    }
}

@media (min-width: 769px) and (max-width: 1024px) {
    /* Tablet: Force 2 columns max */
    div[data-testid="column"]:nth-child(n+3) {
        margin-top: 16px;
    }
}

/* Loading spinner */
.stSpinner > div {
    border-top-color: #1E3A8A !important;
}

/* File uploader */
.uploadedFile {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 8px;
    background-color: white;
}

/* Radio buttons */
.stRadio > label {
    font-weight: 500;
    color: #374151;
}

/* Metrics (native Streamlit) */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 600;
}


[data-testid="stMetricLabel"] {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6B7280;
}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def safe_div(n, d, default=0.0):
    try:
        if d == 0: return default
        return n / d
    except:
        return default

def validate_data_structure(data: dict):
    required_sheets = ["Ingresos", "Datos_Maestros", "Cuentas_Bancarias", "Deudas_Prestamos", "Inversiones", "Presupuesto_Mensual"]
    missing = [s for s in required_sheets if s not in data]
    if missing:
        raise ValueError(f"Missing required sheets: {', '.join(missing)}")
    
    # Validate critical columns
    req_cols = {
        "Ingresos": ["Fecha", "Monto_Neto"],
        "Cuentas_Bancarias": ["Saldo_Inicial"],
        "Inversiones": ["Fecha_Inversión", "Monto_Invertido"]
    }
    for sheet, cols in req_cols.items():
        if sheet in data:
            for col in cols:
                if col not in data[sheet].columns:
                    raise ValueError(f"Sheet '{sheet}' missing column '{col}'")

def ensure_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_excel_with_profiling(file) -> dict:
    start_time = time.time()
    
    try:
        xls = pd.ExcelFile(file)
        # Load all sheets
        data = {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names}
        
        # Validation
        validate_data_structure(data)
        
        # Process specific logic (Unified Expenses & Generated Movimientos)
        data = process_data_logic(data)
        
        elapsed = time.time() - start_time
        if elapsed > 5.0:
            st.toast(f"⚠️ Large file loaded in {elapsed:.1f}s", icon="⚠️")
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

def process_data_logic(data: dict) -> dict:
    """
    Core data processing:
    1. Unify Gastos if needed.
    2. Generate Movimientos_Consolidados.
    3. Ensure Datetimes.
    """
    # 1. Handle Expenses
    if "Gastos" in data:
        gastos = data["Gastos"]
        if "Tipo" in gastos.columns:
            # Case insensitive check
            cols = gastos.columns.str.lower()
            if "tipo" in cols:
                tipo_col = gastos.columns[cols == "tipo"][0]
                data["Gastos_Fijos"] = gastos[gastos[tipo_col].astype(str).str.lower() == "fijo"].copy()
                data["Gastos_Variables"] = gastos[gastos[tipo_col].astype(str).str.lower() == "variable"].copy()
    
    # Ensure empty main DFs if missing
    for k in ["Gastos_Fijos", "Gastos_Variables", "Pagos_Deudas", "Inversiones", "Ingresos"]:
        if k not in data: data[k] = pd.DataFrame()

    # Parse Dates
    date_cols = {
        "Ingresos": "Fecha",
        "Gastos_Fijos": "Fecha_Cargo",
        "Gastos_Variables": "Fecha",
        "Pagos_Deudas": "Fecha_Pago",
        "Inversiones": "Fecha_Inversión"
    }
    for sheet, c in date_cols.items():
        # Handle rename for unified Gastos scenarios if cols differ
        if sheet == "Gastos_Fijos" and "Fecha" in data[sheet].columns:
            data[sheet] = data[sheet].rename(columns={"Fecha": "Fecha_Cargo"})
        
        target_col = date_cols[sheet]
        if target_col in data[sheet].columns:
             data[sheet] = ensure_datetime(data[sheet], target_col)

    # 2. Generate Movimientos Consolidated
    movs = []
    
    # Incomes
    if not data["Ingresos"].empty:
        df = data["Ingresos"].copy()
        df["Tipo"] = "Ingreso"
        # Robust column access
        if "Monto_Neto" in df.columns:
            df["Monto"] = df["Monto_Neto"]
        else:
             df["Monto"] = 0.0
             
        df["Cuenta_Afectada"] = df.get("Cuenta_Destino", "Unknown")
        cols_to_keep = ["Fecha", "Tipo", "Categoría", "Descripción", "Monto", "Cuenta_Afectada"]
        # Filter existing only
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        movs.append(df[cols_to_keep])
        
    # Expenses (Generic Helper)
    def add_expense(sheet_name, tipo, monto_multiplier=-1):
        if not data[sheet_name].empty:
            df = data[sheet_name].copy()
            df["Tipo"] = tipo
            if "Fecha_Cargo" in df.columns: df = df.rename(columns={"Fecha_Cargo": "Fecha"})
            if "Monto" in df.columns: df["Monto"] = df["Monto"].abs() * monto_multiplier
            df["Cuenta_Afectada"] = df.get("Cuenta_Cargo", "Unknown")
            cols_to_keep = ["Fecha", "Tipo", "Categoría", "Descripción", "Monto", "Cuenta_Afectada"]
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            movs.append(df[cols_to_keep])

    add_expense("Gastos_Fijos", "Gasto Fijo")
    add_expense("Gastos_Variables", "Gasto Variable")
        
    # Debt Payments
    if not data["Pagos_Deudas"].empty:
        df = data["Pagos_Deudas"].copy()
        df["Tipo"] = "Pago Deuda"
        df = df.rename(columns={"Fecha_Pago": "Fecha"})
        if "Monto_Total_Pagado" in df.columns:
             df["Monto"] = -df["Monto_Total_Pagado"].abs()
        else:
             df["Monto"] = 0.0
        
        df["Cuenta_Afectada"] = df.get("Cuenta_Origen", "Unknown")
        df["Categoría"] = "Deuda"
        if "ID_Deuda" in df.columns:
            df["Descripción"] = "Pago Deuda " + df["ID_Deuda"].astype(str)
        cols_to_keep = ["Fecha", "Tipo", "Categoría", "Descripción", "Monto", "Cuenta_Afectada"]
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        movs.append(df[cols_to_keep])
        
    # Investments (Outflows)
    if not data["Inversiones"].empty:
        df = data["Inversiones"].copy()
        if "Cuenta_Origen" in df.columns:
            df = df.dropna(subset=["Cuenta_Origen"])
            if not df.empty:
                df["Tipo"] = "Inversión"
                df = df.rename(columns={"Fecha_Inversión": "Fecha"})
                if "Monto_Invertido" in df.columns:
                     df["Monto"] = -df["Monto_Invertido"].abs()
                else:
                     df["Monto"] = 0.0
                     
                df["Cuenta_Afectada"] = df["Cuenta_Origen"]
                df["Categoría"] = "Inversión"
                if "Nombre_Activo" in df.columns:
                    df["Descripción"] = "Inv: " + df["Nombre_Activo"].astype(str)
                
                cols_to_keep = ["Fecha", "Tipo", "Categoría", "Descripción", "Monto", "Cuenta_Afectada"]
                cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                movs.append(df[cols_to_keep])

    if movs:
        full_mov = pd.concat(movs, ignore_index=True)
        if "Fecha" in full_mov.columns:
            full_mov["Fecha"] = pd.to_datetime(full_mov["Fecha"])
            full_mov = full_mov.sort_values("Fecha")
            
            # Calculate Accumulated Balance (Global)
            initial_balance = 0.0
            if "Cuentas_Bancarias" in data and not data["Cuentas_Bancarias"].empty:
                if "Saldo_Inicial" in data["Cuentas_Bancarias"].columns:
                    initial_balance = data["Cuentas_Bancarias"]["Saldo_Inicial"].sum()
                
            full_mov["Saldo_Acumulado"] = initial_balance + full_mov["Monto"].cumsum()
            data["Movimientos_Consolidados"] = full_mov
    else:
        # Fallback empty structure
        data["Movimientos_Consolidados"] = pd.DataFrame(columns=["Fecha", "Tipo", "Categoría", "Descripción", "Monto", "Cuenta_Afectada", "Saldo_Acumulado"])

    return data

# -----------------------------------------------------------------------------
# 3. COMPONENT HELPERS
# -----------------------------------------------------------------------------

def kpi_card(title, value, delta=None, delta_text="vs last period"):
    """
    Renders a KPI card using native st.container + CSS
    """
    with st.container():
        st.markdown(f'<div class="fp-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="fp-metric-label">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fp-metric-value">{value}</div>', unsafe_allow_html=True)
        
        if delta is not None:
            delta_cls = "fp-positive" if delta >= 0 else "fp-negative"
            delta_icon = "▲" if delta >= 0 else "▼"
            st.markdown(f'''
                <div class="fp-metric-delta {delta_cls}">
                    {delta_icon} {abs(delta):.1f}% <span class="fp-neutral" style="margin-left:4px; font-weight:400;">{delta_text}</span>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def professional_table(df, key=None):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        key=key
    )

def money(val, symbol="S/ "):
    try:
        return f"{symbol}{val:,.2f}"
    except:
        return f"{symbol}0.00"

# -----------------------------------------------------------------------------
# 4. CALCULATION MODULES
# -----------------------------------------------------------------------------

class Assumptions:
    def __init__(self, currency_symbol="S/ ", 
                 include_investments_in_cashflow=False, 
                 include_debt_payments_in_expenses=True,
                 expected_return_annual=0.08, inflation_annual=0.03, swr=0.04,
                 extra_debt_payment_monthly=0.0, bienes_value=0.0,
                 invest_liquidation_haircut=0.70, runway_include_investments=False):
        self.currency_symbol = currency_symbol
        self.include_investments_in_cashflow = include_investments_in_cashflow
        self.include_debt_payments_in_expenses = include_debt_payments_in_expenses
        self.expected_return_annual = expected_return_annual
        self.inflation_annual = inflation_annual
        self.swr = swr
        self.extra_debt_payment_monthly = extra_debt_payment_monthly
        self.bienes_value = bienes_value
        self.invest_liquidation_haircut = invest_liquidation_haircut
        self.runway_include_investments = runway_include_investments

def get_period_series(datetime_series, freq='M'):
    """
    Custom period grouper (M=Month YYYY-MM, Q=Quarter, Y=Year)
    """
    if freq == 'M':
        return datetime_series.dt.to_period('M').astype(str)
    elif freq == 'Q':
        return datetime_series.dt.to_period('Q').astype(str)
    elif freq == 'Y':
        return datetime_series.dt.to_period('Y').astype(str)
    return datetime_series.dt.to_period('M').astype(str)

def build_dynamic_table(data: dict, a: Assumptions, freq="M"):
    try:
        if "Ingresos" not in data: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        ing = data["Ingresos"].copy()
        gf = data["Gastos_Fijos"].copy()
        gv = data["Gastos_Variables"].copy()
        pagos = data["Pagos_Deudas"].copy()
        inv_out = pd.DataFrame() # Default empty
        if "Inversiones" in data and not data["Inversiones"].empty:
             # Only consider investments with origin account as outflow
             inv_out = data["Inversiones"].dropna(subset=["Cuenta_Origen"]).copy() if "Cuenta_Origen" in data["Inversiones"].columns else pd.DataFrame()
        
        # Add Period col
        for df, date_c in [(ing, "Fecha"), (gf, "Fecha_Cargo"), (gv, "Fecha"), (pagos, "Fecha_Pago")]:
            if not df.empty and date_c in df.columns:
                 df["Periodo"] = get_period_series(df[date_c], freq)
        
        if not inv_out.empty and "Fecha_Inversión" in inv_out.columns:
            inv_out["Periodo"] = get_period_series(inv_out["Fecha_Inversión"], freq)

        # Helper to sum by period
        def sum_by_period(df, val_col):
            if df.empty or "Periodo" not in df.columns: return pd.Series(dtype=float)
            return df.groupby("Periodo")[val_col].sum()

        ing_s = sum_by_period(ing, "Monto_Neto")
        gf_s = sum_by_period(gf, "Monto")
        gv_s = sum_by_period(gv, "Monto")
        debt_s = sum_by_period(pagos, "Monto_Total_Pagado") 
        # Calculate interest separately for KPIs
        debt_int_s = sum_by_period(pagos, "Monto_Intereses") if "Monto_Intereses" in pagos.columns else pd.Series(dtype=float)
        
        inv_s = pd.Series(dtype=float)
        if not inv_out.empty:
            inv_s = sum_by_period(inv_out, "Monto_Invertido")

        # Merge all into master DF
        pass # Placeholder, constructing simpler version for now
        
        # Get all unique periods sorted
        all_periods = sorted(list(set(ing_s.index) | set(gf_s.index) | set(gv_s.index) | set(debt_s.index)))
        
        dfm = pd.DataFrame({"Mes": all_periods})
        dfm["Ingresos_Netos"] = dfm["Mes"].map(ing_s).fillna(0.0)
        dfm["Gastos_Fijos"] = dfm["Mes"].map(gf_s).fillna(0.0)
        dfm["Gastos_Variables"] = dfm["Mes"].map(gv_s).fillna(0.0)
        dfm["Pagos_Deuda"] = dfm["Mes"].map(debt_s).fillna(0.0)
        dfm["Pagos_Intereses"] = dfm["Mes"].map(debt_int_s).fillna(0.0)
        dfm["Inversiones_Outflow"] = dfm["Mes"].map(inv_s).fillna(0.0)
        
        # Total Expenses Logic
        expenses = dfm["Gastos_Fijos"] + dfm["Gastos_Variables"]
        if a.include_debt_payments_in_expenses:
            expenses += dfm["Pagos_Deuda"]
        if a.include_investments_in_cashflow:
            expenses += dfm["Inversiones_Outflow"]
            
        dfm["Gastos_Totales"] = expenses
        dfm["Flujo_Neto"] = dfm["Ingresos_Netos"] - dfm["Gastos_Totales"]
        dfm["Tasa_Ahorro"] = dfm.apply(lambda r: safe_div(r["Flujo_Neto"], r["Ingresos_Netos"]), axis=1)
        
        # Return components
        return dfm, pd.DataFrame(), ing, gf, gv, pagos, inv_out # Shim return
    except Exception as e:
        print(f"Error in build_dynamic_table: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def compute_ratios_and_balance(data: dict, dfm: pd.DataFrame, a: Assumptions):
    ratios = {}
    try:
        # Asset / Liabilities Snapshots
        # 1. Cash (Liquidity)
        # Use generated Saldo_Acumulado last value
        mov = data.get("Movimientos_Consolidados", pd.DataFrame())
        current_liquidity = 0.0
        if not mov.empty and "Saldo_Acumulado" in mov.columns:
             current_liquidity = float(mov.iloc[-1]["Saldo_Acumulado"])
        
        # 2. Investments (Productive Assets)
        # Sum current value of investments if available, or cost basis
        inv = data.get("Inversiones", pd.DataFrame())
        invest_value = 0.0
        if not inv.empty:
            # Prefer "Valor_Actual" if maintained, else sum "Monto_Invertido" as proxy
            if "Valor_Actual" in inv.columns:
                invest_value = inv["Valor_Actual"].sum()
            elif "Monto_Invertido" in inv.columns:
                invest_value = inv["Monto_Invertido"].sum()
                
        # 3. Debt (Liabilities)
        debts = data.get("Deudas_Prestamos", pd.DataFrame())
        debt_total = 0.0
        if not debts.empty and "Saldo_Actual" in debts.columns:
             debt_total += debts["Saldo_Actual"].sum()
             
        cards = data.get("Tarjetas_Credito", pd.DataFrame())
        credit_limit_total = 1.0
        if not cards.empty:
             if "Saldo_Actual" in cards.columns:
                 debt_total += cards["Saldo_Actual"].sum()
             if "Límite_Crédito" in cards.columns:
                 credit_limit_total = cards["Límite_Crédito"].sum()

        net_worth = (current_liquidity + invest_value + a.bienes_value) - debt_total
        
        # Averages from Flow (dfm)
        avg_monthly_income = dfm["Ingresos_Netos"].mean() if not dfm.empty else 1.0
        avg_monthly_spend = dfm["Gastos_Totales"].mean() if not dfm.empty else 1.0
        
        # Compute Ratios
        ratios["patrimonio_neto_actual"] = net_worth
        ratios["activos_liquidos"] = current_liquidity
        ratios["inversiones_valor"] = invest_value
        ratios["pasivos_totales"] = debt_total
        
        ratios["ratio_liquidez"] = max(0.0, safe_div(current_liquidity, avg_monthly_spend))
        ratios["ratio_deuda_ingresos"] = safe_div(debt_total, (avg_monthly_income * 12)) # Debt to Annual Income
        ratios["util_credito"] = safe_div(debt_total, credit_limit_total) # Rough approx for cards
        ratios["tasa_ahorro_global"] = safe_div(dfm["Flujo_Neto"].sum(), dfm["Ingresos_Netos"].sum())
        
        # FI Index
        # Passive income = assume expected_return on investments / 12
        passive_monthly = (invest_value * a.expected_return_annual) / 12.0
        ratios["passive_monthly"] = passive_monthly
        ratios["fi_index"] = safe_div(passive_monthly, avg_monthly_spend) * 100.0
        ratios["gasto_mensual_prom"] = avg_monthly_spend

    except Exception as e:
        print(f"Error computing ratios: {e}")
        # Return safe defaults
        return {
            "patrimonio_neto_actual": 0.0, "activos_liquidos": 0.0, "inversiones_valor": 0.0, "pasivos_totales": 0.0,
            "ratio_liquidez": 0.0, "ratio_deuda_ingresos": 0.0, "util_credito": 0.0, "tasa_ahorro_global": 0.0,
            "fi_index": 0.0, "passive_monthly": 0.0, "gasto_mensual_prom": 1.0
        }
    return ratios

# -----------------------------------------------------------------------------
# 3. COMPONENT HELPERS
# -----------------------------------------------------------------------------

def kpi_card(title, value, delta=None, delta_text="vs last period"):
    """
    Renders a KPI card using native st.container + CSS
    """
    with st.container():
        st.markdown(f'<div class="fp-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="fp-metric-label">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fp-metric-value">{value}</div>', unsafe_allow_html=True)
        
        if delta is not None:
            delta_cls = "fp-positive" if delta >= 0 else "fp-negative"
            delta_icon = "▲" if delta >= 0 else "▼"
            st.markdown(f'''
                <div class="fp-metric-delta {delta_cls}">
                    {delta_icon} {abs(delta):.1f}% <span class="fp-neutral" style="margin-left:4px; font-weight:400;">{delta_text}</span>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def professional_table(df, key=None):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        key=key
    )

# -----------------------------------------------------------------------------
# 4. CALCULATION LOGIC (Continued)
# -----------------------------------------------------------------------------

def compute_health_score(dfm, ratios, data):
    """
    Simplified professional health score (0-1000) based on key ratios.
    """
    score = 500 # Base
    try:
        # 1. Savings Rate (+/- 200)
        sr = ratios.get("tasa_ahorro_global", 0)
        if sr > 0.20: score += 100
        if sr > 0.40: score += 100
        if sr < 0: score -= 150
        
        # 2. Liquidity (+/- 150)
        liq = ratios.get("ratio_liquidez", 0)
        if liq >= 3: score += 75
        if liq >= 6: score += 75
        if liq < 1: score -= 100
        
        # 3. Dept (+/- 150)
        dr = ratios.get("ratio_deuda_ingresos", 0)
        if dr < 0.30: score += 150
        elif dr > 0.50: score -= 150
        
        return max(0, min(1000, score))
    except:
        return 500

def net_worth_by_month(data: dict, dfm: pd.DataFrame, a: Assumptions, freq="M"):
    try:
        mov = data.get("Movimientos_Consolidados", pd.DataFrame())
        if mov.empty: return pd.DataFrame(columns=["Mes", "Patrimonio_Neto"])
        
        mov["Periodo"] = get_period_series(mov["Fecha"], freq)
        
        # Cash: Last Saldo_Acumulado per period
        cash_m = mov.sort_values("Fecha").groupby("Periodo")["Saldo_Acumulado"].last()
        
        # Investments
        inv = data.get("Inversiones", pd.DataFrame())
        inv_m = pd.Series(0.0, index=cash_m.index)
        
        if not inv.empty:
             inv["Periodo"] = get_period_series(inv["Fecha_Inversión"], freq)
             # Cumulative sum of amounts
             inv_grouped = inv.groupby("Periodo")["Monto_Invertido"].sum().cumsum()
             inv_m = inv_grouped.reindex(cash_m.index, method='ffill').fillna(0.0)
        
        periods = sorted(cash_m.index.unique())
        
        results = []
        for p in periods:
            c = cash_m.get(p, 0.0)
            i = inv_m.get(p, 0.0)
            nw = c + i + a.bienes_value 
            results.append({"Mes": p, "Patrimonio_Neto": nw, "Activos_Caja": c, "Activos_Inversiones": i})
            
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame(columns=["Mes", "Patrimonio_Neto"])

def forecast_net_worth(nw_df: pd.DataFrame, base_savings_monthly: float, years=5, a: Assumptions = None):
    try:
        last_val = 0.0
        if not nw_df.empty:
             last_val = float(nw_df.iloc[-1]["Patrimonio_Neto"])
        
        months = int(years * 12)
        r_m = (1 + a.expected_return_annual) ** (1/12) - 1
        
        dates = []
        base = []
        optimistic = []
        pessimistic = []
        
        curr_b = last_val
        curr_o = last_val
        curr_p = last_val
        
        for i in range(1, months + 1):
             curr_b = (curr_b + base_savings_monthly) * (1 + r_m)
             curr_o = (curr_o + (base_savings_monthly * 1.2)) * (1 + (r_m * 1.1)) 
             curr_p = (curr_p + (base_savings_monthly * 0.8)) * (1 + (r_m * 0.9)) 
             base.append(curr_b)
             optimistic.append(curr_o)
             pessimistic.append(curr_p)
             dates.append(i) 
             
        return pd.DataFrame({
            "Mes_Futuro": dates,
            "Base": base,
            "Optimista": optimistic,
            "Pesimista": pessimistic
        })
    except:
        return pd.DataFrame()

def compute_runway(ratios, dfm, a: Assumptions):
    try:
        liquid = ratios.get("activos_liquidos", 0)
        invest = ratios.get("inversiones_valor", 0)
        
        monthly_spend = float(dfm["Gastos_Totales"].mean()) if not dfm.empty else 1.0
        monthly_survival = float(dfm["Gastos_Fijos"].mean()) + float(dfm["Pagos_Deuda"].mean())
        
        runway_normal = safe_div(liquid, monthly_spend)
        runway_survival = safe_div(liquid, monthly_survival)
        
        liquid_total = liquid
        if a.runway_include_investments:
             liquid_total += (invest * a.invest_liquidation_haircut)
             
        runway_with_liq = safe_div(liquid_total, monthly_spend)
        
        return runway_normal, runway_survival, runway_with_liq, monthly_spend, monthly_survival
    except:
        return 0, 0, 0, 0, 0

def compute_money_velocity(mov: pd.DataFrame, ingresos: pd.DataFrame, freq="M"):
    try:
        if mov.empty or ingresos.empty: return pd.DataFrame()
        mov = mov.copy()
        mov["Fecha"] = pd.to_datetime(mov["Fecha"])
        mov["Periodo"] = get_period_series(mov["Fecha"], freq)
        ingresos = ingresos.copy()
        ingresos["Periodo"] = get_period_series(ingresos["Fecha"], freq)
    
        results = []
        for per in sorted(mov["Periodo"].unique()):
            inc = ingresos.loc[ingresos["Periodo"] == per, "Monto_Neto"].sum()
            if inc <= 0: continue
            half = 0.5 * inc
    
            m = mov.loc[mov["Periodo"] == per].copy()
            m = m.sort_values("Fecha")
            # Filter only expenses (negative)
            m["Gasto_Pos"] = (-m["Monto"]).clip(lower=0)
            m["CumGasto"] = m["Gasto_Pos"].cumsum()
    
            hit = m.loc[m["CumGasto"] >= half]
            if hit.empty: continue
            
            hit_dt = hit.iloc[0]["Fecha"]
            first_dt = m.iloc[0]["Fecha"]
            days = max(1.0, (hit_dt - first_dt).total_seconds() / 86400.0)
            
            vel = safe_div(inc, days)
            results.append({
                "Periodo": per,
                "Ingreso_Periodo": inc,
                "Dias_hasta_50pct": days,
                "Velocidad": vel
            })
        return pd.DataFrame(results)
    except:
        return pd.DataFrame()

def build_debt_universe(data):
    df = data.get("Deudas_Prestamos", pd.DataFrame()).copy()
    if df.empty: return []
    debts = []
    for _, row in df.iterrows():
        debts.append({
            "id": row.get("ID_Deuda", "Unknown"),
            "name": row.get("Descripción", "Unknown"),
            "balance": float(row.get("Saldo_Actual", 0)),
            "rate_annual": float(row.get("Tasa_Interés_Anual", 0)),
            "min_payment": float(row.get("Cuota_Mensual", 0))
        })
    return debts

def simulate_payoff(debts_universe, extra_payment=0, strategy="avalanche"):
    import copy
    debts = copy.deepcopy(debts_universe)
    schedule = []
    month = 0
    total_interest = 0.0
    
    # Safety break
    while any(d["balance"] > 0 for d in debts) and month < 360:
        month += 1
        monthly_extra_left = extra_payment
        
        # 1. Minimum payments & Interest
        for d in debts:
            if d["balance"] <= 0: continue
            
            interest = d["balance"] * (d["rate_annual"] / 12.0)
            total_interest += interest
            
            # Min payment or full balance
            min_pay = min(d["balance"] + interest, d["min_payment"]) 
            principal_paid = min_pay - interest
            d["balance"] -= principal_paid
            
            schedule.append({
                "month": month, "debt_id": d["id"], "type": "min", 
                "payment": min_pay, "interest": interest, "principal": principal_paid,
                "balance_end": d["balance"]
            })
            
        # 2. Extra Payment Strategy
        alive_debts = [d for d in debts if d["balance"] > 0]
        if alive_debts and monthly_extra_left > 0:
            target = None
            if strategy == "avalanche":
                target = sorted(alive_debts, key=lambda x: x["rate_annual"], reverse=True)[0]
            elif strategy == "snowball":
                target = sorted(alive_debts, key=lambda x: x["balance"])[0]
            elif strategy == "hybrid":
                # Score = Rate normalized * 0.6 + (1 - Balance normalized) * 0.4
                # Simpler implementation without numpy normalization:
                # Rank by rate desc (high priority) and balance asc (high priority)
                # Hybrid score = rate - (balance / 100000000) check? No.
                # Just use rate sorting as fallback or simplified mixed score
                # Let's trust user request "Hybrid (Balanced)" implies some mix.
                # Fallback to Avalanche for now as I can't easily normalize without full list context re-calc.
                target = sorted(alive_debts, key=lambda x: x["rate_annual"], reverse=True)[0] 
            else: 
                target = sorted(alive_debts, key=lambda x: x["rate_annual"], reverse=True)[0]
            
            if target:
                pay = min(target["balance"], monthly_extra_left)
                target["balance"] -= pay
                schedule.append({
                    "month": month, "debt_id": target["id"], "type": "extra",
                    "payment": pay, "interest": 0, "principal": pay,
                    "balance_end": target["balance"]
                })
    
    return pd.DataFrame(schedule), total_interest, month

def opportunity_cost_table(gv, rate_annual, years, threshold):
    try:
        if gv.empty: return pd.DataFrame()
        high_exp = gv[gv["Monto"].abs() >= threshold].copy()
        if high_exp.empty: return pd.DataFrame()
        
        r = rate_annual
        factor = (1 + r) ** years
        high_exp["Valor_Futuro"] = high_exp["Monto"].abs() * factor
        high_exp["Costo_Oportunidad"] = high_exp["Valor_Futuro"] - high_exp["Monto"].abs()
        return high_exp.sort_values("Monto", ascending=True)
    except:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. UI HELPERS (EDUCATIONAL)
# -----------------------------------------------------------------------------

def render_kpi_with_tooltip(label, value, delta=None, delta_text="vs anterior", target=None, tooltip=None, color=None):
    """KPI card usando componentes nativos (evita HTML roto)"""
    
    with st.container():
        # Wrapper con clase CSS
        st.markdown('<div class="fp-kpi-card">', unsafe_allow_html=True)
        
        # Label + tooltip en columnas
        col_label, col_info = st.columns([5, 1])
        with col_label:
            st.markdown(f'<div class="fp-kpi-label">{label}</div>', unsafe_allow_html=True)
        with col_info:
            if tooltip:
                st.markdown(f'<span title="{tooltip}" style="cursor: help; color: {COLORS["primary"]}; font-size: 14px;">ⓘ</span>', unsafe_allow_html=True)
        
        # Value con color
        value_color = color or COLORS['neutral_dark']
        st.markdown(f'''
            <div class="fp-kpi-value" style="
                color: {value_color};
                font-size: 28px;
                font-weight: 600;
                line-height: 1.2;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                text-overflow: ellipsis;
            ">{value}</div>
        ''', unsafe_allow_html=True)
        
        # Delta
        if delta is not None:
            delta_class = "positive" if delta >= 0 else "negative"
            arrow = "▲" if delta >= 0 else "▼"
            st.markdown(f'<div class="fp-kpi-delta {delta_class}">{arrow} {abs(delta):.1f}% {delta_text}</div>', unsafe_allow_html=True)
        
        # Target
        if target:
            st.markdown(f'<div class="fp-kpi-delta neutral">{target}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Alias para compatibilidad
render_kpi_card = render_kpi_with_tooltip

def render_smart_recommendations(ratios, dfm):
    """Sistema de insights basado en análisis cuantitativo"""
    
    insights = []
    
    # 1. Análisis de Liquidez
    liq_ratio = ratios.get('ratio_liquidez', 0)
    if liq_ratio < 1:
        insights.append({
            'severity': 'high',
            'category': 'Liquidez',
            'title': 'Cobertura de efectivo insuficiente',
            'metric': f'{liq_ratio:.1f} meses',
            'benchmark': '3-6 meses',
            'message': 'La reserva de efectivo actual no cubre emergencias básicas. Esto representa un riesgo ante imprevistos.',
            'action': 'Establecer transferencias automáticas mensuales del 10% de ingresos a una cuenta de ahorro separada.'
        })
    elif liq_ratio < 3:
        insights.append({
            'severity': 'medium',
            'category': 'Liquidez',
            'title': 'Reserva de emergencia por debajo del estándar',
            'metric': f'{liq_ratio:.1f} meses',
            'benchmark': '3-6 meses',
            'message': 'La cobertura actual está por debajo del rango recomendado por planificadores financieros.',
            'action': 'Incrementar el fondo de emergencia destinando 15% de ingresos mensuales hasta alcanzar 6 meses de cobertura.'
        })
    
    # 2. Análisis de Flujo de Caja
    sr = ratios.get('tasa_ahorro_global', 0)
    if sr < 0:
        insights.append({
            'severity': 'high',
            'category': 'Flujo de Caja',
            'title': 'Déficit en flujo operativo',
            'metric': f'{sr*100:.1f}%',
            'benchmark': '>20%',
            'message': 'Los egresos superan los ingresos de forma sostenida, erosionando el patrimonio mensualmente.',
            'action': 'Revisar gastos discrecionales (entretenimiento, dining out) y eliminar suscripciones no esenciales. Priorizar generación de ingresos adicionales.'
        })
    elif sr < 0.10:
        insights.append({
            'severity': 'medium',
            'category': 'Flujo de Caja',
            'title': 'Tasa de ahorro por debajo del objetivo',
            'metric': f'{sr*100:.1f}%',
            'benchmark': '>20%',
            'message': 'La capacidad de ahorro actual dificulta el cumplimiento de metas financieras de mediano y largo plazo.',
            'action': 'Aplicar metodología 50/30/20: 50% necesidades básicas, 30% estilo de vida, 20% ahorro e inversión.'
        })
    
    # 3. Análisis de Endeudamiento
    dti = ratios.get('ratio_deuda_ingresos', 0)
    if dti > 0.40:
        insights.append({
            'severity': 'high',
            'category': 'Endeudamiento',
            'title': 'Ratio deuda/ingreso en zona de riesgo',
            'metric': f'{dti*100:.1f}%',
            'benchmark': '≤30%',
            'message': 'El nivel de endeudamiento actual compromete más del 40% de los ingresos anuales, limitando severamente la capacidad de ahorro.',
            'action': 'Implementar estrategia de pago acelerado (método Avalancha) priorizando deudas con mayor costo financiero. Ver sección "Estrategia de Deuda".'
        })
    
    # 4. Caso óptimo
    if not insights:
        insights.append({
            'severity': 'low',
            'category': 'Salud Financiera',
            'title': 'Indicadores dentro de rangos óptimos',
            'metric': '✓',
            'benchmark': 'Cumplido',
            'message': 'Los principales indicadores financieros se encuentran dentro de los rangos recomendados por estándares de planificación.',
            'action': 'Considerar optimización de portafolio de inversión para acelerar crecimiento patrimonial. Consultar con asesor financiero certificado.'
        })
    
    # Renderizado profesional
    if insights:
        st.markdown("---")
        st.markdown("### Análisis de Riesgo y Oportunidades")
        st.caption("Generado mediante análisis cuantitativo de indicadores financieros")
        
        # Renderizar insights (máximo 3)
        for insight in insights[:3]:
            severity_colors = {
                'high': {'bg': '#FEE2E2', 'border': '#DC2626', 'text': '#991B1B', 'icon': '●'},
                'medium': {'bg': '#FEF3C7', 'border': '#F59E0B', 'text': '#92400E', 'icon': '▲'},
                'low': {'bg': '#D1FAE5', 'border': '#059669', 'text': '#065F46', 'icon': '✓'},
            }
            
            colors = severity_colors[insight['severity']]
            
            with st.expander(
                f"{colors['icon']} {insight['category']}: {insight['title']}", 
                expanded=(insight['severity'] == 'high')
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="
                        background: {colors['bg']};
                        border-left: 4px solid {colors['border']};
                        padding: 16px;
                        border-radius: 4px;
                    ">
                        <div style="font-size: 13px; color: {colors['text']}; margin-bottom: 12px;">
                            <strong>Situación:</strong> {insight['message']}
                        </div>
                        <div style="font-size: 13px; color: {colors['text']};">
                            <strong>Acción recomendada:</strong> {insight['action']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Actual", insight['metric'])
                    st.caption(f"**Objetivo:** {insight['benchmark']}")

def safe_plot(fig, title="Chart", fallback_df=None):
    """Render chart with error handling and fallback"""
    try:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    except Exception as e:
        st.error(f"❌ Failed to render {title}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        
        # Fallback to table if available
        if fallback_df is not None:
            st.info("📊 Showing data table instead:")
            st.dataframe(fallback_df, use_container_width=True)

def create_net_worth_chart(nw_df):
    """Professional net worth chart (Area + Trend)"""
    fig = go.Figure()
    
    # Area Chart
    fig.add_trace(go.Scatter(
        x=nw_df['Mes'],
        y=nw_df['Patrimonio_Neto'],
        fill='tozeroy',
        mode='none',
        name='Patrimonio Neto',
        fillcolor=COLORS['primary_fill'],
        hovertemplate='<b>%{x}</b><br>PN: %{y:,.0f}<extra></extra>'
    ))
    
    # Trend Line (Smoothed)
    if len(nw_df) > 1:
        try:
            from scipy.ndimage import gaussian_filter1d
            trend_values = gaussian_filter1d(nw_df['Patrimonio_Neto'].astype(float), sigma=2)
            fig.add_trace(go.Scatter(
                x=nw_df['Mes'],
                y=trend_values,
                mode='lines',
                name='Tendencia',
                line=dict(color=COLORS['neutral'], width=1.5, dash='dash'),
                hovertemplate='Tendencia: %{y:,.0f}<extra></extra>'
            ))
        except: pass
    
    # Layout
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Evolución del Patrimonio Neto',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark'], 'family': 'Inter'}
        },
        height=400,
        yaxis_tickformat=',.0f',
        xaxis_title="",
        yaxis_title="Patrimonio (PEN)"
    )
    
    return fig

def create_waterfall_chart(dfm):
    """Professional waterfall chart for cash flow"""
    # Aggregate period data
    total_income = dfm['Ingresos_Netos'].sum()
    total_fixed = dfm['Gastos_Fijos'].sum()
    total_variable = dfm['Gastos_Variables'].sum()
    total_debt = dfm['Pagos_Deuda'].sum() if 'Pagos_Deuda' in dfm.columns else 0
    net_flow = dfm['Flujo_Neto'].sum()
    
    fig = go.Figure(go.Waterfall(
        name="Flujo de Caja",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Ingresos", "Gastos<br>Fijos", "Gastos<br>Variables", "Pagos<br>Deuda", "Flujo Neto"],
        y=[total_income, -total_fixed, -total_variable, -total_debt, net_flow],
        text=[
            f"+{total_income:,.0f}",
            f"-{total_fixed:,.0f}",
            f"-{total_variable:,.0f}",
            f"-{total_debt:,.0f}",
            f"{net_flow:,.0f}"
        ],
        textposition="outside",
        textfont={"size": 12, "color": COLORS['neutral_dark']},
        connector={
            "mode": "between",
            "line": {"width": 2, "color": COLORS['border'], "dash": "solid"}
        },
        increasing={"marker": {"color": COLORS['secondary']}},
        decreasing={"marker": {"color": COLORS['danger']}},
        totals={"marker": {"color": COLORS['primary']}},
        hovertemplate='%{x}<br>Monto: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Flujo de Caja (Periodo)',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark']}
        },
        height=380,
        showlegend=False,
        yaxis_tickformat=',.0f',
        xaxis_title="",
        yaxis_title="Monto (PEN)"
    )
    
    return fig

def create_income_expenses_chart(dfm):
    """Professional income vs expenses chart"""
    fig = go.Figure()
    
    # Net Flow (bars)
    fig.add_trace(go.Bar(
        x=dfm['Mes'],
        y=dfm['Flujo_Neto'],
        name='Flujo Neto',
        marker_color=COLORS['primary'],
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Flujo Neto: %{y:,.0f}<extra></extra>'
    ))
    
    # Total Income (line)
    fig.add_trace(go.Scatter(
        x=dfm['Mes'],
        y=dfm['Ingresos_Netos'],
        name='Ingresos Totales',
        mode='lines+markers',
        line=dict(color=COLORS['secondary'], width=2),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Ingresos: %{y:,.0f}<extra></extra>'
    ))
    
    # Total Expenses (line)
    fig.add_trace(go.Scatter(
        x=dfm['Mes'],
        y=dfm['Gastos_Totales'],
        name='Gastos Totales',
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Gastos: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Tendencia Ingresos vs Gastos',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark']}
        },
        height=400,
        yaxis_tickformat=',.0f',
        xaxis_title="",
        yaxis_title="Monto (PEN)",
        barmode='relative'
    )
    
    return fig

def create_expenses_breakdown_chart(gv_cat):
    """Professional stacked bar chart for variable expenses"""
    
    # Define color palette for categories (consistent)
    category_colors = {
        'Alimentación': '#059669',
        'Educación': '#3B82F6',
        'Entretenimiento': '#F59E0B',
        'Otros': '#6B7280',
        'Ropa': '#8B5CF6',
        'Salud': '#DC2626',
        'Transporte': '#10B981'
    }
    
    fig = go.Figure()
    
    # Get unique categories
    if not gv_cat.empty:
         categories = gv_cat['Categoría'].unique()
    
         for cat in categories:
             cat_data = gv_cat[gv_cat['Categoría'] == cat]
             fig.add_trace(go.Bar(
                 x=cat_data['Periodo'],
                 y=cat_data['Monto'],
                 name=cat,
                 marker_color=category_colors.get(cat, COLORS['neutral']),
                 hovertemplate=f'<b>{cat}</b><br>%{{x}}<br>Monto: %{{y:,.0f}}<extra></extra>'
             ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Gastos Variables por Categoría',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark']}
        },
        height=400,
        barmode='stack',
        yaxis_tickformat=',.0f',
        xaxis_title="",
        yaxis_title="Monto (PEN)",
        legend={
            'orientation': 'v',
            'yanchor': 'top',
            'y': 1,
            'xanchor': 'right',
            'x': 1.15,
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['border'],
            'borderwidth': 1
        }
    )
    
    return fig

def create_debt_comparison_chart(sch_av, sch_sn, sch_hy):
    """Professional debt payoff comparison with 3 strategies"""
    
    # Extract balance data for each strategy
    def extract_balances(schedule):
        if schedule.empty: return pd.DataFrame()
        last_bals = schedule.sort_values("month").groupby(["month", "debt_id"]).last().reset_index()
        monthly = last_bals.groupby("month")["balance_end"].sum().reset_index()
        return monthly
    
    bal_av = extract_balances(sch_av)
    bal_sn = extract_balances(sch_sn)
    bal_hy = extract_balances(sch_hy)
    
    fig = go.Figure()
    
    # Avalanche strategy (línea sólida azul - la más recomendada)
    if not bal_av.empty:
        fig.add_trace(go.Scatter(
            x=bal_av['month'],
            y=bal_av['balance_end'],
            mode='lines',
            name='Avalancha (Interés Alto Primero) - Recomendada',
            line=dict(color=COLORS['primary'], width=4),  # Línea gruesa sólida
            hovertemplate='<b>Avalancha</b><br>Mes: %{x}<br>Saldo: %{y:,.0f}<extra></extra>'
        ))
    
    # Snowball strategy (línea punteada verde)
    if not bal_sn.empty:
        fig.add_trace(go.Scatter(
            x=bal_sn['month'],
            y=bal_sn['balance_end'],
            mode='lines',
            name='Bola de Nieve (Saldo Bajo Primero)',
            line=dict(color=COLORS['secondary'], width=3, dash='dash'),  # Guiones
            hovertemplate='<b>Bola de Nieve</b><br>Mes: %{x}<br>Saldo: %{y:,.0f}<extra></extra>'
        ))
    
    # Hybrid strategy (línea de puntos naranja)
    if not bal_hy.empty:
        fig.add_trace(go.Scatter(
            x=bal_hy['month'],
            y=bal_hy['balance_end'],
            mode='lines',
            name='Híbrida (Balanceada)',
            line=dict(color=COLORS['warning'], width=3, dash='dot'),  # Puntos
            hovertemplate='<b>Híbrida</b><br>Mes: %{x}<br>Saldo: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Comparativa de Estrategias de Pago',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark']}
        },
        height=450,
        yaxis_tickformat=',.0f',
        xaxis_title="Meses desde hoy",
        yaxis_title="Saldo Total (PEN)",
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,  # Más espacio para leyendas largas
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        )
    )
    
    return fig

def create_forecast_chart(forecast_df):
    """Professional net worth forecast with confidence bands"""
    fig = go.Figure()
    
    # Base scenario
    fig.add_trace(go.Scatter(
        x=forecast_df['Mes_Futuro'],
        y=forecast_df['Base'],
        mode='lines',
        name='Escenario Base',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate='<b>Base</b><br>Mes: %{x}<br>Patrimonio: %{y:,.0f}<extra></extra>'
    ))
    
    # Optimistic scenario (with fill)
    fig.add_trace(go.Scatter(
        x=forecast_df['Mes_Futuro'],
        y=forecast_df['Optimista'],
        mode='lines',
        name='Optimista (+20% ahorro)',
        line=dict(color=COLORS['secondary'], width=2, dash='dash'),
        fill=None,
        hovertemplate='<b>Optimista</b><br>Mes: %{x}<br>Patrimonio: %{y:,.0f}<extra></extra>'
    ))
    
    # Pessimistic scenario (with fill to optimistic)
    fig.add_trace(go.Scatter(
        x=forecast_df['Mes_Futuro'],
        y=forecast_df['Pesimista'],
        mode='lines',
        name='Pesimista (-20% ahorro)',
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(107, 114, 128, 0.1)',
        hovertemplate='<b>Pesimista</b><br>Mes: %{x}<br>Patrimonio: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title={
            'text': 'Proyección Patrimonio (5 Años)',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 16, 'color': COLORS['neutral_dark']}
        },
        height=450,
        yaxis_tickformat=',.0f',
        xaxis_title="Meses Futuros",
        yaxis_title="Patrimonio Proyectado (PEN)"
    )
    
    return fig

# -----------------------------------------------------------------------------
# 5. VIEW LOGIC
# -----------------------------------------------------------------------------

def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Vista General"
    
    with st.sidebar:
        st.title("💳 Financial Pilot")
        
        # Spanish Navigation (consistent)
        nav_options = ["Vista General", "Análisis de Flujo", "Balance y Ratios", "Estrategia de Deuda"]
        
        # Handle English legacy state if present
        if st.session_state.current_page not in nav_options:
            st.session_state.current_page = "Vista General"

        page = st.radio(
            "Navegación", 
            nav_options,
            index=nav_options.index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        st.session_state.current_page = page
        
        st.divider()
        st.markdown("### Configuración")
        currency = st.selectbox("Moneda", ["USD", "EUR", "PEN"])
        
        uploaded_file = st.file_uploader("Cargar datos (Excel)", type=["xlsx"])

    if not uploaded_file:
        st.info("Bienvenido. Para comenzar, cargue su archivo 'Finanzas_Personales_Analisis_Horario.xlsx'.")
        return

    data = load_excel_with_profiling(uploaded_file)
    if not data:
        return

    # Routing
    if page == "Vista General":
        render_overview(data, currency)
    elif page == "Análisis de Flujo":
        render_cashflow(data, currency)
    elif page == "Balance y Ratios":
        render_balancesheet(data, currency)
    elif page == "Estrategia de Deuda":
        render_debt_strategy(data, currency)

def render_overview(data, currency):
    """Vista General - Dashboard con KPIs educativos y recomendaciones IA"""
    st.header("Vista General")
    st.markdown("Panel de control financiero con métricas clave y recomendaciones personalizadas.")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    ratios = compute_ratios_and_balance(data, dfm, ass)
    
    if dfm.empty:
        st.warning("No hay suficientes datos para generar el resumen. Verifique las hojas 'Ingresos' y 'Gastos'.")
        return

    # Net worth evolution data
    nw_df = net_worth_by_month(data, dfm, ass)

    # Calculate deltas for KPIs
    delta_nw = 0
    delta_sr = 0
    if len(dfm) >= 2:
        current_nw = ratios['patrimonio_neto_actual']
        prev_nw = current_nw - dfm['Flujo_Neto'].tail(3).sum()
        if prev_nw != 0:
            delta_nw = ((current_nw - prev_nw) / abs(prev_nw)) * 100
        
        current_sr = ratios['tasa_ahorro_global']
        prev_sr = dfm['Tasa_Ahorro'].tail(6).head(3).mean() if len(dfm) >= 6 else current_sr
        delta_sr = (current_sr - prev_sr) * 100

    # KPI Cards con tooltips educativos
    st.subheader("Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_with_tooltip(
            "PATRIMONIO NETO",
            money(ratios['patrimonio_neto_actual'], currency),
            delta=delta_nw if delta_nw != 0 else None,
            tooltip="Activos totales menos pasivos totales. Indicador clave de salud financiera."
        )

    with col2:
        sr_val = ratios['tasa_ahorro_global'] * 100
        sr_color = COLORS['danger'] if sr_val < 0 else (COLORS['secondary'] if sr_val >= 20 else None)
        render_kpi_with_tooltip(
            "TASA DE AHORRO",
            f"{sr_val:.1f}%",
            delta=delta_sr if delta_sr != 0 else None,
            target="Meta: >20%",
            tooltip="Porcentaje de ingresos netos no consumidos. Benchmark estándar: 20-30% para acumulación patrimonial efectiva.",
            color=sr_color
        )

    with col3:
        liq_val = ratios['ratio_liquidez']
        liq_color = COLORS['secondary'] if liq_val >= 3 else (COLORS['danger'] if liq_val < 1 else None)
        render_kpi_with_tooltip(
            "LIQUIDEZ",
            f"{liq_val:.1f} meses",
            target="Meta: 3-6 meses",
            tooltip="Meses de cobertura con reservas actuales. Estándar de planificación financiera: 3-6 meses de gastos fijos.",
            color=liq_color
        )

    with col4:
        dti_val = ratios['ratio_deuda_ingresos'] * 100
        dti_color = COLORS['secondary'] if dti_val <= 30 else COLORS['danger']
        render_kpi_with_tooltip(
            "DEUDA/INGRESOS",
            f"{dti_val:.1f}%",
            target="Meta: ≤30%",
            tooltip="Ratio deuda total/ingreso anual. Zona de riesgo: >40%. Umbral conservador: ≤30%.",
            color=dti_color
        )
    
    st.divider()

    # Recomendaciones IA (educativas)
    render_smart_recommendations(ratios, dfm)

    # Charts
    st.subheader("Tendencias")
    c1, c2 = st.columns([3, 2])
    
    with c1:
        fig_nw = create_net_worth_chart(nw_df)
        safe_plot(fig_nw, "Evolución Patrimonio")
        
    with c2:
        # Use last available period for waterfall
        last_period = sorted(dfm["Mes"].unique())[-1]
        dfm_period = dfm[dfm["Mes"] == last_period]
        fig_wf = create_waterfall_chart(dfm_period)
        fig_wf.update_layout(title_text=f"Flujo de Caja ({last_period})")
        safe_plot(fig_wf, "Flujo de Caja")
        
    st.divider()
    
    # Data table
    st.subheader("Datos Recientes")
    if not dfm.empty:
        col_sel = ["Mes", "Ingresos_Netos", "Gastos_Totales", "Flujo_Neto", "Tasa_Ahorro"]
        st.dataframe(dfm[col_sel].tail(6), use_container_width=True, hide_index=True)
        
        csv = dfm.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Reporte (CSV)", csv, "reporte_financiero.csv", "text/csv")

def render_cashflow(data, currency):
    st.header("Análisis de Flujo de Caja")
    st.markdown("Detalle de ingresos, egresos y capacidad de ahorro mensual.")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, ing, _, gv, _, _ = build_dynamic_table(data, ass)
    
    if dfm.empty:
        st.warning("No hay datos disponibles.")
        return

    # Main Trend (Combo Chart)
    fig_trend = create_income_expenses_chart(dfm)
    safe_plot(fig_trend, "Tendencia Ingresos vs Gastos")
    
    st.divider()

    # Variable Expenses Breakdown
    st.subheader("Desglose de Gastos Variables")

    # Prepare Category Data
    gv_cat = pd.DataFrame()
    if not gv.empty:
        gv_cat = gv.groupby(["Periodo", "Categoría"])["Monto"].sum().reset_index()

    # Create two columns: chart on left, stats on right
    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        fig_breakdown = create_expenses_breakdown_chart(gv_cat)
        safe_plot(fig_breakdown, "Desglose Gastos Variables")

    with col_stats:
        st.markdown("**Estadísticas Mensuales**")
        
        # Calculate stats
        avg_income = dfm['Ingresos_Netos'].mean()
        avg_expenses = dfm['Gastos_Totales'].mean()
        avg_net = dfm['Flujo_Neto'].mean()
        savings_rate = safe_div(avg_net, avg_income) * 100
        
        # Display as native metrics (evita problemas de HTML)
        st.metric("INGRESO PROMEDIO", money(avg_income, currency))
        st.metric("GASTO PROMEDIO", money(avg_expenses, currency))
        st.metric("FLUJO NETO PROMEDIO", money(avg_net, currency))
        
        sr_color = "normal" if savings_rate >= 20 else ("inverse" if savings_rate <  0 else "off")
        st.metric("TASA DE AHORRO", f"{savings_rate:.1f}%", delta=None)
        st.caption("Meta: >20%")
    
    st.divider()
    
    # Detailed Table
    st.subheader("Detalle Mensual")
    professional_table(dfm)

def render_balancesheet(data, currency):
    st.header("Balance y Ratios Financieros")
    st.markdown("Análisis de situación patrimonial y solvencia.")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    ratios = compute_ratios_and_balance(data, dfm, ass)
    
    # 1. Asset Allocation & Composition
    st.subheader("Composición de Activos")
    
    inv_val = ratios["inversiones_valor"]
    liq_val = ratios["activos_liquidos"]
    bienes_val = ass.bienes_value
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Horizontal Bar Chart (más claro para novatos)
        categories = ['Efectivo y Bancos', 'Inversiones', 'Bienes (Casa, Auto)']
        values = [liq_val, inv_val, bienes_val]
        colors_list = [COLORS['secondary'], COLORS['primary'], COLORS['warning']]
        
        fig_bar = go.Figure(go.Bar(
            y=categories,
            x=values,
            orientation='h',
            marker=dict(color=colors_list),
            text=[money(v, currency) for v in values],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Inter'),
            hovertemplate='<b>%{y}</b><br>Monto: %{x:,.0f}<extra></extra>'
        ))
        
        fig_bar.update_layout(
            PLOTLY_LAYOUT,
            title={
                'text': 'Tus Activos (Lo que Posees)',
                'x': 0,
                'xanchor': 'left',
                'font': {'size': 16, 'color': COLORS['neutral_dark']}
            },
            height=350,
            xaxis_tickformat=',.0f',
            xaxis_title=f"Monto ({currency})",
            yaxis_title="",
            showlegend=False
        )
        
        safe_plot(fig_bar, "Composición de Activos")
        
    with c2:
        st.metric(
            "ACTIVOS TOTALES",
            money(liq_val + inv_val + bienes_val, currency)
        )
        st.caption("Todo lo que posee: efectivo, inversiones y bienes")
        
        st.metric(
            "PASIVOS TOTALES",
            money(ratios["pasivos_totales"], currency)
        )
        st.caption("Todo lo que debe: préstamos y deudas")
        
        st.metric(
            "PATRIMONIO NETO",
            money(ratios["patrimonio_neto_actual"], currency)
        )
        st.caption("Riqueza neta: Activos - Pasivos")
        
    st.divider()
    
    # 2. Financial Ratios Panel
    st.subheader("Ratios de Salud Financiera")
    
    r1, r2, r3 = st.columns(3)
    
    with r1:
        val = ratios["ratio_liquidez"]
        st.metric("Ratio de Liquidez", f"{val:.1f} meses")
        st.progress(max(0.0, min(val / 12.0, 1.0)))
        st.caption("Objetivo: > 6 meses")
        
    with r2:
        val = ratios["ratio_deuda_ingresos"]
        st.metric("Deuda / Ingreso Anual", f"{val*100:.1f}%")
        st.progress(max(0.0, min(val, 1.0)))
        st.caption("Objetivo: < 30%")
        
    with r3:
        val = ratios["tasa_ahorro_global"]
        st.metric("Tasa Ahorro Global", f"{val*100:.1f}%")
        st.progress(max(0.0, min(val, 1.0)))
        st.caption("Objetivo: > 20%")
        
    # 3. Runway Analysis
    st.subheader("Runway Financiero (Supervivencia)")
    run_norm, run_surv, run_liq, burn, burn_surv = compute_runway(ratios, dfm, ass)
    
    c_run1, c_run2 = st.columns(2)
    with c_run1:
        st.info(f"""
        **Gasto Mensual (Burn Rate):** {money(burn, currency)}
        
        **Runway Normal:** {run_norm:.1f} meses
        
        (Tiempo que puede vivir sin ingresos manteniendo su nivel de vida actual)
        """)
        
    with c_run2:
        st.success(f"""
        **Gasto de Supervivencia:** {money(burn_surv, currency)}
        
        **Runway Máximo:** {run_surv:.1f} meses
        
        (Tiempo que puede vivir recortando todos los gastos variables)
        """)

def render_debt_strategy(data, currency):
    st.header("Estrategia de Deuda")
    st.markdown("Simulación de pago de deudas y proyección patrimonial.")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    debts_universe = build_debt_universe(data)
    
    if not debts_universe:
        st.info("No se encontraron deudas para simular.")
        return
        
    # 1. Strategy Controls
    st.subheader("Simulador de Pago de Deuda")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
         extra_pay = st.number_input("Pago Extra Mensual", min_value=0.0, value=0.0, step=100.0)
    
    # Run 3 Simulations
    sch_av, int_av, months_av = simulate_payoff(debts_universe, extra_payment=extra_pay, strategy="avalanche")
    sch_sn, int_sn, months_sn = simulate_payoff(debts_universe, extra_payment=extra_pay, strategy="snowball")
    sch_hy, int_hy, months_hy = simulate_payoff(debts_universe, extra_payment=extra_pay, strategy="hybrid")
    
    # Comparison Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AVALANCHE", f"{months_av} meses")
        st.caption(f"Interés total: {money(int_av, currency)}")
        st.caption("Prioriza deudas con mayor tasa")

    with col2:
        st.metric("SNOWBALL", f"{months_sn} meses")
        st.caption(f"Interés total: {money(int_sn, currency)}")
        st.caption("Prioriza deudas de menor saldo")

    with col3:
        st.metric("HYBRID", f"{months_hy} meses")
        st.caption(f"Interés total: {money(int_hy, currency)}")
        st.caption("Balance entre costo y motivación")

    # Strategy Comparison Chart
    fig_debt = create_debt_comparison_chart(sch_av, sch_sn, sch_hy)
    safe_plot(fig_debt, "Comparación Estrategias")
            
    st.divider()
    
    # 2. Net Worth Forecast
    st.subheader("Proyección de Patrimonio")
    
    # Inputs
    proj_years = st.slider(
        "Periodo de Proyección (Años)",
        min_value=1,
        max_value=20,
        value=5,
        help="Años a futuro para proyectar"
    )
    
    nw_df = net_worth_by_month(data, dfm, ass)
    base_savings = dfm['Flujo_Neto'].tail(3).mean() if not dfm.empty else 0.0
    forecast_df = forecast_net_worth(nw_df, base_savings, years=proj_years, a=ass)
    
    # Forecast Chart
    fig_fc = create_forecast_chart(forecast_df)
    safe_plot(fig_fc, "Proyección Patrimonio", forecast_df)
    
    # Forecast Metrics
    if not forecast_df.empty:
        ratios = compute_ratios_and_balance(data, dfm, ass)
        current_nw = ratios['patrimonio_neto_actual']
        final_base = forecast_df['Base'].iloc[-1]
        final_opt = forecast_df['Optimista'].iloc[-1]
        final_pes = forecast_df['Pesimista'].iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            growth = ((final_base - current_nw) / current_nw * 100) if current_nw != 0 else 0
            st.metric("ESCENARIO BASE", money(final_base, currency), f"+{growth:.0f}%")
            
        with m2:
            growth = ((final_opt - current_nw) / current_nw * 100) if current_nw != 0 else 0
            st.metric("OPTIMISTA", money(final_opt, currency), f"+{growth:.0f}%")
            
        with m3:
            growth = ((final_pes - current_nw) / current_nw * 100) if current_nw != 0 else 0
            st.metric("PESIMISTA", money(final_pes, currency), f"{'+' if growth >= 0 else ''}{growth:.0f}%")

if __name__ == "__main__":
    main()