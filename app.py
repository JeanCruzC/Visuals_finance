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
DESIGN_SYSTEM = {
    "primary": "#1E3A8A",      # Navy Blue
    "secondary": "#059669",     # Emerald Green
    "accent": "#DC2626",        # Red
    "background": "#F9FAFB",    # Light Gray
    "text_dark": "#1F2937",
    "text_light": "#6B7280",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444"
}

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "size": 12, "color": DESIGN_SYSTEM["text_dark"]},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 40, "r": 20, "t": 30, "b": 40},
        "xaxis": {"showgrid": False, "zeroline": False},
        "yaxis": {"showgrid": True, "gridcolor": "#E5E7EB", "zeroline": False},
        "colorway": [DESIGN_SYSTEM["primary"], DESIGN_SYSTEM["secondary"], "#F59E0B", "#10B981", "#6366F1"]
    }
}

# CSS Injection
def inject_custom_css():
    st.markdown("""
        <style>
        /* Global Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Scoped Components (.fp-) */
        .fp-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .fp-metric-label {
            font-size: 0.875rem;
            color: #6B7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .fp-metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin: 0.25rem 0;
        }
        
        .fp-metric-delta {
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .fp-positive { color: #10B981; }
        .fp-negative { color: #EF4444; }
        .fp-neutral { color: #6B7280; }

        /* Responsive */
        @media (max-width: 768px) {
            .stColumns { display: flex; flex-direction: column; }
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
        
        ratios["ratio_liquidez"] = safe_div(current_liquidity, avg_monthly_spend)
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
# 5. VIEW LOGIC
# -----------------------------------------------------------------------------

def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Overview"
    
    with st.sidebar:
        st.title("💳 Financial Pilot")
        
        page = st.radio(
            "Navigation", 
            ["Overview", "Cash Flow", "Balance Sheet", "Debt Strategy"],
            index=["Overview", "Cash Flow", "Balance Sheet", "Debt Strategy"].index(st.session_state.current_page)
        )
        st.session_state.current_page = page
        
        st.divider()
        st.markdown("### Settings")
        currency = st.selectbox("Currency", ["USD", "EUR", "PEN"])
        
        uploaded_file = st.file_uploader("Upload Excel Data", type=["xlsx"])

    if not uploaded_file:
        st.info("👋 Welcome! Please upload your 'Finanzas_Personales_Analisis_Horario.xlsx' file to begin.")
        return

    data = load_excel_with_profiling(uploaded_file)
    if not data:
        return

    if page == "Overview":
        render_overview(data, currency)
    elif page == "Cash Flow":
        render_cashflow(data, currency)
    elif page == "Balance Sheet":
        render_balancesheet(data, currency)
    elif page == "Debt Strategy":
        render_debt_strategy(data, currency)

def render_overview(data, currency):
    st.header("Overview")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    
    if dfm.empty:
        st.warning("Not enough data to generate overview.")
        return

    all_months = sorted(dfm["Mes"].unique())
    col_filter, col_export = st.columns([3, 1])
    with col_filter:
        if len(all_months) > 1:
            start, end = st.select_slider("Date Range", options=all_months, value=(all_months[0], all_months[-1]))
        else:
            start, end = (all_months[0], all_months[0]) if all_months else (None, None)
            
    dfm = dfm[(dfm["Mes"] >= start) & (dfm["Mes"] <= end)].copy()
    ratios = compute_ratios_and_balance(data, dfm, ass)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Net Worth", money(ratios["patrimonio_neto_actual"], currency))
    with k2: kpi_card("Savings Rate", f"{ratios['tasa_ahorro_global']*100:.1f}%")
    with k3: kpi_card("Liquidity (Mo)", f"{ratios['ratio_liquidez']:.1f}")
    with k4: kpi_card("Debt/Income", f"{ratios['ratio_deuda_ingresos']*100:.1f}%")
    
    st.divider()
    
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader("Net Worth Evolution")
        nw_df = net_worth_by_month(data, dfm, ass)
        if not nw_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nw_df["Mes"], y=nw_df["Patrimonio_Neto"], fill='tozeroy', mode='lines', name='Net Worth', line=dict(color=DESIGN_SYSTEM["primary"])))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
    with c_right:
        st.subheader("Cash Flow (Period)")
        # Waterfall
        wf_ing = dfm["Ingresos_Netos"].sum()
        wf_exp = dfm["Gastos_Totales"].sum()
        wf_net = dfm["Flujo_Neto"].sum()
        
        fig_wf = go.Figure(go.Waterfall(
            measure=["relative", "relative", "total"],
            x=["Income", "Expenses", "Net Flow"],
            y=[wf_ing, -wf_exp, wf_net],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_wf.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_export:
         csv = dfm.to_csv(index=False).encode('utf-8')
         st.download_button("📥 Export CSV", data=csv, file_name="overview_data.csv", mime="text/csv")

def render_cashflow(data, currency):
    st.header("Cash Flow Analysis")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, ing, _, gv, _, _ = build_dynamic_table(data, ass)
    
    if dfm.empty:
        st.warning("No data available.")
        return

    # 1. Main Trend (Combo Chart)
    st.subheader("Income vs Expenses Trend")
    
    fig_trend = go.Figure()
    # Net Flow as Bars
    fig_trend.add_trace(go.Bar(
        x=dfm["Mes"], y=dfm["Flujo_Neto"], name="Net Flow",
        marker_color="#3B82F6", opacity=0.6
    ))
    # Income as Line
    fig_trend.add_trace(go.Scatter(
        x=dfm["Mes"], y=dfm["Ingresos_Netos"], name="Total Income",
        mode='lines+markers', line=dict(color=DESIGN_SYSTEM["secondary"], width=3)
    ))
    # Expenses as Line
    fig_trend.add_trace(go.Scatter(
        x=dfm["Mes"], y=dfm["Gastos_Totales"], name="Total Expenses",
        mode='lines+markers', line=dict(color=DESIGN_SYSTEM["accent"], width=3)
    ))
    
    fig_trend.update_layout(template=PLOTLY_TEMPLATE, height=400, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 2. Variable Expenses Breakdown
    st.subheader("Variable Expenses Breakdown")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        if not gv.empty and "Categoría" in gv.columns:
            gv_grouped = gv.groupby(["Periodo", "Categoría"])["Monto"].sum().reset_index()
            gv_grouped["Monto"] = gv_grouped["Monto"].abs() # Display positive
            
            fig_stack = px.bar(
                gv_grouped, x="Periodo", y="Monto", color="Categoría",
                title="Variable Expenses by Category",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_stack.update_layout(template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
             st.info("No variable expenses data.")
             
    with c2:
        st.markdown("**Monthly Statistics**")
        avg_inc = dfm["Ingresos_Netos"].mean()
        avg_exp = dfm["Gastos_Totales"].mean()
        avg_save = dfm["Flujo_Neto"].mean()
        
        st.metric("Avg Monthly Income", money(avg_inc, currency))
        st.metric("Avg Monthly Expenses", money(avg_exp, currency))
        st.metric("Avg Monthly Savings", money(avg_save, currency), delta_color="normal")
        
    st.markdown("### Detailed Monthly Data")
    professional_table(dfm[["Mes", "Ingresos_Netos", "Gastos_Fijos", "Gastos_Variables", "Pagos_Deuda", "Gastos_Totales", "Flujo_Neto"]])

def render_balancesheet(data, currency):
    st.header("Balance Sheet & Ratios")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    ratios = compute_ratios_and_balance(data, dfm, ass)
    
    # 1. Asset Allocation & Composition
    st.subheader("Asset Allocation")
    
    inv_val = ratios["inversiones_valor"]
    liq_val = ratios["activos_liquidos"]
    bienes_val = ass.bienes_value
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Pie Chart
        labels = ["Liquidity (Cash)", "Investments", "Hard Assets (Goods)"]
        values = [liq_val, inv_val, bienes_val]
        
        fig_pie = px.pie(names=labels, values=values, hole=0.4, title="Total Assets Distribution")
        fig_pie.update_traces(textinfo='percent+label', marker=dict(colors=[DESIGN_SYSTEM["success"], DESIGN_SYSTEM["primary"], "#9CA3AF"]))
        fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        # Metrics Stack
        st.metric("Total Assets", money(liq_val + inv_val + bienes_val, currency))
        st.metric("Total Liabilities", money(ratios["pasivos_totales"], currency), delta_color="inverse")
        st.metric("Net Worth", money(ratios["patrimonio_neto_actual"], currency))
        
    st.divider()
    
    # 2. Financial Ratios Panel
    st.subheader("Financial Health Ratios")
    
    r1, r2, r3 = st.columns(3)
    
    with r1:
        # Liquidity Ratio
        val = ratios["ratio_liquidez"]
        st.metric("Liquidity Ratio (Months)", f"{val:.1f}")
        st.progress(min(val / 12.0, 1.0))
        st.caption("Target: > 6 months")
        
    with r2:
        # Debt to Income
        val = ratios["ratio_deuda_ingresos"]
        st.metric("Debt-to-Income (Annual)", f"{val*100:.1f}%")
        st.progress(min(val, 1.0))
        st.caption("Target: < 30%")
        
    with r3:
        # Savings Rate
        val = ratios["tasa_ahorro_global"]
        st.metric("Global Savings Rate", f"{val*100:.1f}%")
        st.progress(max(0.0, min(val, 1.0)))
        st.caption("Target: > 20%")
        
    # 3. Runway Analysis
    st.subheader("Runway Analysis")
    run_norm, run_surv, run_liq, burn, burn_surv = compute_runway(ratios, dfm, ass)
    
    c_run1, c_run2 = st.columns(2)
    with c_run1:
        st.info(f"""
        **Burn Rate (Monthly Spend):** {money(burn, currency)}
        
        **Normal Runway:** {run_norm:.1f} months
        
        (Time you can survive without income at current spending)
        """)
        
    with c_run2:
        st.success(f"""
        **Survival Burn Rate (Fixed + Debt):** {money(burn_surv, currency)}
        
        **Survival Runway:** {run_surv:.1f} months
        
        (Time you can survive cutting all variable expenses)
        """)

def render_debt_strategy(data, currency):
    st.header("Debt Strategy & Projections")
    
    ass = Assumptions(currency_symbol=currency)
    dfm, _, _, _, _, _, _ = build_dynamic_table(data, ass)
    debts_universe = build_debt_universe(data)
    
    if not debts_universe:
        st.success("No debts found! You are debt free.")
    else:
        st.subheader("Debt Payoff Simulation")
        
        # User Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
             extra_pay = st.number_input("Monthly Extra Payment", min_value=0.0, value=float(ass.extra_debt_payment_monthly), step=100.0)
        with col_ctrl2:
             strategy = st.selectbox("Strategy", ["Avalanche (High Interest First)", "Snowball (Low Balance First)"])
             strat_code = "avalanche" if "Avalanche" in strategy else "snowball"
        
        # Run Simulation
        sch, total_int, months = simulate_payoff(debts_universe, extra_payment=extra_pay, strategy=strat_code)
        
        # Comparison (Baseline)
        _, base_int, base_months = simulate_payoff(debts_universe, extra_payment=0, strategy="avalanche")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Time to Debt Free", f"{months} months", delta=f"{base_months - months} months saved")
        m2.metric("Total Interest", money(total_int, currency), delta=money(base_int - total_int, currency), delta_color="inverse")
        m3.metric("Extra Payment", money(extra_pay, currency))
        
        # Timeline Chart
        if not sch.empty:
            # Aggregate balance by month
            sch_agg = sch[sch["type"] == "min"].groupby("month")["balance_end"].sum().reset_index()
            # If extra payments exist, they might not be in 'min' rows solely? logic check.
            # Actually schedule has rows for min and extra. We want balance at End of Month. 
            # Ideally we group by month and take sum of last balance of each debt? No, simulation step updates balance.
            # We can just sum balance_end where type is last action for that debt in month?
            # Simpler: The simulation updates debt object in place but we log it.
            # Let's re-process schedule to get Total Balance Time Series.
            
            # Group by month, sum balance_end
            # NOTE: Logic in simulate_payoff appends a "balance" snapshot row at end of loop? No, it appends per payment.
            # Let's fix simulate_payoff output to be easier to chart or interpret here.
            # Actually, let's just use the 'balance_end' from the dataframe.
            # We need to sum balance_end for distinct debt_ids for each month.
            
            # Pivot?
            # Easier: Sum all 'balance_end' for entries where type='min' (assuming min payment happens for all debts once per month)
            # Or just filter unique debt_id per month taking the min() of balance_end?
            daily_bal = sch.groupby(["month", "debt_id"])["balance_end"].min().reset_index()
            monthly_total = daily_bal.groupby("month")["balance_end"].sum().reset_index()
            
            fig_pay = px.line(monthly_total, x="month", y="balance_end", title="Debt Payoff Timeline")
            fig_pay.add_trace(go.Scatter(x=monthly_total["month"], y=monthly_total["balance_end"], fill='tozeroy', mode='none'))
            fig_pay.update_layout(template=PLOTLY_TEMPLATE, height=350)
            st.plotly_chart(fig_pay, use_container_width=True)
            
    st.divider()
    
    # 2. Net Worth Forecast
    st.subheader("Net Worth Forecast")
    
    nw_df = net_worth_by_month(data, dfm, ass)
    base_sav = float(dfm["Flujo_Neto"].tail(3).mean()) if len(dfm) else 0.0
    
    col_proj1, col_proj2 = st.columns([1, 2])
    with col_proj1:
         years = st.slider("Forecast Years", 1, 20, 5)
         st.write(f"Based on avg monthly savings: **{money(base_sav, currency)}**")
         
    with col_proj2:
         forecast = forecast_net_worth(nw_df, base_sav, years=years, a=ass)
         if not forecast.empty:
             fig_fc = go.Figure()
             fig_fc.add_trace(go.Scatter(x=forecast["Mes_Futuro"], y=forecast["Base"], name="Base Scenario", line=dict(color=DESIGN_SYSTEM["primary"])))
             fig_fc.add_trace(go.Scatter(x=forecast["Mes_Futuro"], y=forecast["Optimista"], name="Optimistic (+20%)", line=dict(dash='dash', color=DESIGN_SYSTEM["success"])))
             fig_fc.add_trace(go.Scatter(x=forecast["Mes_Futuro"], y=forecast["Pesimista"], name="Pessimistic (-20%)", line=dict(dash='dash', color=DESIGN_SYSTEM["error"])))
             fig_fc.update_layout(template=PLOTLY_TEMPLATE, height=400, title="Projected Net Worth")
             st.plotly_chart(fig_fc, use_container_width=True)

if __name__ == "__main__":
    main()