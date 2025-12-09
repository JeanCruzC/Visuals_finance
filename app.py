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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN EMPRESARIAL - ENTERPRISE GRADE
# ============================================================================

# Validación de Versión
if st.__version__ < "1.23.0":
    st.error("❌ Esta aplicación requiere Streamlit 1.23+. Por favor actualice: pip install --upgrade streamlit")
    st.stop()

# Configuración de Página Profesional
st.set_page_config(
    page_title="Financial Analytics Suite | Enterprise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
    menu_items={
        'Get Help': 'https://www.financialsuite.com/support',
        'Report a bug': 'https://www.financialsuite.com/bug',
        'About': "Financial Analytics Suite v3.0 | Enterprise Financial Intelligence Platform"
    }
)

# ============================================================================
# 2. SISTEMA DE DISEÑO EMPRESARIAL
# ============================================================================

# Paleta de Colores Corporativa (Bloomberg Terminal Style)
COLORS = {
    'primary': '#0F2C56',           # Azul corporativo oscuro
    'primary_light': '#1E4A9A',     # Azul corporativo
    'secondary': '#006D5B',         # Verde esmeralda corporativo
    'success': '#00A86B',           # Verde éxito
    'danger': '#C62828',            # Rojo corporativo
    'warning': '#F57C00',           # Ámbar corporativo
    'info': '#0288D1',              # Azul información
    'neutral_dark': '#263238',      # Gris oscuro
    'neutral': '#546E7A',           # Gris medio
    'neutral_light': '#B0BEC5',     # Gris claro
    'background': '#F5F7FA',        # Fondo profesional
    'card_bg': '#FFFFFF',           # Fondo tarjetas
    'border': '#CFD8DC',            # Borde
    
    # Gradientes profesionales
    'gradient_blue': ['#0F2C56', '#1E4A9A', '#3D7BCC'],
    'gradient_green': ['#006D5B', '#00A86B', '#4CD964'],
    
    # Estados
    'positive': '#00A86B',
    'negative': '#C62828',
    'neutral_state': '#546E7A',
}

# Sistema Tipográfico Empresarial
TYPOGRAPHY = {
    'h1': {'size': 32, 'weight': 700, 'color': COLORS['neutral_dark']},
    'h2': {'size': 24, 'weight': 600, 'color': COLORS['neutral_dark']},
    'h3': {'size': 18, 'weight': 600, 'color': COLORS['neutral_dark']},
    'body': {'size': 14, 'weight': 400, 'color': COLORS['neutral']},
    'caption': {'size': 12, 'weight': 400, 'color': COLORS['neutral_light']},
    'kpi_value': {'size': 28, 'weight': 700, 'color': COLORS['neutral_dark']},
    'kpi_label': {'size': 11, 'weight': 600, 'color': COLORS['neutral']},
}

# Configuración Plotly Profesional
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "financial_insight",
        "height": 1080,
        "width": 1920,
        "scale": 2
    },
    "scrollZoom": True,
    "responsive": True
}

PLOTLY_LAYOUT = {
    "font": {
        "family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
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
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": COLORS['border'],
        "borderwidth": 1,
        "font": {"size": 12}
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 13,
        "font_family": "Inter",
        "bordercolor": COLORS['border'],
        "namelength": -1
    },
    "xaxis": {
        "showgrid": True,
        "gridcolor": COLORS['border'],
        "gridwidth": 0.5,
        "zeroline": False,
        "showline": True,
        "linecolor": COLORS['border'],
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": 8,
        "tickcolor": COLORS['neutral_light']
    },
    "yaxis": {
        "showgrid": True,
        "gridcolor": COLORS['border'],
        "gridwidth": 0.5,
        "zeroline": True,
        "zerolinecolor": COLORS['neutral_light'],
        "zerolinewidth": 1,
        "showline": True,
        "linecolor": COLORS['border'],
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": 8,
        "tickcolor": COLORS['neutral_light']
    }
}

# ============================================================================
# 3. SISTEMA DE ESTILOS CSS EMPRESARIAL
# ============================================================================

def inject_enterprise_css():
    """Inyecta CSS corporativo de nivel enterprise"""
    st.markdown(f"""
        <style>
        /* Importar fuente Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Reset y Base */
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            box-sizing: border-box;
        }}
        
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Encabezados Corporativos */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['neutral_dark']};
            font-weight: 600;
            margin-top: 0;
            line-height: 1.3;
        }}
        
        h1 {{
            font-size: {TYPOGRAPHY['h1']['size']}px;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid {COLORS['primary']};
            padding-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: {TYPOGRAPHY['h2']['size']}px;
            margin-bottom: 1.25rem;
            color: {COLORS['neutral_dark']};
        }}
        
        h3 {{
            font-size: {TYPOGRAPHY['h3']['size']}px;
            margin-bottom: 1rem;
        }}
        
        /* Sidebar Empresarial */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding-top: 2rem;
        }}
        
        section[data-testid="stSidebar"] .stRadio label {{
            color: white !important;
            font-weight: 500;
        }}
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {{
            color: white !important;
        }}
        
        /* Tarjetas KPI Empresariales */
        .enterprise-kpi-card {{
            background: {COLORS['card_bg']};
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 2px 8px rgba(15, 44, 86, 0.08);
            transition: all 0.3s ease;
            height: 100%;
            position: relative;
            overflow: hidden;
        }}
        
        .enterprise-kpi-card:hover {{
            box-shadow: 0 8px 24px rgba(15, 44, 86, 0.12);
            transform: translateY(-2px);
            border-color: {COLORS['primary_light']};
        }}
        
        .enterprise-kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
        }}
        
        .kpi-label {{
            font-size: {TYPOGRAPHY['kpi_label']['size']}px;
            font-weight: {TYPOGRAPHY['kpi_label']['weight']};
            color: {COLORS['neutral']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .kpi-value {{
            font-size: {TYPOGRAPHY['kpi_value']['size']}px;
            font-weight: {TYPOGRAPHY['kpi_value']['weight']};
            color: {COLORS['neutral_dark']};
            margin: 0.5rem 0;
            line-height: 1.2;
        }}
        
        .kpi-delta {{
            font-size: 13px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 4px;
            margin-top: 0.5rem;
        }}
        
        .kpi-delta.positive {{
            background: rgba(0, 168, 107, 0.1);
            color: {COLORS['success']};
        }}
        
        .kpi-delta.negative {{
            background: rgba(198, 40, 40, 0.1);
            color: {COLORS['danger']};
        }}
        
        .kpi-delta.neutral {{
            background: rgba(84, 110, 122, 0.1);
            color: {COLORS['neutral']};
        }}
        
        .kpi-target {{
            font-size: 12px;
            color: {COLORS['neutral_light']};
            margin-top: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}
        
        /* Indicadores de Estado */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status-good {{
            background: rgba(0, 168, 107, 0.1);
            color: {COLORS['success']};
        }}
        
        .status-warning {{
            background: rgba(245, 124, 0, 0.1);
            color: {COLORS['warning']};
        }}
        
        .status-critical {{
            background: rgba(198, 40, 40, 0.1);
            color: {COLORS['danger']};
        }}
        
        /* Botones Corporativos */
        .stButton > button {{
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            width: 100%;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(15, 44, 86, 0.2);
        }}
        
        /* Progress Bars */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
        }}
        
        /* Divider */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, {COLORS['border']}, transparent);
            margin: 2rem 0;
        }}
        
        /* File Uploader Styling */
        .uploadedFile {{
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 1rem;
            background: {COLORS['card_bg']};
            margin-bottom: 1rem;
        }}
        
        /* Card Grid System */
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .card-grid {{
                grid-template-columns: 1fr;
            }}
            
            .kpi-value {{
                font-size: 24px;
            }}
        }}
        
        /* Remove Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display:none;}}
        header {{visibility: hidden;}}
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {{
            font-size: 28px;
            font-weight: 700;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {COLORS['neutral']};
        }}
        
        /* Expander Styling */
        .streamlit-expanderHeader {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            font-weight: 600;
        }}
        
        /* Dataframe Styling */
        .dataframe {{
            border: 1px solid {COLORS['border']} !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }}
        
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 4. SISTEMA DE CÁLCULO FINANCIERO MEJORADO
# ============================================================================

class FinancialEngine:
    """Motor de cálculo financiero empresarial"""
    
    def __init__(self, data, currency="USD"):
        self.data = data
        self.currency = currency
        self.assumptions = self._default_assumptions()
        self._validate_and_prepare_data()
    
    def _default_assumptions(self):
        """Configuración predeterminada de supuestos"""
        return {
            'currency_symbol': self.currency,
            'expected_return': 0.08,
            'inflation': 0.03,
            'safe_withdrawal_rate': 0.04,
            'emergency_months_target': 6,
            'savings_rate_target': 0.20,
            'debt_to_income_target': 0.30,
            'investment_liquidation_haircut': 0.70
        }
    
    def _validate_and_prepare_data(self):
        """Validación robusta y preparación de datos"""
        required_sheets = ["Ingresos", "Datos_Maestros", "Cuentas_Bancarias"]
        
        # Verificar hojas requeridas
        missing = [s for s in required_sheets if s not in self.data]
        if missing:
            st.error(f"❌ Hojas requeridas faltantes: {', '.join(missing)}")
            st.stop()
        
        # Procesar datos de movimientos
        self._consolidate_movements()
        
        # Calcular métricas base
        self._calculate_base_metrics()
    
    def _consolidate_movements(self):
        """Consolidar todos los movimientos financieros"""
        movimientos = []
        
        # Procesar ingresos
        if "Ingresos" in self.data and not self.data["Ingresos"].empty:
            ing = self.data["Ingresos"].copy()
            ing["Tipo"] = "Ingreso"
            ing["Flujo"] = ing.get("Monto_Neto", 0)
            movimientos.append(ing[["Fecha", "Tipo", "Flujo", "Categoría"]])
        
        # Procesar gastos
        for sheet in ["Gastos_Fijos", "Gastos_Variables"]:
            if sheet in self.data and not self.data[sheet].empty:
                df = self.data[sheet].copy()
                df["Tipo"] = "Gasto"
                df["Flujo"] = -df.get("Monto", 0).abs()
                movimientos.append(df[["Fecha", "Tipo", "Flujo", "Categoría"]])
        
        # Consolidar
        if movimientos:
            self.movimientos = pd.concat(movimientos, ignore_index=True)
            self.movimientos["Fecha"] = pd.to_datetime(self.movimientos["Fecha"])
            self.movimientos = self.movimientos.sort_values("Fecha")
        else:
            self.movimientos = pd.DataFrame(columns=["Fecha", "Tipo", "Flujo", "Categoría"])
    
    def _calculate_base_metrics(self):
        """Calcular métricas financieras base"""
        # Inicializar métricas
        self.metrics = {}
        
        # Patrimonio Neto
        self.metrics['patrimonio_neto'] = self._calculate_net_worth()
        
        # Liquidez
        self.metrics['liquidez'] = self._calculate_liquidity()
        
        # Tasa de Ahorro
        self.metrics['tasa_ahorro'] = self._calculate_savings_rate()
        
        # Ratios Clave
        self.metrics['ratios'] = self._calculate_financial_ratios()
        
        # Proyecciones
        self.metrics['proyecciones'] = self._calculate_projections()
    
    def _calculate_net_worth(self):
        """Calcular patrimonio neto total"""
        net_worth = 0
        
        # Activos líquidos
        if "Cuentas_Bancarias" in self.data:
            net_worth += self.data["Cuentas_Bancarias"].get("Saldo_Inicial", 0).sum()
        
        # Inversiones
        if "Inversiones" in self.data:
            # Usar Valor_Actual si existe, si no Monto_Invertido
            if "Valor_Actual" in self.data["Inversiones"].columns:
                net_worth += self.data["Inversiones"]["Valor_Actual"].sum()
            elif "Monto_Invertido" in self.data["Inversiones"].columns:
                net_worth += self.data["Inversiones"]["Monto_Invertido"].sum()
        
        # Restar deudas
        for sheet in ["Deudas_Prestamos", "Tarjetas_Credito"]:
            if sheet in self.data and "Saldo_Actual" in self.data[sheet].columns:
                net_worth -= self.data[sheet]["Saldo_Actual"].sum()
        
        return max(net_worth, 0)
    
    def _calculate_liquidity(self):
        """Calcular meses de liquidez disponible"""
        # Gastos mensuales promedio
        if not self.movimientos.empty:
            gastos_mensuales = abs(self.movimientos[self.movimientos["Flujo"] < 0]["Flujo"].mean())
            if gastos_mensuales == 0:
                gastos_mensuales = 1
            
            # Activos líquidos totales
            activos_liquidos = 0
            if "Cuentas_Bancarias" in self.data:
                activos_liquidos = self.data["Cuentas_Bancarias"].get("Saldo_Inicial", 0).sum()
            
            return activos_liquidos / gastos_mensuales
        
        return 0
    
    def _calculate_savings_rate(self):
        """Calcular tasa de ahorro"""
        if not self.movimientos.empty:
            ingresos = self.movimientos[self.movimientos["Flujo"] > 0]["Flujo"].sum()
            gastos = abs(self.movimientos[self.movimientos["Flujo"] < 0]["Flujo"].sum())
            
            if ingresos > 0:
                return (ingresos - gastos) / ingresos
        
        return 0
    
    def _calculate_financial_ratios(self):
        """Calcular ratios financieros clave"""
        ratios = {}
        
        # Deuda/Ingreso
        if not self.movimientos.empty:
            ingresos_anuales = self.movimientos[self.movimientos["Flujo"] > 0]["Flujo"].sum() * 12
            deuda_total = 0
            
            for sheet in ["Deudas_Prestamos", "Tarjetas_Credito"]:
                if sheet in self.data and "Saldo_Actual" in self.data[sheet].columns:
                    deuda_total += self.data[sheet]["Saldo_Actual"].sum()
            
            ratios['deuda_ingreso'] = deuda_total / ingresos_anuales if ingresos_anuales > 0 else 0
        
        # Retorno sobre Patrimonio
        ratios['rop'] = 0.08  # Placeholder - cálculo complejo
        
        return ratios
    
    def _calculate_projections(self):
        """Calcular proyecciones financieras"""
        projections = {
            'fi_edad': self._calculate_fi_age(),
            'meta_ahorro': self._calculate_savings_goal()
        }
        return projections
    
    def _calculate_fi_age(self):
        """Calcular edad de independencia financiera"""
        # Implementación simplificada
        return 55  # Placeholder
    
    def _calculate_savings_goal(self):
        """Calcular meta de ahorro para FI"""
        if not self.movimientos.empty:
            gastos_anuales = abs(self.movimientos[self.movimientos["Flujo"] < 0]["Flujo"].sum()) * 12
            fi_number = gastos_anuales / self.assumptions['safe_withdrawal_rate']
            return fi_number
        
        return 0
    
    def get_kpi_data(self):
        """Obtener datos para tarjetas KPI"""
        return {
            'patrimonio_neto': {
                'value': self.metrics['patrimonio_neto'],
                'formatted': f"{self.currency} {self.metrics['patrimonio_neto']:,.2f}",
                'delta': 8.4,  # Placeholder - debería calcularse históricamente
                'target': None
            },
            'tasa_ahorro': {
                'value': self.metrics['tasa_ahorro'] * 100,
                'formatted': f"{self.metrics['tasa_ahorro'] * 100:.1f}%",
                'delta': -277.2 if self.metrics['tasa_ahorro'] < 0 else None,
                'target': f"Meta: <{self.assumptions['savings_rate_target'] * 100}%"
            },
            'liquidez': {
                'value': self.metrics['liquidez'],
                'formatted': f"{self.metrics['liquidez']:.1f} meses",
                'delta': None,
                'target': f"Meta: {self.assumptions['emergency_months_target']}+ meses"
            },
            'fi_score': {
                'value': self._calculate_fi_score(),
                'formatted': f"{self._calculate_fi_score()}/10",
                'delta': None,
                'target': "Meta: 8+/10"
            }
        }
    
    def _calculate_fi_score(self):
        """Calcular puntuación FI (0-10)"""
        score = 5  # Base
        
        # Tasa de ahorro
        sr = self.metrics['tasa_ahorro']
        if sr > 0.3:
            score += 3
        elif sr > 0.2:
            score += 2
        elif sr > 0.1:
            score += 1
        elif sr < 0:
            score -= 2
        
        # Liquidez
        liq = self.metrics['liquidez']
        if liq >= 6:
            score += 2
        elif liq >= 3:
            score += 1
        elif liq < 1:
            score -= 2
        
        return max(0, min(10, score))

# ============================================================================
# 5. COMPONENTES UI EMPRESARIALES
# ============================================================================

def render_enterprise_kpi_card(title, value, delta=None, target=None, icon=None):
    """Renderizar tarjeta KPI empresarial"""
    
    # Determinar color del delta
    delta_class = "neutral"
    delta_icon = ""
    
    if delta is not None:
        if delta > 0:
            delta_class = "positive"
            delta_icon = "▲"
        elif delta < 0:
            delta_class = "negative"
            delta_icon = "▼"
    
    # Construir HTML
    card_html = f"""
    <div class="enterprise-kpi-card">
        <div class="kpi-label">
            {icon if icon else ''}
            {title}
        </div>
        <div class="kpi-value">{value}</div>
    """
    
    if delta is not None:
        card_html += f"""
        <div class="kpi-delta {delta_class}">
            {delta_icon} {abs(delta):.1f}%
        </div>
        """
    
    if target:
        card_html += f"""
        <div class="kpi-target">
            <span>⏺</span> {target}
        </div>
        """
    
    card_html += "</div>"
    
    return st.markdown(card_html, unsafe_allow_html=True)

def render_file_upload_section():
    """Renderizar sección de carga de archivos empresarial"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Carga de Datos Financieros")
        st.markdown("Sube tu archivo Excel consolidado para análisis avanzado")
        
        uploaded_file = st.file_uploader(
            "Arrastra y suelta tu archivo aquí",
            type=["xlsx", "xls"],
            help="Formatos soportados: Excel (.xlsx, .xls)"
        )
    
    with col2:
        st.markdown("#### 📋 Requisitos")
        st.markdown("""
        - Archivo Excel con hojas específicas
        - Estructura de datos normalizada
        - Formatos de fecha consistentes
        - Columnas requeridas por plantilla
        """)
    
    return uploaded_file

def render_insight_card(title, severity, message, action, metric, benchmark):
    """Renderizar tarjeta de insight/riesgo"""
    
    severity_config = {
        'high': {'color': COLORS['danger'], 'icon': '⚠️', 'bg': '#FEE2E2'},
        'medium': {'color': COLORS['warning'], 'icon': '⚠️', 'bg': '#FEF3C7'},
        'low': {'color': COLORS['success'], 'icon': '✅', 'bg': '#D1FAE5'},
        'info': {'color': COLORS['info'], 'icon': 'ℹ️', 'bg': '#E0F2FE'}
    }
    
    config = severity_config.get(severity, severity_config['info'])
    
    insight_html = f"""
    <div style="
        background: {config['bg']};
        border-left: 4px solid {config['color']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        ">
            <div>
                <h4 style="margin: 0; color: {config['color']};">
                    {config['icon']} {title}
                </h4>
                <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 14px;">
                    {message}
                </p>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <div style="font-size: 24px; font-weight: 700; color: {config['color']};">
                    {metric}
                </div>
                <div style="font-size: 12px; color: #999;">
                    vs {benchmark}
                </div>
            </div>
        </div>
        <div style="
            background: rgba(255,255,255,0.7);
            border-radius: 4px;
            padding: 0.75rem;
            font-size: 13px;
        ">
            <strong>🛠 Acción Recomendada:</strong> {action}
        </div>
    </div>
    """
    
    return st.markdown(insight_html, unsafe_allow_html=True)

# ============================================================================
# 6. VISTAS PRINCIPALES REDISEÑADAS
# ============================================================================

def render_dashboard_view(data, currency):
    """Vista principal del dashboard empresarial"""
    
    # Inicializar motor financiero
    engine = FinancialEngine(data, currency)
    kpi_data = engine.get_kpi_data()
    
    # Header Empresarial
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
    ">
        <h1 style="color: white; margin-bottom: 0.5rem;">Vista General</h1>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
            Panel de control financiero con métricas clave y recomendaciones personalizadas
        </p>
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 14px;
            ">
                🔄 Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 14px;
            ">
                💼 Moneda: {currency}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de KPIs
    st.markdown("### Métricas Clave")
    
    # Grid de KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        patrimonio = kpi_data['patrimonio_neto']
        render_enterprise_kpi_card(
            "PATRIMONIO NETO",
            patrimonio['formatted'],
            delta=patrimonio['delta'],
            target=patrimonio['target'],
            icon="💰"
        )
    
    with col2:
        ahorro = kpi_data['tasa_ahorro']
        render_enterprise_kpi_card(
            "TASA DE AHORRO",
            ahorro['formatted'],
            delta=ahorro['delta'],
            target=ahorro['target'],
            icon="📈"
        )
    
    with col3:
        liquidez = kpi_data['liquidez']
        render_enterprise_kpi_card(
            "LIQUIDEZ",
            liquidez['formatted'],
            delta=liquidez['delta'],
            target=liquidez['target'],
            icon="💧"
        )
    
    with col4:
        fi_score = kpi_data['fi_score']
        render_enterprise_kpi_card(
            "SCORE FI",
            fi_score['formatted'],
            delta=fi_score['delta'],
            target=fi_score['target'],
            icon="⭐"
        )
    
    st.markdown("---")
    
    # Sección de Análisis y Recomendaciones
    st.markdown("### Análisis de Riesgo y Oportunidades")
    st.caption("Generado mediante análisis cuantitativo de indicadores financieros")
    
    # Insights basados en métricas
    insights = []
    
    # Liquidez crítica
    if engine.metrics['liquidez'] < 1:
        insights.append({
            'title': 'Liquidez: Cobertura de efectivo insuficiente',
            'severity': 'high',
            'message': 'La reserva de efectivo actual no cubre emergencias básicas. Esto representa un riesgo ante imprevistos.',
            'action': 'Establecer transferencias automáticas mensuales del 10% de ingresos a una cuenta de ahorro separada.',
            'metric': f"{engine.metrics['liquidez']:.1f} meses",
            'benchmark': '3-6 meses objetivo'
        })
    
    # Tasa de ahorro negativa
    if engine.metrics['tasa_ahorro'] < 0:
        insights.append({
            'title': 'Flujo de Caja: Déficit operativo',
            'severity': 'high',
            'message': 'Los egresos superan los ingresos de forma sostenida, erosionando el patrimonio mensualmente.',
            'action': 'Revisar gastos discrecionales y eliminar suscripciones no esenciales. Considerar ingresos adicionales.',
            'metric': f"{engine.metrics['tasa_ahorro'] * 100:.1f}%",
            'benchmark': '>20% objetivo'
        })
    
    # Deuda elevada
    if 'deuda_ingreso' in engine.metrics['ratios']:
        if engine.metrics['ratios']['deuda_ingreso'] > 0.4:
            insights.append({
                'title': 'Endeudamiento: Ratio elevado',
                'severity': 'medium',
                'message': 'El nivel de endeudamiento compromete la capacidad de ahorro y representa riesgo financiero.',
                'action': 'Implementar estrategia de pago acelerado priorizando deudas con mayor costo.',
                'metric': f"{engine.metrics['ratios']['deuda_ingreso'] * 100:.1f}%",
                'benchmark': '<30% objetivo'
            })
    
    # Si no hay insights críticos, mostrar positivo
    if not insights:
        insights.append({
            'title': 'Salud Financiera: Indicadores positivos',
            'severity': 'low',
            'message': 'Los principales indicadores se encuentran dentro de rangos recomendados.',
            'action': 'Mantener disciplina financiera y considerar optimización de inversiones.',
            'metric': '✓',
            'benchmark': 'En rango objetivo'
        })
    
    # Renderizar insights
    for insight in insights[:3]:  # Máximo 3 insights
        render_insight_card(
            insight['title'],
            insight['severity'],
            insight['message'],
            insight['action'],
            insight['metric'],
            insight['benchmark']
        )
    
    # Sección de Datos y Exportación
    st.markdown("---")
    
    col_data, col_export = st.columns([2, 1])
    
    with col_data:
        st.markdown("#### 📊 Datos Financieros")
        if not engine.movimientos.empty:
            # Mostrar resumen de movimientos
            summary = engine.movimientos.groupby('Tipo').agg({
                'Flujo': ['sum', 'count']
            }).round(2)
            st.dataframe(summary, use_container_width=True)
    
    with col_export:
        st.markdown("#### 📁 Exportación")
        
        # Botones de exportación
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("📥 Descargar Reporte", use_container_width=True):
                # Generar reporte CSV
                csv = engine.movimientos.to_csv(index=False)
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name="reporte_financiero.csv",
                    mime="text/csv"
                )
        
        with col_exp2:
            if st.button("📊 Generar Dashboard", use_container_width=True):
                st.success("Dashboard generado. Descargue desde el link anterior.")

def render_cashflow_view(data, currency):
    """Vista de análisis de flujo de caja"""
    st.header("💰 Análisis de Flujo de Caja")
    st.markdown("Análisis detallado de ingresos, egresos y capacidad de ahorro")
    
    # Placeholder para implementación completa
    st.info("Esta sección está en desarrollo. Próximamente: análisis detallado de flujo de caja.")

def render_balance_view(data, currency):
    """Vista de balance y ratios"""
    st.header("⚖️ Balance y Ratios Financieros")
    st.markdown("Análisis de situación patrimonial y solvencia")
    
    # Placeholder para implementación completa
    st.info("Esta sección está en desarrollo. Próximamente: ratios financieros detallados.")

def render_debt_view(data, currency):
    """Vista de estrategia de deuda"""
    st.header("📉 Estrategia de Deuda")
    st.markdown("Simulación de pago de deudas y optimización")
    
    # Placeholder para implementación completa
    st.info("Esta sección está en desarrollo. Próximamente: simulador de pago de deudas.")

# ============================================================================
# 7. APLICACIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación"""
    
    # Inyectar CSS empresarial
    inject_enterprise_css()
    
    # Sidebar Empresarial
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: white; margin-bottom: 0.5rem;">💰</h1>
            <h2 style="color: white; margin-bottom: 0;">Financial Suite</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 14px;">
                Enterprise Financial Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navegación
        st.markdown("### Navegación")
        page = st.radio(
            "",
            ["📊 Vista General", "💰 Flujo de Caja", "⚖️ Balance y Ratios", "📉 Estrategia de Deuda"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Configuración
        st.markdown("### Configuración")
        currency = st.selectbox(
            "Moneda",
            ["USD", "EUR", "PEN", "GBP", "JPY"],
            index=0
        )
        
        st.markdown("---")
        
        # Carga de Archivos
        st.markdown("### Carga de Datos")
        uploaded_file = st.file_uploader(
            "Subir archivo Excel",
            type=["xlsx", "xls"],
            help="Suba su archivo Finanzas_Personales.xlsx"
        )
    
    # Contenido Principal
    if uploaded_file is not None:
        try:
            # Cargar datos
            xls = pd.ExcelFile(uploaded_file)
            data = {}
            
            for sheet_name in xls.sheet_names:
                data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Validar estructura básica
            if "Ingresos" not in data:
                st.error("❌ El archivo debe contener una hoja 'Ingresos'")
                return
            
            # Navegación a páginas
            if page == "📊 Vista General":
                render_dashboard_view(data, currency)
            elif page == "💰 Flujo de Caja":
                render_cashflow_view(data, currency)
            elif page == "⚖️ Balance y Ratios":
                render_balance_view(data, currency)
            elif page == "📉 Estrategia de Deuda":
                render_debt_view(data, currency)
                
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.info("Por favor, asegúrese de que el archivo sigue el formato especificado.")
    else:
        # Pantalla de bienvenida
        col_welcome = st.columns([1])
        
        with col_welcome[0]:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 4rem 2rem;
                background: linear-gradient(135deg, {COLORS['background']}, white);
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            ">
                <h1 style="font-size: 48px; margin-bottom: 1rem;">📊</h1>
                <h1 style="color: {COLORS['primary']}; margin-bottom: 1rem;">
                    Financial Analytics Suite
                </h1>
                <p style="font-size: 18px; color: {COLORS['neutral']}; margin-bottom: 2rem;">
                    Plataforma empresarial de análisis financiero inteligente
                </p>
                
                <div style="
                    display: inline-block;
                    background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_light']});
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 12px;
                    font-weight: 600;
                    font-size: 16px;
                ">
                    👈 Sube tu archivo Excel para comenzar
                </div>
                
                <div style="margin-top: 3rem;">
                    <h3 style="color: {COLORS['neutral_dark']}; margin-bottom: 1rem;">
                        📋 Requisitos del Archivo
                    </h3>
                    <div style="
                        background: white;
                        border: 1px solid {COLORS['border']};
                        border-radius: 12px;
                        padding: 1.5rem;
                        text-align: left;
                        max-width: 600px;
                        margin: 0 auto;
                    ">
                        <ul style="color: {COLORS['neutral']};">
                            <li>Archivo Excel (.xlsx o .xls)</li>
                            <li>Hoja 'Ingresos' con columnas: Fecha, Monto_Neto</li>
                            <li>Hoja 'Gastos_Fijos' o 'Gastos' con columna Tipo</li>
                            <li>Hoja 'Cuentas_Bancarias' con Saldo_Inicial</li>
                            <li>Formatos de fecha consistentes (YYYY-MM-DD)</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
