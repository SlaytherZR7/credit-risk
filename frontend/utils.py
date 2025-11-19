"""
Utilidades y funciones auxiliares para la aplicación Streamlit.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import requests
from datetime import datetime, timedelta

def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Genera datos de muestra para testing.
    
    Args:
        n_samples: Número de muestras a generar
        
    Returns:
        DataFrame con datos sintéticos
    """
    np.random.seed(42)
    
    data = {
        'income': np.random.normal(50000, 15000, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'credit_amount': np.random.uniform(1000, 100000, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples)
    }
    
    # Asegurar valores positivos
    for key in ['income', 'credit_amount']:
        data[key] = np.abs(data[key])
    
    return pd.DataFrame(data)

def validate_profile_data(profile: Dict[str, Any]) -> List[str]:
    """
    Valida los datos del perfil crediticio.
    
    Args:
        profile: Diccionario con datos del perfil
        
    Returns:
        Lista de errores encontrados
    """
    errors = []
    
    # Validar income
    if profile.get('income', 0) <= 0:
        errors.append("Los ingresos deben ser mayores a cero")
    
    # Validar age
    age = profile.get('age', 0)
    if age < 18 or age > 100:
        errors.append("La edad debe estar entre 18 y 100 años")
    
    # Validar credit_amount
    if profile.get('credit_amount', 0) <= 0:
        errors.append("El monto de crédito debe ser mayor a cero")
    
    # Validar employment_length
    if profile.get('employment_length', -1) < 0:
        errors.append("Los años de empleo no pueden ser negativos")
    
    # Validar debt_ratio
    debt_ratio = profile.get('debt_ratio', -1)
    if debt_ratio < 0 or debt_ratio > 1:
        errors.append("El ratio de deuda debe estar entre 0 y 1")
    
    return errors

def format_currency(amount: float) -> str:
    """Formatea una cantidad como moneda."""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Formatea un valor como porcentaje."""
    return f"{value:.2%}"

def create_risk_gauge(risk_score: float, title: str = "Risk Score") -> go.Figure:
    """
    Crea un gauge chart para mostrar el risk score.
    
    Args:
        risk_score: Score de riesgo (0-1)
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    # Determinar color basado en el score
    if risk_score < 0.3:
        color = "#4caf50"  # Verde
    elif risk_score < 0.7:
        color = "#ff9800"  # Naranja
    else:
        color = "#f44336"  # Rojo
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#e8f5e8"},
                {'range': [30, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_distribution_chart(data: List[Dict[str, Any]], 
                            column: str, 
                            title: str) -> go.Figure:
    """
    Crea un gráfico de distribución.
    
    Args:
        data: Lista de diccionarios con los datos
        column: Columna a graficar
        title: Título del gráfico
        
    Returns:
        Figura de Plotly
    """
    values = [item[column] for item in data if column in item]
    
    fig = px.histogram(
        x=values,
        title=title,
        labels={'x': column.replace('_', ' ').title(), 'y': 'Frecuencia'},
        nbins=20
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_comparison_chart(data: pd.DataFrame) -> go.Figure:
    """
    Crea un gráfico de comparación entre variables.
    
    Args:
        data: DataFrame con los datos
        
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    # Scatter plot de income vs credit_amount coloreado por risk_level
    if all(col in data.columns for col in ['income', 'credit_amount', 'risk_level']):
        color_map = {'BAJO': '#4caf50', 'MEDIO': '#ff9800', 'ALTO': '#f44336'}
        
        for risk_level in data['risk_level'].unique():
            mask = data['risk_level'] == risk_level
            fig.add_trace(go.Scatter(
                x=data.loc[mask, 'income'],
                y=data.loc[mask, 'credit_amount'],
                mode='markers',
                name=f'Riesgo {risk_level}',
                marker=dict(
                    color=color_map.get(risk_level, '#999999'),
                    size=8,
                    opacity=0.7
                )
            ))
    
    fig.update_layout(
        title='Relación entre Ingresos y Monto de Crédito por Nivel de Riesgo',
        xaxis_title='Ingresos Anuales ($)',
        yaxis_title='Monto de Crédito ($)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def display_api_status():
    """Muestra el estado de conexión con la API."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("🟢 API conectada")
            return True
        else:
            st.error("🔴 API no responde correctamente")
            return False
    except:
        st.error("🔴 No se puede conectar con la API")
        st.info("Asegúrate de que FastAPI esté ejecutándose en http://localhost:8000")
        return False

def cache_predictions(predictions: List[Dict[str, Any]], key: str = "predictions"):
    """
    Cachea las predicciones en la sesión de Streamlit.
    
    Args:
        predictions: Lista de predicciones
        key: Clave para almacenar en session_state
    """
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    
    st.session_state.cached_data[key] = {
        'data': predictions,
        'timestamp': datetime.now()
    }

def get_cached_predictions(key: str = "predictions", 
                         max_age_minutes: int = 30) -> List[Dict[str, Any]]:
    """
    Obtiene predicciones cacheadas.
    
    Args:
        key: Clave de los datos cacheados
        max_age_minutes: Edad máxima del cache en minutos
        
    Returns:
        Lista de predicciones o None si no hay cache válido
    """
    if 'cached_data' not in st.session_state:
        return None
    
    cached = st.session_state.cached_data.get(key)
    if not cached:
        return None
    
    # Verificar edad del cache
    age = datetime.now() - cached['timestamp']
    if age.total_seconds() > max_age_minutes * 60:
        return None
    
    return cached['data']

def export_results_to_csv(data: pd.DataFrame, filename: str = None) -> str:
    """
    Exporta resultados a CSV.
    
    Args:
        data: DataFrame con los resultados
        filename: Nombre del archivo (opcional)
        
    Returns:
        String con el CSV
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"credit_risk_results_{timestamp}.csv"
    
    return data.to_csv(index=False)

def create_summary_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crea métricas de resumen de las predicciones.
    
    Args:
        predictions: Lista de predicciones
        
    Returns:
        Diccionario con métricas de resumen
    """
    if not predictions:
        return {}
    
    risk_scores = [p['risk_score'] for p in predictions]
    risk_levels = [p['risk_level'] for p in predictions]
    recommendations = [p['recommendation'] for p in predictions]
    
    return {
        'total_predictions': len(predictions),
        'average_risk_score': np.mean(risk_scores),
        'median_risk_score': np.median(risk_scores),
        'max_risk_score': np.max(risk_scores),
        'min_risk_score': np.min(risk_scores),
        'risk_distribution': pd.Series(risk_levels).value_counts().to_dict(),
        'recommendation_distribution': pd.Series(recommendations).value_counts().to_dict(),
        'approval_rate': (pd.Series(recommendations) == 'APROBAR').mean(),
        'rejection_rate': (pd.Series(recommendations) == 'RECHAZAR').mean()
    }

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_sample_dataset() -> pd.DataFrame:
    """Carga un dataset de muestra (con cache)."""
    return create_sample_data(1000)

def display_error_message(error_type: str, details: str = ""):
    """
    Muestra mensajes de error formateados.
    
    Args:
        error_type: Tipo de error
        details: Detalles adicionales
    """
    error_messages = {
        'api_connection': "🔌 Error de Conexión: No se puede conectar con la API",
        'validation': "⚠️ Error de Validación: Los datos ingresados no son válidos",
        'processing': "⚙️ Error de Procesamiento: Error al procesar la solicitud",
        'file_upload': "📁 Error de Archivo: Problema al cargar el archivo"
    }
    
    message = error_messages.get(error_type, "❌ Error Desconocido")
    
    if details:
        message += f"\n\nDetalles: {details}"
    
    st.error(message)

def create_feature_importance_chart(feature_names: List[str], 
                                  importances: List[float]) -> go.Figure:
    """
    Crea un gráfico de importancia de features.
    
    Args:
        feature_names: Nombres de las features
        importances: Valores de importancia
        
    Returns:
        Figura de Plotly
    """
    # Ordenar por importancia
    sorted_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_data)
    
    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(names),
        orientation='h',
        marker=dict(color='skyblue')
    ))
    
    fig.update_layout(
        title='Importancia de Features',
        xaxis_title='Importancia',
        yaxis_title='Features',
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig