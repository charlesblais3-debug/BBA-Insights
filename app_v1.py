
texts = {
    "es": {
        "title": "Generador de Gráficos Operacionales",
        "upload": "Seleccione archivo CSV o Excel",
        "sheet": "Seleccione hoja del Excel",
        "header_rows": "Ingrese filas para encabezado (ej: 0,1,2)",
        "preview": "Vista previa de datos:",
        "delete_rows": "Ingrese índices de filas a eliminar (separados por coma)",
        "after_delete": "Datos después de eliminar filas:",
        "date_col": "Seleccione columna de fecha",
        "chart_type": "Seleccione tipo de gráfico",
        "scatter": "Scatter",
        "normal_dist": "Distribución Normal",
        "scatter_normal": "Scatter + Distribución Normal",
        "density_scatter": "Densidad + Scatter",
        "cycle_detection": "Detección de Ciclos",
        "start_date": "Fecha inicio",
        "end_date": "Fecha fin",
        "custom_name": "Nombre personalizado para {col} (opcional)",
        "ref_value": "Valor de referencia para {col} (ej: '20 kW' o dejar vacío)",
        "scatter_options": "### Opciones de visualización en Scatter",
        "show_avg": "Mostrar Promedio Diario",
        "show_p70": "Mostrar Percentil 70 Diario",
        "show_p80": "Mostrar Percentil 80 Diario",
        "show_p90": "Mostrar Percentil 90 Diario",
        "min_val": "Valor mínimo para {col}",
        "max_val": "Valor máximo para {col}",
        "color_select": "Seleccione color del gráfico",
        "line_style": "Estilo de línea",
        "ref_line_color": "Color de la línea de referencia",
        "generate": "Generar Gráficos",
        "no_data": "No hay datos después del filtrado.",
        "cycles_table": "Tabla de ciclos detectados:",
        "excel_generated": "Archivo Excel generado."
    },
    "en": {
        "title": "Operational Charts Generator",
        "upload": "Upload CSV or Excel file",
        "sheet": "Select Excel sheet",
        "header_rows": "Enter rows for header (e.g., 0,1,2)",
        "preview": "Data preview:",
        "delete_rows": "Enter row indices to delete (comma separated)",
        "after_delete": "Data after deleting rows:",
        "date_col": "Select date column",
        "chart_type": "Select chart type",
        "scatter": "Scatter",
        "normal_dist": "Normal Distribution",
        "scatter_normal": "Scatter + Normal Distribution",
        "density_scatter": "Density + Scatter",
        "cycle_detection": "Cycle Detection",
        "start_date": "Start date",
        "end_date": "End date",
        "custom_name": "Custom name for {col} (optional)",
        "ref_value": "Reference value for {col} (e.g., '20 kW' or leave blank)",
        "scatter_options": "### Scatter Visualization Options",
        "show_avg": "Show Daily Average",
        "show_p70": "Show Daily Percentile 70",
        "show_p80": "Show Daily Percentile 80",
        "show_p90": "Show Daily Percentile 90",
        "min_val": "Minimum value for {col}",
        "max_val": "Maximum value for {col}",
        "color_select": "Select chart color",
        "line_style": "Line style",
        "ref_line_color": "Reference line color",
        "generate": "Generate Charts",
        "no_data": "No data after filtering.",
        "cycles_table": "Detected cycles table:",
        "excel_generated": "Excel file generated."
    }
}

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

import seaborn as sns
from scipy import stats
import os

# Selector de idioma
lang = st.selectbox('Select language / Seleccione idioma', ['English', 'Español'])
lang_code = 'en' if lang == 'English' else 'es'



# ==============================
# Función: Cargar datos
# ==============================
colores = {"Azul Oscuro": "#002952",
           "Azul Medio": "#193E63",
           "Azul Claro": "#335475",
           "Verde Oscuro": "#2D6E57",
           "Turquesa": "#14ACBA",
           "Celeste Claro": "#D0EEF1",
           "Celeste Medio": "#89D5DD",
           "Celeste Intenso": "#5BC5CF",
           "Dorado": "#CB9B5D",
           "Gris Azulado": "#4C6986"
       }

fecha_inicio, fecha_fin = None, None
columnas, filtros = [], {}

       
def cargar_datos(archivo, hoja=None, header_row=None):
    ext = os.path.splitext(archivo.name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(archivo, header=header_row)
    elif ext in [".xls", ".xlsx"]:
        xls = pd.ExcelFile(archivo)
        if hoja:
            df = pd.read_excel(xls, sheet_name=hoja, header=header_row, engine='openpyxl')
        else:
            df = pd.read_excel(xls, header=header_row, engine='openpyxl')
    else:
        st.error("Formato no soportado. Use CSV o Excel.")
        return None
    return df

# ==============================
# Función: Filtrar datos
# ==============================

def filtrar_datos(df, fecha_col, fecha_inicio, fecha_fin, filtros):
    # Convertir columna de fecha
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')

    # Filtrar por rango de fechas
    df = df[(df[fecha_col] >= fecha_inicio) & (df[fecha_col] <= fecha_fin)]

    # Convertir columnas numéricas y aplicar filtros
    for col, (min_val, max_val) in filtros.items():
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Forzar numérico
        df = df.dropna(subset=[col])  # Eliminar filas con NaN
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]

    return df

# ==============================
# Función: Calcular estadísticas
# ==============================
def calcular_estadisticas(serie):
    return {
        "Máximo": serie.max(),
        "Mínimo": serie.min(),
        "Media": serie.mean(),
        "Desv.Std": serie.std(),
        "Percentil 80": np.percentile(serie, 80),
        "Cantidad": len(serie)
    }

def parse_valor_referencia(valor):
    try:
        partes = valor.split(" ", 1)
        numero = float(partes[0])
        unidades = partes[1] if len(partes) > 1 else ""
        return numero, unidades
    except:
        return None, ""

# ==============================
# Función: Graficar
# ==============================
def graficar(df, fecha_col, columnas, color, color_linea, estilo_linea,
             nombres_personalizados, valores_referencia,
             mostrar_promedio, mostrar_p70, mostrar_p80, mostrar_p90):
    for col in columnas:
        serie = df[col].dropna()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df[fecha_col], serie, color=color, alpha=0.7)
        ax.set_title(f"Scatter de {nombres_personalizados[col]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(nombres_personalizados[col])

        # Línea de referencia específica para esta columna
        if valores_referencia[col]:
            numero, unidades = parse_valor_referencia(valores_referencia[col])
            if numero is not None:
                ax.axhline(y=numero, color=color_linea, linestyle=estilo_linea, linewidth=2,
                           label=f"Referencia: {numero} {unidades}")

        # Curvas adicionales
        df_resample = df.set_index(fecha_col).resample('D')
        if mostrar_promedio:
            promedio_diario = df_resample[col].mean()
            ax.plot(promedio_diario.index, promedio_diario.values, color='orange', linewidth=2, label="Promedio Diario")
        if mostrar_p70:
            p70_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 70))
            ax.plot(p70_diario.index, p70_diario.values, color='green', linewidth=2, linestyle='--', label="Percentil 70")
        if mostrar_p80:
            p80_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 80))
            ax.plot(p80_diario.index, p80_diario.values, color='blue', linewidth=2, linestyle='--', label="Percentil 80")
        if mostrar_p90:
            p90_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 90))
            ax.plot(p90_diario.index, p90_diario.values, color='red', linewidth=2, linestyle='--', label="Percentil 90")

        # Estadísticas
        stats_dict = calcular_estadisticas(serie)
        stats_text = "\n".join([
            f"Máx: {stats_dict['Máximo']:.2f}",
            f"Mín: {stats_dict['Mínimo']:.2f}",
            f"Media: {stats_dict['Media']:.2f}",
            f"Desv.Std: {stats_dict['Desv.Std']:.2f}",
            f"P80: {stats_dict['Percentil 80']:.2f}",
            f"n: {stats_dict['Cantidad']}"
        ])
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5))
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        fig.savefig(f"{nombres_personalizados[col]}_scatter.png", dpi=300, bbox_inches='tight')

def graficar_distribucion_normal(df, columnas, color, valores_referencia_referencia, color_linea, estilo_linea, nombres_personalizados):
    for col in columnas:
        serie = pd.to_numeric(df[col], errors='coerce').dropna()
        if serie.empty:
            st.warning(f"No hay datos válidos para {col}.")
            continue

        media = serie.mean()
        std = serie.std()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(serie, bins=30, density=True, color=color, alpha=0.6, label="Histograma")
        x = np.linspace(serie.min(), serie.max(), 1000)
        p = stats.norm.pdf(x, media, std)
        ax.plot(x, p, 'k', linewidth=2, label="Distribución Normal")

        if valores_referencia[col]:
            numero, unidades = parse_valor_referencia(valores_referencia[col])
            if numero is not None:
                ax.axvline(x=numero, color=color_linea, linestyle=estilo_linea, linewidth=2,
                           label=f"Referencia: {numero} {unidades}")

        ax.set_title(f"Distribución Normal de {nombres_personalizados[col]}", fontsize=14, fontweight="bold")
        ax.set_xlabel(nombres_personalizados[col])
        ax.set_ylabel("Densidad")

        stats_text = f"Media: {media:.2f}\nDesv.Std: {std:.2f}\nn: {len(serie)}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5))

        ax.legend()
        st.pyplot(fig)
        fig.savefig(f"{nombres_personalizados[col]}_distribucion_normal.png", dpi=300, bbox_inches='tight')

def graficar_ambos(df, fecha_col, columnas, color, color_linea, estilo_linea,
                   nombres_personalizados, valores_referencia,
                   mostrar_promedio, mostrar_p70, mostrar_p80, mostrar_p90):
    for col in columnas:
        serie = pd.to_numeric(df[col], errors='coerce').dropna()
        if serie.empty:
            st.warning(f"No hay datos válidos para {col}.")
            continue

        media = serie.mean()
        std = serie.std()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [2.5, 1]})

        # Scatter
        ax1.scatter(df[fecha_col], serie, color=color, alpha=0.7)
        ax1.set_title(f"{nombres_personalizados[col]}", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Date")
        ax1.set_ylabel(nombres_personalizados[col])

        if valores_referencia[col]:
            numero, unidades = parse_valor_referencia(valores_referencia[col])
            if numero is not None:
                ax1.axhline(y=numero, color=color_linea, linestyle=estilo_linea, linewidth=2,
                            label=f"Referencia: {numero} {unidades}")

        df_resample = df.set_index(fecha_col).resample('D')
        if mostrar_promedio:
            promedio_diario = df_resample[col].mean()
            ax1.plot(promedio_diario.index, promedio_diario.values, color='#002952', linewidth=2, label="Daily Average")
        if mostrar_p70:
            p70_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 70))
            ax1.plot(p70_diario.index, p70_diario.values, color='#2D6E57', linewidth=2, linestyle='--', label="Percentile 70th")
        if mostrar_p80:
            p80_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 80))
            ax1.plot(p80_diario.index, p80_diario.values, color='#89D5DD', linewidth=2, linestyle='--', label="Percentile 80th")
        if mostrar_p90:
            p90_diario = df_resample[col].apply(lambda x: np.nanpercentile(x, 90))
            ax1.plot(p90_diario.index, p90_diario.values, color='#CB9B5D', linewidth=2, linestyle='--', label="Percentile 90th")

        stats_dict = calcular_estadisticas(serie)
        stats_text = "\n".join([
            f"Max.: {stats_dict['Máximo']:.2f}",
            f"Min.: {stats_dict['Mínimo']:.2f}",
            f"Average: {stats_dict['Media']:.2f}",
            f"Std.Dev.: {stats_dict['Desv.Std']:.2f}",
            f"Perc. 80: {stats_dict['Percentil 80']:.2f}",
            f"n: {stats_dict['Cantidad']}"
        ])
        ax1.text(0.98, 0.98, stats_text,
                transform=ax1.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5))

        ax1.grid(True)
        ax1.legend()

        # Distribución
        ax2.hist(serie, bins=30, density=True, color=color, alpha=0.6, label="Histogram")
        x = np.linspace(serie.min(), serie.max(), 1000)
        p = stats.norm.pdf(x, media, std)
        ax2.plot(x, p, 'k', linewidth=2, label="Normal Distribution")

        if valores_referencia[col]:
            numero, unidades = parse_valor_referencia(valores_referencia[col])
            if numero is not None:
                ax2.axvline(x=numero, color=color_linea, linestyle=estilo_linea, linewidth=2,
                            label=f"Reference: {numero} {unidades}")

        ax2.set_title(nombres_personalizados[col], fontsize=14, fontweight="bold")
        ax2.set_xlabel(nombres_personalizados[col])
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper right")

        st.pyplot(fig)
        fig.savefig(f"{nombres_personalizados[col]}_ambos.png", dpi=300, bbox_inches='tight')


def graficar_densidad_scatter(df, fecha_col, eje_x, eje_y, eje_z, rangos_z, bw_adjust, nombre_x, nombre_y, filtro_x, filtro_y, fecha_inicio=None, fecha_fin=None):
    # Convertir columnas a numéricas
    df[eje_x] = pd.to_numeric(df[eje_x], errors='coerce')
    df[eje_y] = pd.to_numeric(df[eje_y], errors='coerce')
    if eje_z:
        df[eje_z] = pd.to_numeric(df[eje_z], errors='coerce')

    # Convertir columna de fecha y aplicar filtro de fechas
    if fecha_col in df.columns and fecha_inicio is not None and fecha_fin is not None:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
        df = df[(df[fecha_col] >= fecha_inicio) & (df[fecha_col] <= fecha_fin)]

    # Eliminar NaN
    df.dropna(subset=[eje_x, eje_y] + ([eje_z] if eje_z else []), inplace=True)

    # Aplicar filtros en Eje X y Eje Y
    df = df[(df[eje_x] >= filtro_x[0]) & (df[eje_x] <= filtro_x[1])]
    df = df[(df[eje_y] >= filtro_y[0]) & (df[eje_y] <= filtro_y[1])]

    # Layout dinámico
    cantidad = len(rangos_z)
    if cantidad == 1:
        filas, cols = 1, 1
    elif cantidad == 2:
        filas, cols = 1, 2
    elif cantidad == 4:
        filas, cols = 2, 2
    else:
        filas = (cantidad + 2) // 3
        cols = 3

    fig, axs = plt.subplots(filas, cols, figsize=(6 * cols, 5 * filas), sharex=True, sharey=True)
    axs = np.array(axs).flatten()

    for i, rango in enumerate(rangos_z):
        ax = axs[i]
        if eje_z and rango[0] is not None:
            df_filtrado = df[(df[eje_z] >= rango[0]) & (df[eje_z] < rango[1])]
        else:
            df_filtrado = df.copy()

        # Gráfico KDE con sensibilidad ajustable
        sns.kdeplot(x=df_filtrado[eje_x], y=df_filtrado[eje_y], ax=ax, cmap="viridis", fill=True, bw_adjust=bw_adjust)

        # Scatter superpuesto
        ax.scatter(df_filtrado[eje_x], df_filtrado[eje_y], edgecolor='black', color='orange', s=30, alpha=0.7, label='Datos')

        # Cuadro de estadísticas compacto

# Tablas estadísticas más estéticas
        if not df_filtrado.empty:
            
            serie_x = df_filtrado[eje_x]
            serie_y = df_filtrado[eje_y]

            
            stats_x = (f"**Statistics \n{nombre_x}**\n"
                       f"  - Media: {serie_x.mean():.1f}\n"
                       f"  - Mediana: {serie_x.median():.1f}\n"
                       f"  - Máx: {serie_x.max():.1f}\n"
                       f"  - Mín: {serie_x.min():.1f}\n"
                       f"  - Desv. Est.: {serie_x.std():.1f}")
        
            stats_y = (f"**Statistics \n{nombre_y}**\n"
                       f"  - Media: {serie_y.mean():.1f}\n"
                       f"  - Mediana: {serie_y.median():.1f}\n"
                       f"  - Máx: {serie_y.max():.1f}\n"
                       f"  - Mín: {serie_y.min():.1f}\n"
                       f"  - Desv. Est.: {serie_y.std():.1f}")
        
            ax.text(0.05, 0.3, stats_x, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
            ax.text(0.95, 0.3, stats_y, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))


        # Ajustar leyenda dentro del gráfico
        ax.legend(loc='upper left', fontsize=7, frameon=True, bbox_to_anchor=(0.02, 0.98))

        # Configuración de ejes
        ax.set_title(f"Rango {rango[0]} - {rango[1]}", fontsize=12, fontweight="bold")
        ax.set_xlabel(nombre_x)
        ax.set_ylabel(nombre_y)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    fig.savefig("densidad_scatter.png", dpi=300, bbox_inches='tight')


def detectar_ciclos(df, fecha_col, variable_col, fecha_inicio, fecha_fin, umbral, nombre_variable):
    """
    Detecta ciclos en una serie temporal según un umbral, genera tabla resumen y gráficos.
    
    Parámetros:
    - df: DataFrame con datos.
    - fecha_col: Nombre de la columna de fecha.
    - variable_col: Columna a analizar (ej. temperatura, presión).
    - fecha_inicio, fecha_fin: Rango de fechas para filtrar.
    - umbral: Valor límite para considerar inicio-fin del ciclo.
    
    Salida:
    - DataFrame con ciclos detectados (inicio, fin, duración en minutos).
    - Archivo Excel con la tabla.
    - Gráfico combinado: perfil con ciclos + distribución normal de duraciones.
    """

    # Convertir fecha y ordenar
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    df = df.set_index(fecha_col).sort_index()

    # Filtrar por rango de fechas
    df_filtrado = df[(df.index >= fecha_inicio) & (df.index <= fecha_fin)].copy()

    if df_filtrado.empty:
        st.warning(f"No hay datos en el rango {fecha_inicio} a {fecha_fin}.")
        return pd.DataFrame()

    # Identificar ciclos
    df_filtrado['above_threshold'] = df_filtrado[variable_col] > umbral
    df_filtrado['block_id'] = (df_filtrado['above_threshold'] != df_filtrado['above_threshold'].shift()).cumsum()

    ciclos = []
    for block_id, block_df in df_filtrado.groupby('block_id'):
        if block_df['above_threshold'].iloc[0]:
            inicio = block_df.index.min()
            fin = block_df.index.max()
            duracion = (fin - inicio).total_seconds() / 60  # minutos
            if duracion > 0:
                ciclos.append({
                    'Fecha_Inicio_Ciclo': inicio,
                    'Fecha_Fin_Ciclo': fin,
                    'Duracion_Ciclo_min': duracion
                })

    df_ciclos = pd.DataFrame(ciclos)

    # Guardar tabla en Excel
    if not df_ciclos.empty:
        archivo_excel = f"Perfil_Ciclos_{fecha_inicio.date()}_{fecha_fin.date()}.xlsx"
        df_ciclos.to_excel(archivo_excel, index=False)
        st.success(f"Archivo Excel generado: {archivo_excel}")

    # Gráfico combinado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [2.5, 1]})

    # Perfil con ciclos
    ax1.plot(df_filtrado.index, df_filtrado[variable_col], label='Perfil', color='blue', marker='.', linestyle='-')
    labeled = False
    for _, row in df_ciclos.iterrows():
        inicio = row['Fecha_Inicio_Ciclo']
        fin = row['Fecha_Fin_Ciclo']
        ax1.axvspan(inicio, fin, color='green', alpha=0.1, label='Ciclo Detectado' if not labeled else "")
        ax1.plot(inicio, df_filtrado.loc[inicio, variable_col], 'go', markersize=8, label='Inicio Ciclo' if not labeled else "")
        ax1.plot(fin, df_filtrado.loc[fin, variable_col], 'ro', markersize=8, label='Fin Ciclo' if not labeled else "")
        labeled = True

    ax1.set_title(f"Perfil con Ciclos Detectados ({fecha_inicio.date()} a {fecha_fin.date()})")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(nombre_variable)
    ax1.grid(True)
    ax1.legend()

    # Distribución de duraciones
    if not df_ciclos.empty:
        duraciones = df_ciclos['Duracion_Ciclo_min']
        media = duraciones.mean()
        std = duraciones.std()
        x = np.linspace(duraciones.min(), duraciones.max(), 1000)
        p = stats.norm.pdf(x, media, std)

        ax2.hist(duraciones, bins=30, density=True, color='orange', alpha=0.6, label="Histograma")
        ax2.plot(x, p, 'k', linewidth=2, label="Normal Distribution")
        ax2.set_title("Distribución de Duración de Ciclos")
        ax2.set_xlabel("Duración (minutos)")
        ax2.set_ylabel("Densidad")
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    fig.savefig(f"Perfil_Ciclos_{fecha_inicio.date()}_{fecha_fin.date()}.png", dpi=300, bbox_inches='tight')

    return df_ciclos


def xgboost_analysis(df, fecha_col, fecha_inicio, fecha_fin, target_col, feature_cols):
    """
    Entrena y analiza un modelo XGBoost sobre datos filtrados por rango de fechas.
    Incluye métricas, gráficos y análisis en 8 tabs (sin SHAP).
    """

    # 1. Filtrar por fecha
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    df = df[(df[fecha_col] >= fecha_inicio) & (df[fecha_col] <= fecha_fin)]
    if df.empty:
        st.warning("No hay datos en el rango seleccionado.")
        return

    # 2. Convertir columnas a numéricas
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[target_col] + feature_cols)
    if df.empty:
        st.warning("No hay datos válidos después de la limpieza.")
        return

    # 3. Preparar datos
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Tamaño del dataset
    n_rows = len(df)
    
    if n_rows < 1000:
        st.info("Dataset pequeño detectado: recomendamos max_depth=3 y n_estimators≈100 para evitar overfitting.")
    elif n_rows > 5000:
        st.info("Dataset grande: puedes usar más árboles y menor learning_rate para mejor generalización.")

    # Ajuste dinámico
    if n_rows < 1000:
        n_estimators_default = 100
        max_depth_default = 3
    elif n_rows < 5000:
        n_estimators_default = 300
        max_depth_default = 5
    else:
        n_estimators_default = 500
        max_depth_default = 6
    
    learning_rate_default = 0.05 if n_rows > 1000 else 0.1


    st.sidebar.subheader("Hyperparameter Settings (Adaptive Defaults)")
    n_estimators = st.sidebar.slider("n_estimators", 50, 1000, n_estimators_default)
    max_depth = st.sidebar.slider("max_depth", 2, 10, max_depth_default)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, learning_rate_default)
    reg_alpha = st.sidebar.slider("reg_alpha", 0.0, 10.0, 0.0)
    reg_lambda = st.sidebar.slider("reg_lambda", 0.0, 10.0, 1.0)

    # 4. Entrenar modelo
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)

    # 5. Predicciones y métricas
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Model Performance")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**Root Mean Squared Error:** {rmse:.2f}")

    # 6. Tabs para análisis (sin SHAP)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Feature Importance", "Predictions", "PDP", "Correlation", "Learning Curve",
        "Error Distribution", "Pairplot", "What-If"
    ])

    # --- Tab 1: Feature Importance ---
    with tab1:
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    # --- Tab 2: Predictions ---
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    # --- Tab 3: PDP (Partial Dependence Plot) ---
    with tab3:
        st.write("Partial Dependence Plot (PDP)")
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(model, X_train, feature_cols, ax=ax)
        st.pyplot(fig)

    # --- Tab 4: Correlation Heatmap ---
    with tab4:
        corr = df[feature_cols + [target_col]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # --- Tab 5: Learning Curve ---
    with tab5:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2'
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
        ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
        ax.set_title('Learning Curve')
        ax.legend()
        st.pyplot(fig)

    # --- Tab 6: Error Distribution ---
    with tab6:
        errors = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(errors, bins=30, kde=True, color='orange', ax=ax)
        ax.set_title("Error Distribution")
        ax.set_xlabel("Prediction Error")
        st.pyplot(fig)

    # --- Tab 7: Pairplot ---
    with tab7:
        st.write("Pairplot of Features and Target")
        sample_df = df.sample(min(500, len(df)))  # limitar tamaño para performance
        fig = sns.pairplot(sample_df[feature_cols + [target_col]], diag_kind='kde')
        st.pyplot(fig)

    # --- Tab 8: What-If Analysis ---
    with tab8:
        st.write("Simulación What-If")
        st.write("Modifica valores de las features para ver impacto en la predicción:")
        input_data = {}
        for col in feature_cols:
            val = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.success(f"Predicción estimada: **{pred:.2f}**")


# ==============================
# Interfaz Streamlit
# ==============================
st.title(texts[lang_code]['title'])

archivo = st.file_uploader(texts[lang_code]['upload'], type=["csv", "xls", "xlsx"])

if archivo:
    # --- Procesamiento inicial del archivo ---
    ext = os.path.splitext(archivo.name)[1].lower()
    hoja = None
    if ext in [".xls", ".xlsx"]:
        xls = pd.ExcelFile(archivo)
        hoja = st.selectbox("Seleccione hoja del Excel", xls.sheet_names)
        df = pd.read_excel(archivo, sheet_name=hoja, header=None)
    else:
        df = pd.read_csv(archivo, header=None)

    # Encabezados dinámicos
    filas_encabezado = st.text_input("Ingrese filas para encabezado (ej: 0,1,2)")
    if filas_encabezado:
        indices_header = [int(i) for i in filas_encabezado.split(",") if i.strip().isdigit()]
        encabezado = df.iloc[indices_header].astype(str).agg("_".join)
        encabezado = encabezado.str.replace("nan", "", regex=False)
        encabezado = [col if col.strip() != "" else f"Columna_{i}" for i, col in enumerate(encabezado)]

        # Asegurar nombres únicos
        from collections import Counter
        contador = Counter()
        columnas_unicas = []
        for col in encabezado:
            contador[col] += 1
            columnas_unicas.append(f"{col}_{contador[col]}" if contador[col] > 1 else col)

        df.columns = columnas_unicas
        df = df.drop(indices_header)

    st.write("Vista previa de datos:", df.head())

    # Eliminar filas antes de análisis
    filas_a_eliminar = st.text_input("Ingrese índices de filas a eliminar (separados por coma)")
    if filas_a_eliminar:
        indices = [int(i) for i in filas_a_eliminar.split(",") if i.strip().isdigit()]
        df = df.drop(indices)
        st.write("Datos después de eliminar filas:", df.head())

    # Selección de columna de fecha
    fecha_col = st.selectbox("Seleccione columna de fecha", df.columns)

    # --- Selector de tipo de gráfico ---
    tipo_grafico = st.radio("Seleccione tipo de gráfico", ["Scatter", "Distribución Normal", "Scatter + Distribución Normal", "Densidad + Scatter", "Detección de Ciclos", "XGBoost Analysis"])


    # --- Opciones condicionales ---
    if tipo_grafico in ["Scatter", "Scatter + Distribución Normal"]:
        columnas = st.multiselect("Seleccione columnas para graficar", [c for c in df.columns if c != fecha_col])
        fecha_inicio = st.date_input("Fecha inicio")
        fecha_fin = st.date_input("Fecha fin")

        # Nombres personalizados
        nombres_personalizados = {}
        for col in columnas:
            nuevo_nombre = st.text_input(f"Nombre personalizado para {col} (opcional)")
            nombres_personalizados[col] = nuevo_nombre if nuevo_nombre.strip() != "" else col

        # Valores de referencia
        valores_referencia = {}
        for col in columnas:
            ref_input = st.text_input(f"Valor de referencia para {col} (ej: '20 kW' o dejar vacío)")
            valores_referencia[col] = ref_input.strip() if ref_input.strip() != "" else None

        # Opciones de visualización en Scatter
        st.markdown("### Opciones de visualización en Scatter")
        mostrar_promedio = st.checkbox("Mostrar Promedio Diario")
        mostrar_p70 = st.checkbox("Mostrar Percentil 70 Diario")
        mostrar_p80 = st.checkbox("Mostrar Percentil 80 Diario")
        mostrar_p90 = st.checkbox("Mostrar Percentil 90 Diario")

        # Filtros
        filtros = {}
        for col in columnas:
            min_val = st.number_input(f"Valor mínimo para {col}", value=float(df[col].min()))
            max_val = st.number_input(f"Valor máximo para {col}", value=float(df[col].max()))
            filtros[col] = (min_val, max_val)

        # Colores y estilos
        color_nombre = st.selectbox("Seleccione color del gráfico", list(colores.keys()))
        color = colores[color_nombre]
        estilo_linea = st.selectbox("Estilo de línea", ["-", "--", ":"])
        color_linea_nombre = st.selectbox("Color de la línea de referencia", list(colores.keys()))
        color_linea = colores[color_linea_nombre]

    elif tipo_grafico == "Distribución Normal":
        columnas = st.multiselect("Seleccione columnas para graficar", [c for c in df.columns if c != fecha_col])
        fecha_inicio = st.date_input("Fecha inicio")
        fecha_fin = st.date_input("Fecha fin")

        nombres_personalizados = {}
        for col in columnas:
            nuevo_nombre = st.text_input(f"Nombre personalizado para {col} (opcional)")
            nombres_personalizados[col] = nuevo_nombre if nuevo_nombre.strip() != "" else col

        valores_referencia = {}
        for col in columnas:
            ref_input = st.text_input(f"Valor de referencia para {col} (ej: '20 kW' o dejar vacío)")
            valores_referencia[col] = ref_input.strip() if ref_input.strip() != "" else None
        filtros = {}
        for col in columnas:
            min_val = st.number_input(f"Valor mínimo para {col}", value=float(df[col].min()))
            max_val = st.number_input(f"Valor máximo para {col}", value=float(df[col].max()))
            filtros[col] = (min_val, max_val)
        # Colores y estilos
        color_nombre = st.selectbox("Seleccione color del gráfico", list(colores.keys()))
        color = colores[color_nombre]
        estilo_linea = st.selectbox("Estilo de línea", ["-", "--", ":"])
        color_linea_nombre = st.selectbox("Color de la línea de referencia", list(colores.keys()))
        color_linea = colores[color_linea_nombre]

    elif tipo_grafico == "Densidad + Scatter":
        st.subheader("Configuración del gráfico Densidad + Scatter")
        fecha_inicio = st.date_input("Fecha inicio")
        fecha_fin = st.date_input("Fecha fin")
        

        eje_x = st.selectbox("Seleccione Eje X", df.columns)
        eje_y = st.selectbox("Seleccione Eje Y", df.columns)
        bw_adjust = st.slider("Sensibilidad del KDE (bw_adjust)", 0.1, 3.0, 1.0, 0.1)
        nombre_x = st.text_input("Nombre personalizado para Eje X", value=eje_x)
        nombre_y = st.text_input("Nombre personalizado para Eje Y", value=eje_y)
        
        # Filtros para Eje X y Eje Y
        filtro_x_min = st.number_input(f"Límite inferior para {eje_x}", value=float(df[eje_x].min()))
        filtro_x_max = st.number_input(f"Límite superior para {eje_x}", value=float(df[eje_x].max()))
        filtro_y_min = st.number_input(f"Límite inferior para {eje_y}", value=float(df[eje_y].min()))
        filtro_y_max = st.number_input(f"Límite superior para {eje_y}", value=float(df[eje_y].max()))
        usar_eje_z = st.checkbox("¿Desea filtrar por una tercera variable (Eje Z)?")
        rangos_z = []
        eje_z = None
        if usar_eje_z:
            eje_z = st.selectbox("Seleccione Eje Z", df.columns)
            cantidad_graficos = st.slider("Cantidad de gráficos", 1, 6, 3)
            for i in range(cantidad_graficos):
                col1, col2 = st.columns(2)
                with col1:
                    min_z = st.number_input(f"Mínimo Z para gráfico {i+1}", key=f"min_{i}")
                with col2:
                    max_z = st.number_input(f"Máximo Z para gráfico {i+1}", key=f"max_{i}")
                rangos_z.append((min_z, max_z))
        else:
            rangos_z = [(None, None)] * 1
        ##################################
    
    elif tipo_grafico == "Detección de Ciclos":
        st.subheader("Configuración para Detección de Ciclos")
        columna_variable = st.selectbox("Seleccione columna para analizar ciclos", [c for c in df.columns if c != fecha_col])
        nombre_variable = st.text_input("Nombre personalizado para la variable", value=columna_variable)
        fecha_inicio = st.date_input("Fecha inicio")
        fecha_fin = st.date_input("Fecha fin")
        umbral = st.number_input("Umbral para inicio-fin del ciclo", value=100.0)

    

    elif tipo_grafico == "XGBoost Analysis":
        st.subheader("XGBoost Predictive Analysis")
    
        # Selección de columna objetivo
        target_col = st.selectbox(
            "Select target column (dependent variable)",
            df.columns,
            key="xgboost_target"
        )
    
        # Selección de columnas predictoras
        feature_cols = st.multiselect(
            "Select predictor columns (independent variables)",
            [c for c in df.columns if c != target_col],
            key="xgboost_features"
        )
    
        # Selección de rango de fechas
        fecha_inicio = st.date_input("Start date")
        fecha_fin = st.date_input("End date")
    
        # Filtrar por fecha
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
        df = df[(df[fecha_col] >= pd.to_datetime(fecha_inicio)) & (df[fecha_col] <= pd.to_datetime(fecha_fin))]
    
        if df.empty:
            st.warning("No hay datos en el rango seleccionado.")
            st.stop()
    
        # ✅ Filtros para límites superiores e inferiores de cada variable seleccionada
        st.sidebar.subheader("Filtros de Variables")
        filtros = {}
        for col in feature_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            lower = st.sidebar.slider(f"Límite inferior para {col}", min_val, max_val, min_val)
            upper = st.sidebar.slider(f"Límite superior para {col}", min_val, max_val, max_val)
            filtros[col] = (lower, upper)
    
        # Aplicar filtros
        for col, (lower, upper) in filtros.items():
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
        if df.empty:
            st.warning("No hay datos después de aplicar los filtros.")
            st.stop()

    
# Botón para generar gráficos y lógica de ejecución
# Botón global para generar gráficos
# Botón global para generar gráficos

    if st.button("Generar Gráficos"):
        if tipo_grafico in ["Scatter", "Distribución Normal", "Scatter + Distribución Normal"]:
            df_filtrado = filtrar_datos(df, fecha_col, pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin), filtros)
        else:
            df_filtrado = df
    
        # Validar si hay datos
        if df_filtrado.empty:
            st.warning("No hay datos después del filtrado.")
        else:
            # Ejecutar según el tipo de gráfico
            if tipo_grafico == "Scatter":
                graficar(df_filtrado, fecha_col, columnas, color, color_linea, estilo_linea,
                         nombres_personalizados, valores_referencia,
                         mostrar_promedio, mostrar_p70, mostrar_p80, mostrar_p90)
    
            elif tipo_grafico == "Distribución Normal":
                graficar_distribucion_normal(df_filtrado, columnas, color, valores_referencia,
                                             color_linea, estilo_linea, nombres_personalizados)
    
            elif tipo_grafico == "Scatter + Distribución Normal":
                graficar_ambos(df_filtrado, fecha_col, columnas, color, color_linea, estilo_linea,
                               nombres_personalizados, valores_referencia,
                               mostrar_promedio, mostrar_p70, mostrar_p80, mostrar_p90)
    
            elif tipo_grafico == "Densidad + Scatter":
                    graficar_densidad_scatter(df_filtrado, fecha_col, eje_x, eje_y, eje_z, rangos_z,
                    bw_adjust, nombre_x, nombre_y,
                    (filtro_x_min, filtro_x_max), (filtro_y_min, filtro_y_max),
                    pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
                )

                
            elif tipo_grafico == "Detección de Ciclos":
                df_ciclos = detectar_ciclos(df, fecha_col, columna_variable, pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin), umbral, nombre_variable)
                if not df_ciclos.empty:
                    st.write("Tabla de ciclos detectados:", df_ciclos)
                    st.success("Archivo Excel generado.")
            

            if tipo_grafico == "XGBoost Analysis":
                    if target_col and feature_cols:
                        xgboost_analysis(df, fecha_col, pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin), target_col, feature_cols)
                    else:
                        st.warning("Please select target and predictor columns.")



