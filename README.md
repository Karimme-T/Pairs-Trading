# Pairs-Trading

Implementación completa de una estrategia cuantitativa de pairs trading (trading de pares). Este proyecto combina tests de cointegración, filtros de Kalman secuenciales y representación VECM (Vector Error Correction Model) para identificar señales de venta y compra entre dos activos.

## Características Principales

* Tests de Cointegración: Implementación de Engle-Granger y Johansen
* Doble Filtro de Kalman:
  * Kalman 1: Estimación dinámica de hedge ratio
  * Kalman 2: Suavización de eigenvector de cointegración
* Señales basadas en Z-score: Normalización mediante VECM para detección de divergencias
* Backtesting: Walk-forward analysis con períodos train/validation/test

Par Analizado

Activos: AAPL (Apple Inc.) / ORCL (Oracle Corporation)
Período Total: 2010-2025 (15 años de datos)
Capital Inicial: $1,000,000 USD

## Instalación y uso
### Prerrequisitos
Python 3.8+
pip
virtualenv (recomendado)

### Instalación
Clonar el repositorio
git clone https://github.com/Karimme-T/Pairs-Trading.git

### Crear entorno virtual
python -m venv venv
source venv/bin/activate 

### Instalar dependencias
pip install -r requirements.txt

### Correr código
python main.py

## Visualizaciones
El proyecto genera las siguientes visualizaciones:

* Equity Curve: Evolución del valor del portfolio
* VECM Normalizado: Z-score con umbrales de entrada/salida
* Hedge Ratio Dinámico: Evolución temporal del β_t
* Posiciones Activas: Períodos con exposición al mercado

## Licencia
Este proyecto está bajo la Licencia GPL-3.0. Ver archivo LICENSE para más detalles.

## Autoras
Ana Luisa Espinoza López
Karimme Anahi Tejeda Olmos

# IMPORTANTE
Este proyecto es únicamente para fines educativos y de investigación. No constituye asesoramiento financiero ni recomendación de inversión. El trading de valores financieros conlleva riesgo de pérdida de capital. Resultados pasados no garantizan rendimientos futuros.
