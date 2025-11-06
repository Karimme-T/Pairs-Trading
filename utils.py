import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import dateparser
import matplotlib.pyplot as plt
import numpy as np



def download_historical_data(selected_pair: list, years: int = 7) -> pd.DataFrame:
    """
    Download historical price data for the selected pair.
    
    Args:
        selected_pair: List containing two stock symbols [stock_a, stock_b]
        years: Number of years of historical data to download
    
    Returns:
        DataFrame with stock_a and stock_b close prices
    """

    stock_a, stock_b = selected_pair[0], selected_pair[1]
    
    print(f"\n=== Downloading data for {stock_a} & {stock_b} ===")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        # Download data for both stocks
        print(f"Fetching {stock_a} data...")
        data_a = yf.download(stock_a, start=start_date, end=end_date, progress=False)
        
        print(f"Fetching {stock_b} data...")
        data_b = yf.download(stock_b, start=start_date, end=end_date, progress=False)
        
        if data_a.empty or data_b.empty:
            raise ValueError("Could not retrieve data for one or both stocks.")
        
        # Extract Close prices and create DataFrame
        # Use .squeeze() to handle both single and multi-column DataFrames
        close_a = data_a['Close'].squeeze()
        close_b = data_b['Close'].squeeze()
        
        # Combine into single DataFrame with ticker names as columns
        df = pd.DataFrame({
    'stock_a': close_a,
    'stock_b': close_b
    })
        
        # Parse and format dates
        df.index = pd.to_datetime(df.index.map(lambda x: dateparser.parse(str(x))))
        
        # Sort from oldest to newest
        df = df.sort_index(ascending=True)
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"\n✓ Successfully downloaded:")
        print(f"  Combined dataset: {len(df)} trading days")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def split_val_test(df):
    """
    Split a price DataFrame (already sorted by date) into two halves:
    validation (first half) and test (second half).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['Date', 'stock_a', 'stock_b'].

    Returns
    -------
    val_df : pd.DataFrame
        First half (older dates).
    test_df : pd.DataFrame
        Second half (recent dates).
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    midpoint = len(df) // 2

    # Keep all columns, including 'Date'
    val_df = df.iloc[:midpoint].copy()
    test_df = df.iloc[midpoint:].copy()

    return val_df, test_df

#test for kalman 1 debug
def plot_kalman_estimation(results_df: pd.DataFrame, 
                          stock_a_name: str = 'Stock A', 
                          stock_b_name: str = 'Stock B',
                          figsize: tuple = (14, 8),
                          save_path: str = None):
    """
    Grafica la comparación entre stock_a real y la estimación usando el hedge ratio de Kalman.
    
    La estimación se calcula como: stock_a_estimated = b_0 + b_1 * stock_b
    
    Args:
        results_df: DataFrame con columnas ['stock_a', 'stock_b', 'b_0', 'b_1']
        stock_a_name: Nombre del activo A para el título
        stock_b_name: Nombre del activo B para el título
        figsize: Tamaño de la figura (ancho, alto)
        save_path: Ruta para guardar la figura (opcional)
    """
    
    # Calcular la estimación de stock_a usando el hedge ratio dinámico
    # stock_a_estimated = b_0 + b_1 * stock_b
    results_df['stock_a_estimated'] = results_df['b_0'] + results_df['b_1'] * results_df['stock_b']
    
    # Calcular el error de estimación
    results_df['estimation_error'] = results_df['stock_a'] - results_df['stock_a_estimated']
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Subplot 1: Comparación de precios reales vs estimados de stock_a
    ax1 = axes[0]
    ax1.plot(results_df.index, results_df['stock_a'], 
             label=f'{stock_a_name} (Real)', color='blue', linewidth=1.5, alpha=0.8)
    ax1.plot(results_df.index, results_df['stock_a_estimated'], 
             label=f'{stock_a_name} (Estimado: β₀ + β₁·{stock_b_name})', 
             color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax1.set_ylabel('Precio ($)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Verificación del Filtro de Kalman: {stock_a_name} Real vs Estimado', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 2: Hedge Ratio dinámico (β₁)
    ax2 = axes[1]
    ax2.plot(results_df.index, results_df['b_1'], 
             label='Hedge Ratio (β₁)', color='green', linewidth=1.5)
    ax2.axhline(y=results_df['b_1'].mean(), color='darkgreen', 
                linestyle=':', linewidth=1.5, label=f'Media: {results_df["b_1"].mean():.4f}')
    ax2.set_ylabel('Hedge Ratio (β₁)', fontsize=11, fontweight='bold')
    ax2.set_title('Evolución del Hedge Ratio Dinámico', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 3: Error de estimación
    ax3 = axes[2]
    ax3.plot(results_df.index, results_df['estimation_error'], 
             label='Error de Estimación', color='purple', linewidth=1.0, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax3.fill_between(results_df.index, results_df['estimation_error'], 0, 
                     alpha=0.3, color='purple')
    ax3.set_ylabel('Error ($)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Fecha', fontsize=11, fontweight='bold')
    ax3.set_title('Error de Estimación (Real - Estimado)', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Calcular y mostrar estadísticas
    mae = np.abs(results_df['estimation_error']).mean()
    rmse = np.sqrt((results_df['estimation_error']**2).mean())
    r_squared = 1 - (np.sum(results_df['estimation_error']**2) / 
                     np.sum((results_df['stock_a'] - results_df['stock_a'].mean())**2))
    
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS DE LA ESTIMACIÓN DEL FILTRO DE KALMAN")
    print(f"{'='*60}")
    print(f"MAE (Error Absoluto Medio):     ${mae:.4f}")
    print(f"RMSE (Error Cuadrático Medio):  ${rmse:.4f}")
    print(f"R² (Coeficiente de determinación): {r_squared:.4f}")
    print(f"\nHedge Ratio (β₁):")
    print(f"  Inicial:  {results_df['b_1'].iloc[0]:.4f}")
    print(f"  Final:    {results_df['b_1'].iloc[-1]:.4f}")
    print(f"  Media:    {results_df['b_1'].mean():.4f}")
    print(f"  Std:      {results_df['b_1'].std():.4f}")
    print(f"\nIntercept (β₀):")
    print(f"  Inicial:  {results_df['b_0'].iloc[0]:.4f}")
    print(f"  Final:    {results_df['b_0'].iloc[-1]:.4f}")
    print(f"  Media:    {results_df['b_0'].mean():.4f}")
    print(f"{'='*60}\n")
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")
    
    plt.show()
    
    return fig, axes