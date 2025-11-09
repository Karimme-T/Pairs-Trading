import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import download_historical_data, split_train_val_test, plot_kalman_estimation
from kalman_filters import fit_kalman_hedge_ratio, analyze_vecm_window
from backtesting import run_backtest, walk_forward_analysis, calculate_performance_metrics


# Manually select pair by changing this index (0, 1, or 2)
selected_pair_idx = 0
pairs = [
    ['AAPL', 'ORCL'],
    ['ADBE', 'CSCO'],
    ['ADBE', 'MSFT'],
    ['CL', 'KMB'],
    ['INTC', 'ORCL'],
    ['KMB', 'MDLZ'],
    ['MMM', 'UPS']
]


def main():
    """Main pipeline execution."""
    print("=" * 50)
    print("    PAIRS TRADING")
    print("=" * 50)
    
    # Get selected pair info
    stock_a, stock_b = pairs[selected_pair_idx]
    print(f"\nSelected pair: {stock_a} & {stock_b}")
    
    # Download historical data
    selected_pair = pairs[selected_pair_idx]
    df = download_historical_data(selected_pair, years=15)
    train_df, val_df, test_df = split_train_val_test(df)

    print("\n" + "=" * 50)
    print("Vista previa de los datos")
    print("=" * 50)
    
    print(f"\n Primeros 5 díass:")
    print(val_df.head())
    print(f"\nÚltimos 5 días:")
    print(test_df.tail())


    print("\n" + "=" * 70)
    print("Análisis exploratorio usando train")
    print("=" * 70)

    kf1_train, resultados_kf1_train = fit_kalman_hedge_ratio(df=train_df, process_noise=0.01, observation_noise=10.0, verbose=True)

    print("\nResultados del Kalman 1 (Training - últimos 5 días):")
    print(resultados_kf1_train[['b_0', 'b_1', 'spread']].tail())

    print("\n Kalman 2: VECM")

    if len(train_df) >= 252:
        vecm_train = analyze_vecm_window(df=train_df, window_size=252, verbose=True)
    
        print(f"\nResultados del Kalman 2 (Training - última ventana de 252 días):")
        print(f"  Eigenvector: [{vecm_train['eigenvector'][0]:.6f}, {vecm_train['eigenvector'][1]:.6f}]")
        print(f"  VECM coefs: [{vecm_train['vecm_coefs'][0]:.6f}, {vecm_train['vecm_coefs'][1]:.6f}]")
        print(f"  Theta: {vecm_train['theta']:.6f}")

        if vecm_train['theta'] < 0:
            print("Theta negativo")
        else:
            print("Theta positivo")
    else:
        print("No hay suficientes datos en train")


    print("\n" + "=" * 50)
    print("Walk forward analysis")
    print("=" * 50)

    wf_resultados = walk_forward_analysis(train_df=train_df, val_df=val_df, test_df=test_df, initial_cash=1_000_000, verbose=True)

    test_resultados = wf_resultados['test_results']

    print("\n" + "=" * 70)
    print("RESULTADOS DEL TEST SET")
    print("=" * 70)

    print("\nPrimeros 10 días (Test Set):")
    print(test_resultados[['date', 'portfolio_value', 'vecm_norm', 'active_position']].head(10))
    
    print("\nÚltimos 10 días (Test Set):")
    print(test_resultados[['date', 'portfolio_value', 'vecm_norm', 'active_position']].tail(10))
    
    print("\n" + "=" * 70)
    print("VISUALIZACIONES")
    print("=" * 70)

    fig, axes = plt.subplots(4, 1, figsize=(16, 22))
    
    axes[0].plot(test_resultados['date'], test_resultados['portfolio_value'], linewidth=2)
    axes[0].axhline(y=1_000_000, color='r', linestyle='--', label='Inicial', linewidth=1.5)
    axes[0].set_title(f'Portfolio Value - {stock_a}/{stock_b}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('USD', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(test_resultados['date'], test_resultados['vecm_norm'], linewidth=1.5)
    axes[1].axhline(y=1.0, color='g', linestyle='--', label='Entrada')
    axes[1].axhline(y=-1.0, color='g', linestyle='--')
    axes[1].axhline(y=0.05, color='r', linestyle='--', label='Salida')
    axes[1].axhline(y=-0.05, color='r', linestyle='--')
    axes[1].set_title('VECM Normalizado', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(test_resultados['date'], test_resultados['hedge_ratio'], color='purple')
    axes[2].set_title('Hedge Ratio Dinámico', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].fill_between(test_resultados['date'], 0, test_resultados['active_position'], alpha=0.4)
    axes[3].set_title('Posiciones Activas', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Fecha', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'backtest_{stock_a}_{stock_b}.png', dpi=300)
    plt.show()
    

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\nPar seleccionado: {stock_a} / {stock_b}")
    print(f"Período completo: {df.index[0].date()} a {df.index[-1].date()}")
    print(f"Total de días: {len(df)} días ({len(df)/252:.1f} años)")

    print(f"\nSplit de datos:")
    print(f"  Training (60%):   {len(train_df):,} días | {train_df.index[0].date()} a {train_df.index[-1].date()}")
    print(f"  Validation (20%): {len(val_df):,} días | {val_df.index[0].date()} a {val_df.index[-1].date()}")
    print(f"  Testing (20%):    {len(test_df):,} días | {test_df.index[0].date()} a {test_df.index[-1].date()}")
    
    print(f"\nMejores parámetros:")
    for key, value in wf_resultados['best_params'].items():
        print(f"  {key}: {value}")

    print(f"\nMétricas de Performance (Test Set):")

    test_metrics = wf_resultados['test_metrics']

    print(f"  Retorno Total:      {test_metrics['total_return_pct']:>8.2f}%")
    print(f"  Retorno Anualizado: {test_metrics['annualized_return_pct']:>8.2f}%")
    print(f"  Sharpe Ratio:       {test_metrics['sharpe_ratio']:>8.3f}")
    print(f"  Sortino Ratio:      {test_metrics['sortino_ratio']:>8.3f}")
    print(f"  Max Drawdown:       {test_metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Calmar Ratio:       {test_metrics['calmar_ratio']:>8.3f}")
    print(f"  Total Trades:       {test_metrics['total_trades']:>8}")

if __name__ == "__main__":
    main()