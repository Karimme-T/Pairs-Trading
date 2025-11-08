import yfinance as yf
import pandas as pd
import numpy as np
from utils import download_historical_data, split_val_test, plot_kalman_estimation
from kalman_filters import fit_kalman_hedge_ratio, analyze_vecm_window

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
    print("    PAIRS TRADING PIPELINE")
    print("=" * 50)
    
    # Get selected pair info
    stock_a, stock_b = pairs[selected_pair_idx]
    print(f"\nSelected pair: {stock_a} & {stock_b}")
    
    # Download historical data
    selected_pair = pairs[selected_pair_idx]
    df = download_historical_data(selected_pair, years=7)
    val_df, test_df = split_val_test(df)
    
    print("\n" + "=" * 50)
    print("Pipeline initialization complete!")
    print("=" * 50)
    
    # Preview the data
    print(f"\nFirst 5 days:")
    print(val_df.head())
    print(f"\nLast 5 days:")
    print(test_df.tail())

    #visualizamos los resultados de kalman
    print("\n" + "="*50)
    print("Ajustando Kalman 1 (Hedge rario)")
    print("="*50)

    kalman1, resultados_kalman1 = fit_kalman_hedge_ratio(
        df=val_df,
        process_noise=0.01,
        observation_noise=10.0,
        verbose=True
    )

    print("\n Resultados del Kalman 1:")
    print(resultados_kalman1[['b_0', 'b_1', 'spread']].tail())

    print("\n" + "="*50)
    print("Kalman 2 (VECM)")
    print("="*50)

    if len(val_df) >= 252:
        vecm_resultados = analyze_vecm_window(
            df=val_df,
            window_size=252,
            verbose=True
        )

        print("\n" + "="*50)
        print("Resultados")
        print("=" * 50)
        print(f"\n Kalman 1 (último día):")
        print(f" Hedge ratio (b_1): {resultados_kalman1['b_1'].iloc[-1]:.4f}")
        print(f" Intercept (b_0): {resultados_kalman1['b_0'].iloc[-1]:.4f}")
        print(f" Spread: {resultados_kalman1['spread'].iloc[-1]:.4f}")

        print(f"\n Kalman 2 (ventana de 252 días):")
        print(f" Eigenvector: [{vecm_resultados['eigenvector'][0]:.6f}, {vecm_resultados['eigenvector'][1]:.6f}]")
        print(f" VECM coeficientes: [{vecm_resultados['vecm_coefs'][0]:.6f}, {vecm_resultados['vecm_coefs'][1]:.6f}")
        print(f" Theta: {vecm_resultados['theta']:.6f}")

        if vecm_resultados['theta'] < 0:
            print(f"n Theta negativo")
        else:
            print(f"\n Theta positivo")
    else:
        print(f"\n No hay suficentes datos")
        
    
    # TODO: Add further pipeline steps here
    # - Cointegration testing
    # - Spread calculation
    # - Signal generation
    # - Backtesting


if __name__ == "__main__":
    main()