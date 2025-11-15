import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from kalman_filters import KalmanFilterReg
from kalman_filters import compute_johansen_eigenvector, compute_vecm_representation, compute_theta


class KalmanFilterVECM:
    """
    Actualiza dinámicamente los eigenvectores estimados.
    """
    
    def __init__(self, 
                 dim: int = 2,
                 process_noise: float = 0.001,
                 observation_noise: float = 1.0):
        self.dim = dim
        self.w = np.zeros(dim)  # [e1_hat, e2_hat]
        self.A = np.eye(dim)
        self.Q = np.eye(dim) * process_noise
        self.R = np.array([[observation_noise]])
        self.P = np.eye(dim)
    
    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, eigenvector: np.ndarray):
        """
        Actualiza el filtro con el eigenvector observado.
        
        Argumentos:
            eigenvector: Eigenvector de Johansen [e1, e2]
        """
        # Matriz de observación
        C = np.eye(self.dim)
        
        # Covarianza de la innovación
        S = C @ self.P @ C.T + self.R * np.eye(self.dim)
        
        # Ganancia de Kalman
        K = self.P @ C.T @ np.linalg.inv(S)
        
        # Actualizar covarianza
        self.P = (np.eye(self.dim) - K @ C) @ self.P
        
        # Actualizar estado
        innovation = eigenvector - C @ self.w
        self.w = self.w + (K @ innovation).flatten()
    
    @property
    def params(self) -> Tuple[float, float]:
        """Retorna eigenvector estimado [e1_hat, e2_hat]."""
        return self.w[0], self.w[1]


class PairTradingBacktest:
    
    def __init__(self,
                 initial_cash: float = 1_000_000,
                 commission_rate: float = 0.00125,
                 borrow_rate: float = 0.0025,
                 position_allocation: float = 0.80,
                 window_size: int = 252,
                 theta: float = 0.5,
                 kalman1_process_noise: float = 0.01,
                 kalman1_obs_noise: float = 10.0,
                 kalman2_process_noise: float = 0.001,
                 kalman2_obs_noise: float = 1.0):
        """
        Inicia backtesting.
        
        Argumentos:
            initial_cash: Capital inicial ($1,000,000)
            commission_rate: Comisión por trade (0.125% = 0.00125)
            borrow_rate: Tasa de préstamo anualizada (0.25% = 0.0025)
            position_allocation: % del cash a usar (80% = 0.80)
            window_size: Ventana para Johansen (252 días)
            entry_threshold: Umbral de entrada (theta > entry_threshold)
            exit_threshold: Umbral de salida (|vecm_norm| < exit_threshold)
        """

        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.borrow_rate = borrow_rate
        self.position_allocation = position_allocation
        self.window_size = window_size
        self.theta = theta
        
        # Filtros de Kalman
        self.kalman1 = KalmanFilterReg(
            process_noise=kalman1_process_noise,
            observation_noise=kalman1_obs_noise
        )

        self.kalman2 = KalmanFilterVECM(
            dim=2,
            process_noise=kalman2_process_noise,
            observation_noise=kalman2_obs_noise
        )
        
        # cartera
        self.cash = initial_cash
        self.n_shares_long = 0  
        self.n_shares_short = 0  
        self.active_long = None 
        self.active_short = None  
        self.borrowed_amount = 0  
        self.position_type = None
        
        self.vecms_hat = []
        self.history = []

    
    def calculate_commission(self, price: float, n_shares: float) -> float:
        return abs(price * n_shares * self.commission_rate)
    
    
    def calculate_daily_borrow_cost(self) -> float:
        if self.borrowed_amount > 0:
            daily_rate = self.borrow_rate / 252  
            return self.borrowed_amount * daily_rate
        return 0.0
    

    def open_long_A_short_B(self, p1: float, p2: float, hr: float, date: pd.Timestamp):
        """
        Abre posición: LONG A (stock_a), SHORT B (stock_b).
        Señal: Z > θ 
        """
        inversion_total = self.initial_cash * self.position_allocation
        amount_per_side = inversion_total / 2
        
        # LONG stock_a
        self.n_shares_long = amount_per_side / (p1 * (1 + self.commission_rate))
        commission_long = self.calculate_commission(p1, self.n_shares_long)
        cost_long = p1 * self.n_shares_long + commission_long
        
        if cost_long > self.cash:
            print(f"[{date.date()}] Error, no hay suficiente dinero para abrir posición")
            return
        
        self.cash -= cost_long
        self.active_long = p1
        
        # SHORT stock_b
        self.n_shares_short = self.n_shares_long * hr
        commission_short = self.calculate_commission(p2, self.n_shares_short)

        self.borrowed_amount = p2 * self.n_shares_short

        self.cash += self.borrowed_amount - commission_short
        self.active_short = p2
        self.position_type = 'long_A_short_B'
        
        print(f"\n[{date.date()}] Abriendo posición: LONG A, SHORT B")
        print(f"  LONG {self.n_shares_long:.0f} shares A @ ${p1:.2f}")
        print(f"  SHORT {self.n_shares_short:.0f} shares B @ ${p2:.2f}")
        print(f"  Cash: ${self.cash:.2f}")
    
    
    def open_short_A_long_B(self, p1: float, p2: float, hr: float, date: pd.Timestamp):
        """
        Abre posición: SHORT A (stock_a), LONG B (stock_b).
        Señal: Z < -θ 
        """
        inversion_total = self.initial_cash * self.position_allocation
        amount_per_side = inversion_total / 2
        
        # SHORT stock_a
        self.n_shares_short = amount_per_side / (p1 * (1 + self.commission_rate))
        commission_short = self.calculate_commission(p1, self.n_shares_short)

        self.borrowed_amount = p1 * self.n_shares_short
        self.cash += self.borrowed_amount - commission_short
        self.active_short = p1
        
        # LONG stock_b
        self.n_shares_long = self.n_shares_short * hr
        commission_long = self.calculate_commission(p2, self.n_shares_long)
        cost_long = p2 * self.n_shares_long + commission_long
        
        if cost_long > self.cash:
            #si no hay suficiente cash
            print(f"[{date.date()}] Error, no hay suficiente cash")
            self.cash -= (self.borrowed_amount - commission_short)
            self.borrowed_amount = 0
            self.active_short = None
            self.n_shares_short = 0
            return
        
        self.cash -= cost_long
        self.active_long = p2
        self.position_type = 'short_A_long_B'
        
        print(f"\n[{date.date()}] Abriendo posición: SHORT A, LONG B")
        print(f"  SHORT {self.n_shares_short:.0f} shares A @ ${p1:.2f}")
        print(f"  LONG {self.n_shares_long:.0f} shares B @ ${p2:.2f}")
        print(f"  Cash: ${self.cash:.2f}")

    
    def close_position(self, p1: float, p2: float, date: pd.Timestamp, reason: str = ""):
        """
        Cierra la posición actual.
        
        Args:
            p1: Precio actual del activo 1
            p2: Precio actual del activo 2
            date: Fecha actual
            reason: Razón del cierre
        """
        if self.position_type is None:
            return
        
        if self.position_type == 'long_A_short_B':
            #Cerramos Long A
            proceeds_long = p1 * self.n_shares_long 
            commission_long = self.calculate_commission(p1, self.n_shares_long)
            self.cash += proceeds_long - commission_long

            #Cerramos Short B
            cost_short = p2 * self.n_shares_short
            commission_short = self.calculate_commission(p2, self.n_shares_short)
            self.cash -= cost_short + commission_short

            pnl_long = (p1 - self.active_long) * self.n_shares_long
            pnl_short = (self.active_short - p2) * self.n_shares_short
            pnl = pnl_long + pnl_short - commission_long - commission_short

        else:
            # cerramos short A
            cost_short = p1 * self.n_shares_short 
            commission_short  = self.calculate_commission(p1, self.n_shares_short)
            self.cash -= cost_short + commission_short

            # ceramos long B
            proceeds_long = p2 * self.n_shares_long
            commission_long = self.calculate_commission(p2, self.n_shares_long)
            self.cash += proceeds_long - commission_long

            pnl_short = (self.active_short - p1) * self.n_shares_short
            pnl_long = (p2 - self.active_long) * self.n_shares_long
            pnl = pnl_short + pnl_long - commission_short - commission_long

        print(f"\n[{date.date()}] Cerrando: {self.position_type} {reason}")
        print(f" PnL: ${pnl:.2f}")
        print(f" Cash: ${self.cash:.2f}")

        
        # Reset posición
        self.active_long = None
        self.active_short = None
        self.n_shares_long = 0
        self.n_shares_short = 0
        self.borrowed_amount = 0
        self.position_type = None

    
    def calculate_portfolio_value(self, p1:float, p2:float) -> float:
        """
        Calcula el valor actual del portafolio.
        """

        portfolio_value = self.cash

        if self.position_type == 'long_A_short_B':
            portfolio_value += p1 * self.n_shares_long
            portfolio_value -= p2 * self.n_shares_short

        elif self.position_type == 'short_A_long_B':
            portfolio_value -= p1 * self.n_shares_short
            portfolio_value += p2 * self.n_shares_long
        
        return portfolio_value

    
    def run(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Ejecuta el backtesting.
        
        Argumentos:
            df: DataFrame con columnas 'stock_a' y 'stock_b'
        
        Returns:
            DataFrame con resultados del backtesting
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f" Backtesting - PAIRS TRADING")
            print(f"{'='*60}")
            print(f"\n Configuración:")
            print(f"  Capital inicial: ${self.initial_cash:,.2f}")
            print(f"  Comisión: {self.commission_rate*100:.3f}%")
            print(f"  Tasa de préstamo: {self.borrow_rate*100:.3f}% anual")
            print(f"  Umbral entrada: {self.theta}")
            print(f"{'='*60}\n")

        df = df.reset_index()
        
        # Empezar después de window_size días
        for i in range(self.window_size, len(df)):
            current_date = df.loc[i, 'Date']
            p1 = df.loc[i, 'stock_a']
            p2 = df.loc[i, 'stock_b']

            # Costos de préstamos diarios
            daily_borrow_cost = self.calculate_daily_borrow_cost()
            self.cash -= daily_borrow_cost

            #Ventana de históricos
            window_data = df.iloc[i-self.window_size:i]
            

            # UPDATE KALMAN 1 (Hedge Ratio)
            self.kalman1.predict()
            self.kalman1.update(x=p2, y=p1)
            hr_hat = self.kalman1.hedge_ratio

            try:
                eigenvector = compute_johansen_eigenvector(window_data)

                # UPDATE KALMAN 2 (VECM)
                self.kalman2.predict()
                self.kalman2.update(eigenvector)
                e1_hat, e2_hat = self.kalman2.params

                #coeficientes alpha
                vecm_coefs = compute_vecm_representation(window_data, eigenvector)
                theta_revertion = compute_theta(eigenvector, vecm_coefs)

                #calcular vecm 
                vecm_current = e1_hat * p1 + e2_hat * p2
                self.vecms_hat.append(vecm_current)

                if len(self.vecms_hat) >= 252:
                    vecm_window = self.vecms_hat[-252:]
                else:
                    vecm_window = self.vecms_hat
            
                # Normalizar VECM
                vecm_mean = np.mean(vecm_window)
                vecm_std = np.std(vecm_window)

                if vecm_std > 0:
                    z_score = (vecm_current - vecm_mean) / vecm_std
                else:
                    z_score = 0
            
            except Exception as e:
                if verbose:
                    print(f"[{current_date.date()}] Error en cálculos: {str(e)}")
                z_score = 0
                e1_hat, e2_hat = 0,0
                theta_revertion = 0
                vecm_current = 0.0

            #señales usando theta
            if self.position_type is None:
                if z_score > self.theta:
                    self.open_long_A_short_B(p1, p2, hr_hat, current_date)
                
                elif z_score < -self.theta:
                    self.open_short_A_long_B(p1, p2, hr_hat, current_date)
            else:
                if abs(z_score) < 0.05:
                    self.close_position(p1, p2, current_date, "Z < 0.05")
                
                elif self.position_type == 'long_A_short_B' and z_score < -self.theta:
                    self.close_position(p1, p2, current_date, "Reversión de señal")
                
                elif self.position_type == 'short_A_long_B' and z_score > self.theta:
                    self.close_position(p1, p2, current_date, "Reversión de señal")

            # GUARDAR ESTADO
            portfolio_value = self.calculate_portfolio_value(p1,p2)

            self.history.append({
                'date': current_date,
                'stock_a_price': p1,
                'stock_b_price': p2,
                'hedge_ratio': hr_hat,
                'e1_hat': e1_hat,
                'e2_hat': e2_hat,
                'theta_revertion': theta_revertion,
                'vecm_current': vecm_current,
                'z_score': z_score,
                'theta_threshold': self.theta,
                'cash': self.cash,
                'borrowed_amount': self.borrowed_amount,
                'n_shares_long': self.n_shares_long,
                'n_shares_short': self.n_shares_short,
                'portfolio_value': portfolio_value,
                'position_type': self.position_type if self.position_type else 'none',
                'active_position': 1 if self.position_type is not None else 0,
                'borrow_cost': daily_borrow_cost
            })

                        
        # Cerrar posición final si está abierta
        if self.position_type is not None:
            final_date = df.loc[len(df)-1, 'Date']
            final_p1 = df.loc[len(df)-1, 'stock_a']
            final_p2 = df.loc[len(df)-1, 'stock_b']
            self.close_position(final_p1, final_p2, final_date, "Fin del periodo")

        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(self.history)
        
        if verbose:
            final_value = results_df['portfolio_value'].iloc[-1]
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100
            
            print(f"\n{'='*60}")
            print("RESULTADOS FINALES")
            print(f"{'='*60}")
            print(f"Capital inicial: ${self.initial_cash:,.2f}")
            print(f"Capital final: ${final_value:,.2f}")
            print(f"Retorno total: {total_return:.2f}%")
            print(f"{'='*60}\n")

        if verbose:
            results_df = pd.DataFrame(self.history)
            print(f"\n{'='*60}")
            print(f"ANÁLISIS DE SEÑALES")
            print(f"{'='*60}")
            print(f"Theta threshold usado: {self.theta}")
            print(f"Z-score estadísticas:")
            print(f"  Min: {results_df['z_score'].min():.2f}")
            print(f"  Max: {results_df['z_score'].max():.2f}")
            print(f"  Mean: {results_df['z_score'].mean():.2f}")
            print(f"  Std: {results_df['z_score'].std():.2f}")
            print(f"\nVeces que |z_score| > {self.theta}: {(results_df['z_score'] > self.theta).sum()}")
            print(f"Veces que |z_score| < 0.05: {(results_df['z_score'] < 0.05).sum()}")
            print(f"{'='*60}\n")

        return results_df



def calculate_performance_metrics(results_df: pd.DataFrame, 
                                  initial_cash: float,
                                  verbose: bool = True) -> dict:
    """
    Calcula métricas de performance del backtesting.
    """
    results_df = results_df.copy()
    results_df['daily_return'] = results_df['portfolio_value'].pct_change()
    
    final_value = results_df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash
    
    mean_daily_return = results_df['daily_return'].mean()
    std_daily_return = results_df['daily_return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

    negative_returns = results_df['daily_return'][results_df['daily_return'] < 0]
    downside_std = negative_returns.std()
    sortino_ratio = (mean_daily_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    cumulative = results_df['portfolio_value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    years = len(results_df) / 252
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    total_borrow_costs = results_df['borrow_cost'].sum()

    
    # Contar trades 
    position_changes = results_df['active_position'].diff().fillna(0)
    total_trades = len(position_changes[position_changes == -1])  
    
    metrics = {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annualized_return': annualized_return,
        'annualized_return_pct': annualized_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'total_borrow_costs': total_borrow_costs,
        'days_traded': len(results_df),
        'years_traded': years
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("MÉTRICAS DE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Capital Inicial:        ${metrics['initial_cash']:,.2f}")
        print(f"Capital Final:          ${metrics['final_value']:,.2f}")
        print(f"Retorno Total:          {metrics['total_return_pct']:.2f}%")
        print(f"Retorno Anualizado:     {metrics['annualized_return_pct']:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:          {metrics['sortino_ratio']:.3f}")
        print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        print(f"Calmar Ratio:           {metrics['calmar_ratio']:.3f}")
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Costos de Préstamo:     ${metrics['total_borrow_costs']:,.2f}")
        print(f"Días Trading:           {metrics['days_traded']}")
        print(f"{'='*60}\n")
    
    return metrics


def walk_forward_analysis(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         initial_cash: float = 1_000_000,
                         verbose: bool = True) -> dict:
    """
    Walk-forward analysis: optimiza en val, testea en test.
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print("WALK-FORWARD ANALYSIS")
        print(f"{'='*60}\n")
        print("FASE 1: OPTIMIZACIÓN EN VALIDATION SET")
        print("-" * 60)

    theta_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    best_calmar = -np.inf
    best_params = None
    validation_results = []

    for theta in theta_grid:
        config = {
            'theta': theta,
            'kalman1_process_noise': 0.01,
            'kalman2_process_noise': 0.001
        }

        try:
            backtest = PairTradingBacktest(
                initial_cash=initial_cash,
                **config
            )

            results = backtest.run(val_df, verbose=False)
            metrics = calculate_performance_metrics(results, initial_cash, verbose=False)

            validation_results.append({
                'theta': theta,
                'config': config,
                'calmar': metrics['calmar_ratio'],
                'sortino': metrics['sortino_ratio'],
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['total_return_pct'],
                'max_dd': metrics['max_drawdown_pct'],
                'trades': metrics['total_trades'],
                'final_value': metrics['final_value']
            })

            if verbose:
                print(f"{theta:<8.2f} {metrics['total_trades']:<8} "
                      f"{metrics['total_return_pct']:<12.2f} "
                      f"{metrics['sharpe_ratio']:<10.3f} "
                      f"{metrics['sortino_ratio']:<10.3f} "
                      f"{metrics['calmar_ratio']:<10.3f} "
                      f"{metrics['max_drawdown_pct']:<10.2f}")
                
            if metrics['total_trades'] >= 3:
                if metrics['calmar_ratio'] > best_sortino:
                    best_calmar = metrics['calmar_ratio']
                    best_params = config.copy()
        
        except Exception as e:
            if verbose:
                print(f"{theta:<8.2f} ERROR: {str(e)[:40]}")
            continue
    
    if verbose:
        print("-" * 70)

    if best_params is None:
        if len(validation_results) > 0:
            validation_results.sort(key=lambda x: (x['trades'], x['calmar']), reverse=True)
            best_result = validation_results[0]
            best_params = best_result['config']
            best_sortino = best_result['calmar']

            if verbose:
                print("\n Ninguna configuración funcionó. Usando parámetros por defecto.")
            
            
        else:
            if verbose:
                print("\n Ninguna configuración funcionó. Usando default")
            
            best_params = {
                'theta': 0.5,
                'kalman1_process_noise': 0.01,
                'kalman2_process_noise': 0.001
            }
            best_calmar = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print("MEJORES PARÁMETROS:")
        print(f"{'='*60}")
        print(f"  Theta (θ): {best_params['theta']:.3f}")
        print(f"  Kalman 1 process noise: {best_params['kalman1_process_noise']}")
        print(f"  Kalman 2 process noise: {best_params['kalman2_process_noise']}")
        print(f"  Sortino en Validation: {best_calmar:.3f}")
        print(f"{'='*60}\n")

        print("Top configuraciones:")
        validation_results.sort(key=lambda x: x['sortino'], reverse=True)
        for i, result in enumerate(validation_results[:5], 1):
            print(f"{i}. θ={result['config']['theta']:.2f} | "
                  f"Calmar={result['calmar']:.3f} | "
                  f"Sharpe={result['sharpe']:.3f} | "
                  f"Sortino={result['sortino']:.3f} | "
                  f"Return={result['return']:.2f}% "
                  f"MaxDD={result['max_dd']:.2f}% | "
                  f"Trades={result['trades']} ")
        
        print("\nFASE 2: TESTING CON MEJORES PARÁMETROS")
        print("-" * 60)

    
    backtest_test = PairTradingBacktest(initial_cash=initial_cash, **best_params)
    test_results = backtest_test.run(test_df, verbose=verbose)
    test_metrics = calculate_performance_metrics(test_results, initial_cash, verbose=verbose)
    
    return {
        'best_params': best_params,
        'validation_results': validation_results,
        'test_results': test_results,
        'test_metrics': test_metrics,
        'best_theta': best_params['theta']
    }