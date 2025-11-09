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
                 entry_threshold: float = 1.0,
                 exit_threshold: float = 0.05,
                 kalman1_process_noise: float = 0.01,
                 kalman1_obs_noise: float = 10.0,
                 kalman2_process_noise: float = 0.001,
                 kalman2_obs_noise: float = 1.0):
        """
        Inicializa el sistema de backtesting.
        
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
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
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
        
        # Estado de la cartera
        self.cash = initial_cash
        self.n_shares_long = 0  # Acciones en largo (activo 1)
        self.n_shares_short = 0  # Acciones en corto (activo 2)
        self.active_long = None  # Precio de entrada largo
        self.active_short = None  # Precio de entrada corto
        self.borrowed_amount = 0  # Monto prestado para short
        
        # Historial
        self.vecms_hat = []
        self.history = []
    
    def calculate_commission(self, price: float, n_shares: float) -> float:
        return price * n_shares * self.commission_rate
    
    def calculate_daily_borrow_cost(self) -> float:
        if self.borrowed_amount > 0:
            daily_rate = self.borrow_rate / 252  
            return self.borrowed_amount * daily_rate
        return 0.0
    
    def open_position(self, p1: float, p2: float, hr: float, date: pd.Timestamp):
        """
        Abre una posición: largo en activo 1, corto en activo 2.
        
        Argumentos:
            p1: Precio del activo 1
            p2: Precio del activo 2
            hr: Hedge ratio
            date: Fecha actual
        """
        # Invertir 40% del cash disponible en cada activo
        available_per_asset = self.cash * (self.position_allocation / 2)
        
        # LONG en activo 1
        self.n_shares_long = available_per_asset // (p1 * (1 + self.commission_rate))
        cost_long = p1 * self.n_shares_long * (1 + self.commission_rate)
        
        if cost_long <= available_per_asset:
            self.cash -= cost_long
            self.active_long = p1
        else:
            return
        
        # SHORT en activo 2 
        self.n_shares_short = self.n_shares_long * hr
        commission_short = self.calculate_commission(p2, self.n_shares_short)
        
        # Monto prestado para el short
        self.borrowed_amount = p2 * self.n_shares_short
        
        # Recibimos el efectivo del short, pero pagamos comisión
        self.cash += self.borrowed_amount - commission_short
        self.active_short = p2
        
        print(f"\n[{date.date()}] ABRIENDO POSICIÓN:")
        print(f"  LONG {self.n_shares_long:.0f} acciones @ ${p1:.2f} = ${cost_long:.2f}")
        print(f"  SHORT {self.n_shares_short:.0f} acciones @ ${p2:.2f} = ${self.borrowed_amount:.2f}")
        print(f"  Hedge Ratio: {hr:.4f}")
        print(f"  Cash restante: ${self.cash:.2f}")
    
    def close_position(self, p1: float, p2: float, date: pd.Timestamp, reason: str = ""):
        """
        Cierra la posición actual.
        
        Args:
            p1: Precio actual del activo 1
            p2: Precio actual del activo 2
            date: Fecha actual
            reason: Razón del cierre
        """
        if self.active_long is None or self.active_short is None:
            return
        
        # CERRAR LONG (vender activo 1)
        proceeds_long = p1 * self.n_shares_long * (1 - self.commission_rate)
        self.cash += proceeds_long
        pnl_long = (p1 - self.active_long) * self.n_shares_long
        
        # CERRAR SHORT (comprar de vuelta activo 2)
        cost_short = p2 * self.n_shares_short * (1 + self.commission_rate)
        self.cash -= cost_short
        pnl_short = (self.active_short - p2) * self.n_shares_short
        
        # Devolver el monto prestado
        self.cash -= self.borrowed_amount
        
        total_pnl = pnl_long + pnl_short - (
            self.calculate_commission(self.active_long, self.n_shares_long) +
            self.calculate_commission(self.active_short, self.n_shares_short) +
            self.calculate_commission(p1, self.n_shares_long) +
            self.calculate_commission(p2, self.n_shares_short)
        )
        
        print(f"\n[{date.date()}] CERRANDO POSICIÓN {reason}:")
        print(f"  Vender LONG {self.n_shares_long:.0f} acciones @ ${p1:.2f}")
        print(f"  Cubrir SHORT {self.n_shares_short:.0f} acciones @ ${p2:.2f}")
        print(f"  PnL Long: ${pnl_long:.2f}")
        print(f"  PnL Short: ${pnl_short:.2f}")
        print(f"  PnL Total (aprox): ${total_pnl:.2f}")
        print(f"  Cash total: ${self.cash:.2f}")
        
        # Reset posición
        self.active_long = None
        self.active_short = None
        self.n_shares_long = 0
        self.n_shares_short = 0
        self.borrowed_amount = 0
    
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
            print(f"  BACKTESTING - PAIRS TRADING CON DOBLE KALMAN")
            print(f"{'='*60}")
            print(f"\nConfiguración:")
            print(f"  Capital inicial: ${self.initial_cash:,.2f}")
            print(f"  Comisión: {self.commission_rate*100:.3f}%")
            print(f"  Tasa de préstamo: {self.borrow_rate*100:.3f}% anual")
            print(f"  Asignación de posición: {self.position_allocation*100:.1f}%")
            print(f"  Ventana Johansen: {self.window_size} días")
            print(f"  Umbral entrada: {self.entry_threshold}")
            print(f"  Umbral salida: {self.exit_threshold}")
            print(f"{'='*60}\n")
        
        # Empezar después de window_size días
        for i in range(self.window_size, len(df)):
            row = df.iloc[i]
            date = df.index[i]
            p1 = row['stock_a']
            p2 = row['stock_b']
            
            # ================================================================
            # UPDATE KALMAN 1 (Hedge Ratio)
            # ================================================================
            self.kalman1.predict()
            self.kalman1.update(x=p2, y=p1)
            w0, w1 = self.kalman1.params
            hr = w1
            
            # ================================================================
            # UPDATE KALMAN 2 (VECM)
            # ================================================================
            window = df.iloc[i-self.window_size:i]
            eigenvector = compute_johansen_eigenvector(window)
            e1, e2 = eigenvector
            
            self.kalman2.predict()
            self.kalman2.update(eigenvector)
            e1_hat, e2_hat = self.kalman2.params
            
            # VECM estimado
            vecm_hat = e1_hat * p1 + e2_hat * p2
            self.vecms_hat.append(vecm_hat)
            
            # Obtener muestra de los últimos 252 VECMs
            vecms_sample = self.vecms_hat[-min(len(self.vecms_hat), self.window_size):]
            
            # Normalizar VECM
            if len(vecms_sample) > 1:
                vecm_mean = np.mean(vecms_sample)
                vecm_std = np.std(vecms_sample)
                vecm_norm = (vecm_hat - vecm_mean) / vecm_std if vecm_std > 0 else 0
            else:
                vecm_norm = 0
            
            # ================================================================
            # ACUMULAR COSTOS DE PRÉSTAMO DIARIOS
            # ================================================================
            borrow_cost = self.calculate_daily_borrow_cost()
            if borrow_cost > 0:
                self.cash -= borrow_cost
            
            # ================================================================
            # SEÑALES DE TRADING
            # ================================================================
            
            # ENTRADA: vecm_norm > threshold y no hay posición activa
            if (vecm_norm > self.entry_threshold and 
                self.active_long is None and 
                self.active_short is None):
                self.open_position(p1, p2, hr, date)
            
            # SALIDA: |vecm_norm| < exit_threshold y hay posición activa
            elif (abs(vecm_norm) < self.exit_threshold and 
                  self.active_long is not None):
                self.close_position(p1, p2, date, reason="(Reversión)")
            
            # ================================================================
            # GUARDAR ESTADO
            # ================================================================
            portfolio_value = self.cash
            if self.active_long is not None:
                # Valor de posición larga
                portfolio_value += p1 * self.n_shares_long
                # Pasivo de posición corta
                portfolio_value -= p2 * self.n_shares_short
            
            self.history.append({
                'date': date,
                'p1': p1,
                'p2': p2,
                'hedge_ratio': hr,
                'intercept': w0,
                'eigenvector_1': e1_hat,
                'eigenvector_2': e2_hat,
                'vecm_hat': vecm_hat,
                'vecm_norm': vecm_norm,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'n_shares_long': self.n_shares_long,
                'n_shares_short': self.n_shares_short,
                'active_position': 1 if self.active_long is not None else 0,
                'borrow_cost': borrow_cost
            })
            
            # Mostrar progreso
            if verbose and (i - self.window_size) % max(1, (len(df) - self.window_size) // 20) == 0:
                progress = ((i - self.window_size) / (len(df) - self.window_size)) * 100
                print(f"[{date.date()}] Progreso: {progress:5.1f}% | "
                      f"Portfolio: ${portfolio_value:,.2f} | "
                      f"VECM norm: {vecm_norm:6.2f}")
        
        # Cerrar posición final si está abierta
        if self.active_long is not None:
            final_row = df.iloc[-1]
            self.close_position(
                final_row['stock_a'], 
                final_row['stock_b'], 
                df.index[-1],
                reason="(Fin del backtest)"
            )
        
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
        
        return results_df


def run_backtest(df: pd.DataFrame, 
                 initial_cash: float = 1_000_000,
                 verbose: bool = True) -> Tuple[PairTradingBacktest, pd.DataFrame]:
    """
    Función auxiliar para ejecutar el backtesting.
    
    Argumentos:
        df: DataFrame con columnas 'stock_a' y 'stock_b'
        initial_cash: Capital inicial
    
    Returns:
        Tuple (backtest_engine, results_df)
    """
    backtest = PairTradingBacktest(
        initial_cash=initial_cash,
        commission_rate=0.00125,  
        borrow_rate=0.0025, 
        position_allocation=0.80,  
        window_size=252,
        entry_threshold=1.0,
        exit_threshold=0.05
    )
    
    results_df = backtest.run(df, verbose=verbose)
    
    return backtest, results_df


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
        'sortini_ratio': sortino_ratio,
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
    
    test_configs = [
        {'entry_threshold': 1.0, 'exit_threshold': 0.05, 
         'kalman1_process_noise': 0.01, 'kalman2_process_noise': 0.001},
        {'entry_threshold': 1.5, 'exit_threshold': 0.05, 
         'kalman1_process_noise': 0.01, 'kalman2_process_noise': 0.001},
        {'entry_threshold': 1.0, 'exit_threshold': 0.1, 
         'kalman1_process_noise': 0.01, 'kalman2_process_noise': 0.001},
        {'entry_threshold': 2.0, 'exit_threshold': 0.05, 
         'kalman1_process_noise': 0.001, 'kalman2_process_noise': 0.0001},
    ]
    
    best_sharpe = -np.inf
    best_params = None
    validation_results = []
    
    for i, config in enumerate(test_configs, 1):
        if verbose:
            print(f"\nConfig {i}/{len(test_configs)}: {config}")
        
        backtest = PairTradingBacktest(
            initial_cash=initial_cash,
            entry_threshold=config['entry_threshold'],
            exit_threshold=config['exit_threshold'],
            kalman1_process_noise=config['kalman1_process_noise'],
            kalman2_process_noise=config['kalman2_process_noise']
        )
        
        results = backtest.run(val_df, verbose=False)
        metrics = calculate_performance_metrics(results, initial_cash, verbose=False)
        
        validation_results.append({
            'config': config,
            'sharpe': metrics['sharpe_ratio'],
            'sortino': metrics['sortino_ratio'],
            'return': metrics['total_return_pct'],
            'max_dd': metrics['max_drawdown_pct']
        })
        
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_sortino = metrics['sortino_ratio']
            best_params = config
        
        if verbose:
            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f" Sortino: {metrics['sortino_ratio']:.3f} |"
                  f"Return: {metrics['total_return_pct']:.2f}% | "
                  f"MaxDD: {metrics['max_drawdown_pct']:.2f}%")
    
    if verbose:
        print(f"\n{'='*60}")
        print("MEJORES PARÁMETROS:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"  Sharpe (Validation): {best_sharpe:.3f}")
        print(f" Sortino (Validation): {best_sortino:.3f}")
        print(f"{'='*60}\n")
        print("\nFASE 2: TESTING CON MEJORES PARÁMETROS")
        print("-" * 60)
    
    backtest_test = PairTradingBacktest(initial_cash=initial_cash, **best_params)
    test_results = backtest_test.run(test_df, verbose=verbose)
    test_metrics = calculate_performance_metrics(test_results, initial_cash, verbose=verbose)
    
    return {
        'best_params': best_params,
        'validation_results': validation_results,
        'test_results': test_results,
        'test_metrics': test_metrics
    }