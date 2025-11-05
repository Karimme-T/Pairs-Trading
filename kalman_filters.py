import numpy as np
import pandas as pd
from typing import Tuple, Optional


class KalmanFilterReg:
    """
    Filtro de Kalman para regresión lineal dinámica.
    
    Modela la relación: stock_a = b_0 + b_1 * stock_b + epsilon
    donde b_0 (intercept) y b_1 (hedge ratio) se actualizan dinámicamente.
    
    Attributes:
        w: Vector de parámetros [b_0, b_1]
        A: Matriz de transición de estado
        Q: Covarianza del ruido del proceso
        R: Covarianza del ruido de observación
        P: Matriz de covarianza del error de predicción
    """
    
    def __init__(self, 
                 initial_params: Optional[np.ndarray] = None,
                 transition_matrix: Optional[np.ndarray] = None,
                 process_noise: float = 0.01,
                 observation_noise: float = 10.0):
        """
        Inicializa el filtro de Kalman.
        
        Args:
            initial_params: Vector inicial [b_0, b_1]. Default: [-3, 1.5]
            transition_matrix: Matriz de transición A. Default: identidad 2x2
            process_noise: Varianza del ruido del proceso (Q). Default: 0.01
            observation_noise: Varianza del ruido de observación (R). Default: 10.0
        """
        # Estimación inicial de parámetros [intercept, hedge_ratio]
        self.w = initial_params if initial_params is not None else np.array([1, 1])
        
        # Matriz de transición (identidad = parámetros constantes + ruido)
        self.A = transition_matrix if transition_matrix is not None else np.eye(2)
        
        # Ruido en las estimaciones (covarianza del proceso)
        self.Q = np.eye(2) * process_noise
        
        # Ruido en las observaciones
        self.R = np.array([[observation_noise]])
        
        # Error en covarianza de predicciones
        self.P = np.eye(2)
        
        # Historial de parámetros
        self.history = {
            'b_0': [],
            'b_1': [],
            'P_trace': []
        }
    
    def predict(self):
        """
        Paso de predicción del filtro de Kalman.
        Actualiza la estimación del estado y su covarianza.
        """
        # Predicción de covarianza: P_t|t-1 = A * P_t-1|t-1 * A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, x: float, y: float):
        """
        Paso de actualización del filtro de Kalman.
        
        Args:
            x: Valor de stock_b (variable independiente)
            y: Valor de stock_a (variable dependiente)
        """
        # Matriz de observación C = [1, x] para el modelo y = b_0 + b_1*x
        C = np.array([[1, x]])
        
        # Covarianza de la innovación: S = C * P * C^T + R
        S = C @ self.P @ C.T + self.R
        
        # Ganancia de Kalman: K = P * C^T * S^-1
        K = self.P @ C.T @ np.linalg.inv(S)
        
        # Actualizar covarianza: P_t|t = (I - K*C) * P_t|t-1
        self.P = (np.eye(2) - K @ C) @ self.P
        
        # Actualizar estimaciones: w_t|t = w_t|t-1 + K * (y - C * w_t|t-1)
        innovation = y - C @ self.w
        self.w = self.w + (K @ innovation).flatten()
        
        # Guardar en historial
        self.history['b_0'].append(self.w[0])
        self.history['b_1'].append(self.w[1])
        self.history['P_trace'].append(np.trace(self.P))
    
    @property
    def params(self) -> Tuple[float, float]:
        """
        Retorna los parámetros actuales del modelo.
        
        Returns:
            Tuple (b_0, b_1) donde:
                b_0: intercept
                b_1: hedge ratio
        """
        return self.w[0], self.w[1]
    
    @property
    def hedge_ratio(self) -> float:
        """Retorna el hedge ratio actual (b_1)."""
        return self.w[1]
    
    @property
    def intercept(self) -> float:
        """Retorna el intercept actual (b_0)."""
        return self.w[0]
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Retorna el historial de parámetros como DataFrame.
        
        Returns:
            DataFrame con columnas ['b_0', 'b_1', 'P_trace']
        """
        return pd.DataFrame(self.history)


def fit_kalman_hedge_ratio(df: pd.DataFrame, 
                           initial_params: Optional[np.ndarray] = None,
                           process_noise: float = 0.01,
                           observation_noise: float = 10.0,
                           verbose: bool = True) -> Tuple[KalmanFilterReg, pd.DataFrame]:
    """
    Ajusta un filtro de Kalman para estimar el hedge ratio dinámico entre dos activos.
    Itera día por día actualizando el hedge ratio (beta_1) en cada observación.
    
    Modelo: stock_a = b_0 + b_1 * stock_b + epsilon
    
    Args:
        df: DataFrame con columnas 'stock_a' y 'stock_b'
        initial_params: Parámetros iniciales [b_0, b_1]. Default: [-3, 1.5]
        process_noise: Varianza del ruido del proceso. Default: 0.01
        observation_noise: Varianza del ruido de observación. Default: 10.0
        verbose: Si True, imprime información del proceso. Default: True
    
    Returns:
        Tuple (kf, results_df) donde:
            kf: Filtro de Kalman ajustado
            results_df: DataFrame con hedge ratios y spread a lo largo del tiempo
    """
    # Inicializar filtro de Kalman
    kf = KalmanFilterReg(
        initial_params=initial_params,
        process_noise=process_noise,
        observation_noise=observation_noise
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  AJUSTE DE FILTRO DE KALMAN - HEDGE RATIO DINÁMICO")
        print(f"{'='*60}")
        print(f"\nParámetros iniciales:")
        print(f"  b_0 (intercept): {kf.w[0]:.4f}")
        print(f"  b_1 (hedge ratio): {kf.w[1]:.4f}")
        print(f"\nConfiguracion del filtro:")
        print(f"  Process noise (Q): {process_noise}")
        print(f"  Observation noise (R): {observation_noise}")
        print(f"  Número de observaciones: {len(df)}")
        print(f"\n{'='*60}")
        print("Iterando día por día...")
        print(f"{'='*60}")
    
    # Preparar listas para almacenar resultados día por día
    dates = []
    b_0_values = []
    b_1_values = []
    p_trace_values = []
    
    # Iterar sobre cada día de datos
    total_days = len(df)
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        # Paso de predicción
        kf.predict()
        
        # Paso de actualización con los precios del día
        kf.update(x=row['stock_b'], y=row['stock_a'])
        
        # Almacenar resultados del día
        dates.append(idx)
        b_0_values.append(kf.w[0])
        b_1_values.append(kf.w[1])
        p_trace_values.append(np.trace(kf.P))
        
        # Mostrar progreso cada 10% de los datos
        if verbose and i % max(1, total_days // 10) == 0:
            progress = (i / total_days) * 100
            print(f"  Día {i:5d}/{total_days} ({progress:5.1f}%) | "
                  f"b_0: {kf.w[0]:7.4f} | b_1: {kf.w[1]:7.4f} | "
                  f"Date: {idx.date()}")
    
    # Crear DataFrame con resultados
    results_df = df.copy()
    
    # Agregar parámetros del filtro
    results_df['b_0'] = b_0_values
    results_df['b_1'] = b_1_values
    results_df['P_trace'] = p_trace_values
    
    # Calcular spread día por día: stock_a - (b_0 + b_1 * stock_b)
    results_df['spread'] = results_df['stock_a'] - (
        results_df['b_0'] + results_df['b_1'] * results_df['stock_b']
    )
    
    # Calcular estadísticas del spread
    spread_mean = results_df['spread'].mean()
    spread_std = results_df['spread'].std()
    spread_min = results_df['spread'].min()
    spread_max = results_df['spread'].max()
    
    if verbose:
        print(f"\n{'='*60}")
        print("✓ AJUSTE COMPLETADO")
        print(f"{'='*60}")
        print(f"\nParámetros finales (último día):")
        print(f"  b_0 (intercept): {kf.w[0]:.4f}")
        print(f"  b_1 (hedge ratio): {kf.w[1]:.4f}")
        print(f"\nEstadísticas del hedge ratio (b_1):")
        print(f"  Inicial: {b_1_values[0]:.4f}")
        print(f"  Final: {b_1_values[-1]:.4f}")
        print(f"  Media: {np.mean(b_1_values):.4f}")
        print(f"  Std: {np.std(b_1_values):.4f}")
        print(f"  Min: {np.min(b_1_values):.4f}")
        print(f"  Max: {np.max(b_1_values):.4f}")
        print(f"\nEstadísticas del spread:")
        print(f"  Media: {spread_mean:.4f}")
        print(f"  Std: {spread_std:.4f}")
        print(f"  Min: {spread_min:.4f}")
        print(f"  Max: {spread_max:.4f}")
        print(f"{'='*60}\n")
    
    # Actualizar el historial del objeto KalmanFilterReg
    kf.history['b_0'] = b_0_values
    kf.history['b_1'] = b_1_values
    kf.history['P_trace'] = p_trace_values
    
    return kf, results_df


def calculate_spread(df: pd.DataFrame, b_0: float, b_1: float) -> pd.Series:
    """
    Calcula el spread entre dos activos dado un hedge ratio.
    
    Spread = stock_a - (b_0 + b_1 * stock_b)
    
    Args:
        df: DataFrame con columnas 'stock_a' y 'stock_b'
        b_0: Intercept
        b_1: Hedge ratio
    
    Returns:
        Series con el spread
    """
    return df['stock_a'] - (b_0 + b_1 * df['stock_b'])