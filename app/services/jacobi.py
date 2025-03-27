import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def jacobi_method(A: List[List[float]], b: List[float],
                 initial_guess: Optional[List[float]] = None,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6) -> Tuple[List[float], int, float, bool, List[Dict[str, Any]]]:
    """
    Implementa el método de Jacobi para resolver sistemas de ecuaciones lineales.

    Args:
        A: Matriz de coeficientes
        b: Vector de términos independientes
        initial_guess: Vector inicial de aproximación
        max_iterations: Número máximo de iteraciones
        tolerance: Tolerancia para el criterio de convergencia

    Returns:
        Tuple con (solución, número de iteraciones, error, convergencia, historial de iteraciones)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Verificar si la matriz es diagonalmente dominante
    is_diag_dominant = all(abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i) for i in range(n))

    # Inicializar vector solución
    x = np.zeros(n) if initial_guess is None else np.array(initial_guess, dtype=float)

    # Historial de iteraciones
    iteration_history = []

    # Guardar la aproximación inicial en el historial
    iteration_history.append({
        "values": x.tolist(),
        "error": 0.0  # No hay error en la primera iteración
    })

    # Iterar hasta convergencia o máximo de iteraciones
    iteration = 0
    error = float('inf')
    converged = False

    while iteration < max_iterations and error > tolerance:
        x_new = np.zeros(n)

        for i in range(n):
            # Calcular la suma de los términos no diagonales
            sum_term = sum(A[i][j] * x[j] for j in range(n) if j != i)

            # Calcular el nuevo valor de x[i]
            x_new[i] = (b[i] - sum_term) / A[i][i]

        # Calcular el error
        error = np.linalg.norm(x_new - x, np.inf)

        # Actualizar la solución
        x = x_new.copy()
        iteration += 1

        # Guardar esta iteración en el historial
        iteration_history.append({
            "values": x.tolist(),
            "error": float(error)
        })

        if error <= tolerance:
            converged = True
            break

    return x.tolist(), iteration, float(error), converged, iteration_history