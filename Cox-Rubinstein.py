import numpy as np

def crr_cash_or_nothing(S0, K, H, r, sigma, T, Q=1, N=200, 
                        option_type="call", barrier_type=None):
    """
    Modelo Cox-Ross-Rubinstein (CRR) para opciones cash-or-nothing
    con barreras knock-in / knock-out.
    
    Parámetros:
    S0 : float
        Precio spot inicial
    K : float
        Strike
    H : float or None
        Barrera (si None, es vanilla digital)
    r : float
        Tasa libre de riesgo
    sigma : float
        Volatilidad
    T : float
        Tiempo al vencimiento
    Q : float
        Pago fijo
    N : int
        Número de pasos del árbol
    option_type : str
        "call" o "put"
    barrier_type : str o None
        "down-in", "down-out", "up-in", "up-out" o None
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Matriz de valores terminales
    V = np.zeros((N+1, N+1))

    for j in range(N+1):
        S_T = S0 * (u**j) * (d**(N-j))
        
        # Condiciones de barrera
        knocked = False
        if barrier_type is not None:
            # reconstruir camino máximo/mínimo
            S_path = [S0 * (u**k) * (d**(i-k)) for i in range(N+1) for k in range(i+1)]
            if "down" in barrier_type:
                if np.min(S_path) <= H: knocked = True
            elif "up" in barrier_type:
                if np.max(S_path) >= H: knocked = True

        # Payoff digital
        if option_type == "call":
            payoff = Q if S_T > K else 0
        else:  # put
            payoff = Q if S_T < K else 0

        # Ajustar con barrera
        if barrier_type == "down-out" and knocked:
            payoff = 0
        elif barrier_type == "down-in" and not knocked:
            payoff = 0
        elif barrier_type == "up-out" and knocked:
            payoff = 0
        elif barrier_type == "up-in" and not knocked:
            payoff = 0

        V[N, j] = payoff

    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            V[i, j] = disc * (p * V[i+1, j+1] + (1-p) * V[i+1, j])

    return V[0,0]

# Ejemplo
S0, K, H = 100, 100, 90
r, sigma, T, Q = 0.05, 0.2, 1.0, 10

precio_crr = crr_cash_or_nothing(S0, K, H, r, sigma, T, Q, N=500,
                                 option_type="call", barrier_type="down-out")
print("Prima CRR (down-and-out cash-or-nothing call):", round(precio_crr, 4))
