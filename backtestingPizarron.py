theta = 1
kalman1 = kalman1
kalman2 = kalman2
vecms_hat = []

for i, raw in data.iterrows:
    p1 = row.activo1
    p2 = row.activo2

    #UPDATE KALMAN 1

    y = p1
    x = p2
    kalman1.update(x,y)
    w0,w1 = kalman1.params
    hr = w1


    # UPDATE KALMAN 2
    x1 = p1
    x2 = p2

    eigenvector = cointegracion_johansen(data.iloc[i-252:i,:])
    e1, e2 = eigenvector
    vecm = e1 * x1 + e2 * x2
    kalman2.update(x1,x2,vecm)
    e1_hat, e2_hat = kalman2.params
    vecm_hat = e1_hat * x1 + e2_hat * x2
    vecms_hat.append(vecm_hat)
    vecms_sample = vecms_hat[-252:]


    vecm_norm = ....

    if vecm_norm > theta and active_long is None and active_short is None:
        # Buy activo 1
        available = cash * 0.4 
        n_shares_long = available // (p1 * (1 + comision))
        if available > n_shares_long * (p1 * (1 + comision)):
            cash -=
        

        # short activo 2
        n_shares_short = n_shares_long * hr
        cost = p2 * n_shares_short * comision


    # CLOSE POSITIONS
    if abs(vecm_norm) < 0.05