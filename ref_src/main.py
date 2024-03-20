from scipy.integrate import odeint
import numpy as np
import gradients as gr
import non_gradients as non_gr
from scipy.optimize import minimize, Bounds


def minimizeFirst(tettaMin, params, dx):
    non_gradX = params["non_gradX"]
    gradX = params['gradX']
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=non_gradX.Xi, x0=tettaMin, args=params,  method="SLSQP", bounds=bounds)
    grad_result = minimize(fun=non_gradX.Xi, x0=tettaMin, jac=dx, args=params, method="SLSQP", bounds=bounds)
    print("Тетты для первого порядка:\n", result)
    return np.array(result.__getitem__("x")). reshape(2, 1)

def xTransform(massive):
    m = len(massive)
    n = len(massive[0])
    result = []
    for i in range(m):
        for j in range(n):
            result.append(massive[i][j])
    return result
def y(xtk, params):
    H = params['H']
    R = params['R']
    yEnd = []
    for i in range(len(xtk)):
        yEnd.append((np.dot(H, xtk[i]) + np.random.normal(0, 1) * R))
    return yEnd

def dXi(tetta, params):
    a = 0

def main():
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 30  # Число испытаний
    ki = 1  # Initial number of system start
    q = 1
    m = v = 1.
    tetta_false = [-2.0, 0.05]
    tetta_true = [-1.5, 1]
    tk = np.arange(N + 1)
    gradX = gr.Gradients(n, N, s=s, tetta=tetta_true)
    non_gradX = non_gr.Non_gradients(n, N, tetta_true)
    F, Psi, H, R, x0, u = gradX.getValues(mode=0)
    gradX.getValues(mode=2)


    dxtdt_start = []
    x0 = [0.0, 0.0]
    for i in range(len(tk) - 1):
        tNow = [tk[i], tk[i + 1]]
        if i == 0:
            dxtdt_start.append((np.array(odeint(gradX.dxdt, x0, tNow)[1])).reshape(2, 1))
        else:
            dxtdt_start.append((np.array(odeint(gradX.dxdt, xTransform(dxtdt_start[i - 1]), tNow)[1])).reshape(2, 1))

    dx = np.array(
        [[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)]
         for alpha in range(s)])

    for i in range(len(tk) - 1):
        for alpha in range(s):
            tNow = [tk[i], tk[i + 1]]
            if i == 0:
                dx[alpha][i] = ((np.array(odeint(func=gradX.dxdtAlpha, y0=x0, t=tNow, args=(alpha, 0, 0))[1])).reshape(2, 1))
            else:
                c = xTransform(dxtdt_start[i - 1])
                dx[alpha][i] = ((np.array(odeint(gradX.dxdtAlpha, xTransform(dx[alpha][i - 1]), tNow, args=(alpha, c[0], c[1]))[1])).reshape(2, 1))
    print(dx.shape)
    params = {"N": N, "ki": ki, "q": q, "s":s, "m":m, "v":v, "y": y(dxtdt_start, {"H":H, "R":R}),"non_gradX":non_gradX, "gradX":gradX}
    print(f'Result is: \n{minimizeFirst(tetta_false, params, dx)}')
    return 0

main()
