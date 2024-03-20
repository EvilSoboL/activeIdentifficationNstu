import numpy as np
import non_gradients
from scipy.integrate import odeint

class Gradients(non_gradients.IGradient):
    def __init__(self, n, N, s, tetta):
        self.n = n
        self.N = N
        self.s = s
        self.tetta = tetta

    # return F, Psi, H, x_t0
    def getValues(self, mode):
        # mode == 0 был необходим для первой лабы, для вычисления dxdt
        if mode == 0:
            self.F = np.array([[-0.8, 1.0], [self.tetta[0], 0]])
            self.Psi = np.array([[self.tetta[1]], [1.0]])
            self.H = np.array([[1.0, 0]])
            self.R = 0.1
            self.x0 = np.zeros((self.n, 1))
            self.u = np.ones((self.N, 1))
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
        elif mode == 1:
            self.dF = np.array([np.array([[0]]), np.array([[0]])])
            self.dPsi = np.array([np.array([[1, 0]]), np.array([[0, 1]])])
            self.dH = np.array([np.array([[0]]), np.array([[0]])])
            self.dR = np.array([np.array([[0]]), np.array([[0]])])
            self.dx0 = np.array(np.zeros((self.n, 1)) for i in range(self.s))
            self.du_dua = np.array([[[1], [0]], [[0], [1]]])
            return self.dF, self.dPsi, self.dH, self.dR, self.dx0, self.du_dua
        if mode == 2:
            self.dF = np.array([np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])])
            self.dPsi = np.array([np.array([[0], [0]]), np.array([[1], [0]])])
            self.dH = np.array([np.array([[0, 0]]), np.array([[0, 0]])])
            self.dR = np.array([np.array([[0]]), np.array([[0]])])
            self.dx0 = np.array([np.zeros((self.n, 1)) for i in range(self.s)])
            self.du_dua = 1
            return self.dF, self.dPsi, self.dH, self.dR, self.dx0, self.du_dua

    def setTetta(self, tetta):
        self.tetta = tetta
    def xTransform(self, massive):
        m = len(massive)
        n = len(massive[0])
        result = []
        for i in range(m):
            for j in range(n):
                result.append(massive[i][j])
        return result
    def dxdt(self, xi, tk):
        massive = [0, 0]
        xDot = np.dot(self.F, (np.array(xi)).reshape((2, 1))) + np.dot(self.Psi, self.u[0][0])

        for i in range(len(xDot)):
            massive[i] = xDot[i][0]
        return massive

    def dxdtAlpha(self, dxi, tk, *args):
        massive = [0, 0]
        alpha = args[0]
        xi = [args[1], args[2]]
        dxDot = np.dot(self.dF[alpha], xi) + np.dot(self.F, (np.array(dxi)).reshape((2, 1))) \
                                                                                    + np.dot(self.Psi[alpha], self.u[0][0])
        for i in range(len(dxDot)):
            massive[i] = dxDot[i][0]
        return massive

    def gradXi(self):
        pass