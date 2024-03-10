import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import odeint


class IGradient(ABC):
    @abstractmethod
    def getValues(self, mode):
        pass

    @abstractmethod
    def setTetta(self, tetta):
        pass

class Non_gradients(IGradient):
    def __init__(self, n, N, tetta):
        self.n = n
        self.N = N
        self.tetta = tetta


    def getValues(self, mode):
        if mode == 0:
            self.F = np.array([[-0.8, 1.0], [self.tetta[0], 0]])
            self.Psi = np.array([[self.tetta[1]], [1.0]])
            self.H = np.array([[1.0, 0]])
            self.R = 0.1
            self.x0 = np.zeros((self.n, 1))
            self.u = np.ones((self.N, 1))
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
        if mode == 1:
            self.F = np.array([[0]])
            self.Psi = np.array([[self.tetta[0], self.tetta[1]]])
            self.H = np.array([[1.0]])
            self.R = 0.3
            self.x0 = np.zeros((self.n, 1))
            self.u = np.array([[[2.], [1.]], [[1.], [2.]]])
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
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
    def Xi(self, tetta, params):
        N = params["N"]
        ki = params["ki"]
        q = params["q"]
        ytk = params["y"]

        valuesAndGradX = params["gradX"]
        valuesAndGradX.setTetta(tetta)
        F, Psi, H, R, x0, u = valuesAndGradX.getValues(mode=0)
        xt = [[(np.array([0., 0.])).reshape(2, 1) for j in range(N)] for i in range(q)]
        #print(xt)
        tk = np.arange(N)
        Xi = N * params['m'] * params['v'] * np.log(2 * np.pi) + N * params['v'] * np.log(R)

        for k in range(N - 1):
            delta = 0
            for i in range(q):
                if k == 0:
                    xt[i][k] = x0

                # Поиск производной dxdt:
                tNow = [tk[k], tk[k + 1]]
                dxdtk_One = odeint(valuesAndGradX.dxdt, Non_gradients.xTransform(self, massive=xt[i][k]), tNow)[1]
                xt[i][k + 1] = (np.array(dxdtk_One)).reshape(2, 1)
                for j in range(ki):
                    epstk = ytk[k + 1] - np.dot(H, xt[i][k + 1])
                    delta += np.dot(np.dot(epstk.transpose(), pow(R, -1)), epstk)
            Xi += delta
        return Xi[0][0] / 2.0
