import numpy as np
from scipy.optimize import minimize


class IdentificationCriteria:
    def __init__(self, R: np.ndarray, H: np.ndarray, N: int, P0: np.ndarray):
        self.R: np.ndarray = R
        self.H: np.ndarray = H
        self.N: int = N
        self.P0: np.ndarray = P0

    def generate_input_x(self, theta_true) -> list[np.ndarray[float]]:
        """
        Генерация сигнала входных значений x
        """
        F: np.ndarray = self.get_F(theta_true[0])
        Psi: np.ndarray = self.get_Psi(theta_true[1])

        x: list = [None] * self.N
        U_i: int = 2

        for k in range(self.N):
            if k == 0:
                x[k] = np.array([[0],
                                 [0]])
            else:
                x[k] = F @ x[k - 1] + Psi * U_i

        return x

    @staticmethod
    def _check_eigenvalues(array: np.ndarray) -> bool:
        eigenvalues = np.linalg.eigvals(array)
        return np.all(eigenvalues < 1)

    def generate_output_y(self, theta_true) -> np.ndarray:
        y: np.ndarray = np.zeros(self.N)
        x: list[np.array] = self.generate_input_x(theta_true)
        v = np.random.normal(0, np.sqrt(self.R), self.N)

        for k in range(self.N):
            y[k] = self.H @ x[k] + v[k]

        return y

    def get_F(self, theta: int or float) -> np.ndarray:
        F: np.ndarray = np.array([[-0.8, 1],
                                  [theta, 0]])
        if self._check_eigenvalues(F):
            return F
        else:
            Warning('Собственные значения матрицы F должны быть < 1!')

    @staticmethod
    def get_Psi(theta: int or float) -> np.ndarray:
        Psi: np.ndarray = np.array([[theta], [1]])
        return Psi

    def get_identification_criteria(self, theta: list[int], observation: np.ndarray) -> float:
        m: int = 1
        v: int = 1

        x: list[np.array] = self.generate_input_x(theta)

        delta: float = 0

        for k in range(self.N):
            if k == 0:
                identification_criteria: float = (self.N * m * v * np.log(2 * np.pi) +
                                                  self.N * v * np.log(np.linalg.det(np.array([[self.R]]))))
            else:
                epsilon_tk: np.ndarray = observation[k] - self.H @ x[k]
                delta: np.ndarray = delta + epsilon_tk.T * self.R ** (-1) * epsilon_tk
                identification_criteria += delta
            if k <= (self.N - 1):
                delta = 0
        identification_criteria /= 2
        return identification_criteria

    def get_non_grad_estimate(self, theta_true):
        n: int = 5
        theta_1 = np.zeros((2, n))

        for i in range(n):
            y: np.ndarray = self.generate_output_y(theta_true)
            mmp_x = lambda t: self.get_identification_criteria(t, y)
            result = minimize(mmp_x, np.array([-1, 1]), method='SLSQP', bounds=[(-2, -0.05), (0.01, 1.5)])
            theta_1[:, i] = result.x

        return theta_1

    @staticmethod
    def get_derivatives() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dF_1: np.ndarray = np.array([[np.array([0, 0]), np.array([0, 0])],
                                   [np.array([1, 0]), np.array([0, 0])]])
        dF_2: np.ndarray = np.array([[np.array([0, 0]), np.array([0, 0])],
                                     [np.array([0, 0]), np.array([0, 0])]])
        dPsi_1: np.ndarray = np.array([np.array([0, 1]), np.array([0, 0])])
        dPsi_2: np.ndarray = np.array([np.array([0, 1]), np.array([0, 0])])
        dH: np.ndarray = np.array([np.array([0, 0]), np.array([0, 0])])
        dR: np.ndarray = np.array([[0, 0]])
        dx0 = np.array([np.array([0, 0]), np.array([0, 0])])

        return dF_1, dF_2, dPsi_1, dPsi_2, dH, dR, dx0

    def get_grad_identification_criteria(self, theta: list[int], observation: np.ndarray) -> float:
        v = 1
        dF_1, dF_2, dPsi_1, dPsi_2, dH, dR, dx0 = self.get_derivatives()
        for k in range(self.N):
            if k == 0:
                d_identification_criteria_1 = v / 2 * self.N * np.trace(self.R ** (-1) * dR)
                d_identification_criteria_2 = v / 2 * self.N * np.trace(self.R ** (-1) * dR)
                delta_1 = 0
                delta_2 = 0

        return d_identification_criteria_1





