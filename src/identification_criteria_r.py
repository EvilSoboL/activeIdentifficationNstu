import numpy as np
from scipy.optimize import minimize


class IdentificationCriteria:
    def __init__(self, R: float, H: np.ndarray, N: int, P0: np.ndarray):
        self.R: float = R
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
    def get_derivatives() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dF: np.ndarray = np.array([[np.array([0]), np.array([0])],
                                   [np.array([1]), np.array([0])]])
        dPsi: np.ndarray = np.array([np.array([0]), np.array([0])])
        dH: np.ndarray = np.array([np.array([0]), np.array([0])])
        dR: np.ndarray = np.array([[0]])
        dx0 = np.array([np.array([0]), np.array([0])])

        return dF, dPsi, dH, dR, dx0

    def get_grad_identification_criteria(self, theta: list[int], observation: np.ndarray) -> np.ndarray[float, float]:
        v = 1
        F: np.ndarray = self.get_F(theta[0])
        Psi: np.ndarray = self.get_Psi(theta[1])
        dF, dPsi, dH, dR, dx0 = self.get_derivatives()

        x: list[np.array] = self.generate_input_x(theta)
        u: int = 1
        delta: int = 0

        for k in range(self.N):
            d_identification_criteria = v / 2 * self.N * np.trace(self.R ** (-1) * dR)
            if k == 0:
                x_tk = dx0
            else:
                x_tk = F @ x_tk + Psi * u
            dx_tk: np.ndarray = dF @ x_tk + F @ dx0 + dPsi * u
            depsilon_tk: np.ndarray = -dH @ x_tk - self.H @ dx_tk
            epsilon_tk: np.ndarray = observation[k] - self.H @ x_tk
            delta = (delta + depsilon_tk.T * self.R ** (-1) * epsilon_tk -
                     0.5 * epsilon_tk.T * self.R ** (-1) * dR * self.R ** (-1) * epsilon_tk)
            d_identification_criteria = d_identification_criteria + delta

        return d_identification_criteria[1]

    def get_grad_estimate(self, theta_true):
        n: int = 5
        theta_1 = np.zeros((2, n))

        for i in range(n):
            y: np.ndarray = self.generate_output_y(theta_true)
            mmp_x = lambda t: self.get_identification_criteria(t, y)
            mmp_grad = lambda t: self.get_grad_identification_criteria(t, y)
            result = minimize(mmp_x, np.array([-1, 1]), jac=mmp_grad, method='SLSQP', bounds=[(-2, -0.05), (0.01, 1.5)])
            theta_1[:, i] = result.x

        return theta_1

    @staticmethod
    def u(t_k: int or float, return_value: float = 1.0) -> float:
        return return_value

    def identification_criteria(self, theta: np.ndarray) -> float:
        """Критерий идентификации"""
        N: int = self.N
        F: np.ndarray = self.get_F(theta[0])
        Psi: np.ndarray = self.get_Psi(theta[1])
        H: np.ndarray = self.H
        R: float = self.R
        x_to: np.ndarray = np.array([[0],
                                     [0]])
        m, v = 1, 1
        delta: int or float = 0

        for k in range(N):
            if k == 0:
                ident_criteria: float = N * m * v * np.log(2 * np.pi) + N * v * np.log(np.linalg.det(np.array([[self.R]])))
                x_tk: np.ndarray = F @ x_to + Psi * self.u(k)
            else:
                x_tk: np.ndarray = F @ x_tk + Psi * self.u(k)
            epsilon: np.ndarray = self.epsilon(k, x_tk)
            delta += epsilon.T * R ** (-1) * epsilon
            if k <= (N-1):
                delta = 0
        return 0.5 * ident_criteria

    def epsilon(self, t_k: int or float, x_tk: np.ndarray) -> np.ndarray:
        """m-мерный вектор обновления в момент времени t_k."""
        return self.y(t_k) - self.H @ x_tk

    def y(self, t_k: int) -> float:
        """m-мерный вектор измерения (выхода) в момент времени t_k."""
        f = 10
        A = 1
        sigma = 0.5

        signal = A * np.sin(2 * np.pi * f * t_k)

        noise = np.random.normal(0, sigma, 1)

        noisy_signal = signal + noise
        return noisy_signal

    def grad_identification_criteria(self, theta: list[int]) -> np.ndarray[float, float]:
        v = 1
        F: np.ndarray = self.get_F(theta[0])
        Psi: np.ndarray = self.get_Psi(theta[1])
        dF, dPsi, dH, dR, dx0 = self.get_derivatives()

        u: int = 1
        delta: int = 0
        x_to: np.ndarray = np.array([[0],
                                     [0]])

        for k in range(self.N):
            delta: int = 0
            if k == 0:
                d_identification_criteria = v / 2 * self.N * np.trace(self.R ** (-1) * dR)
                x_tk = x_to
                dx_tk = dx0
            else:
                x_tk = F @ x_tk + Psi * u
                dx_tk = dF * x_tk + F @ dx_tk + dPsi * u
            depsilon_tk: np.ndarray = -dH * x_tk - self.H @ dx_tk
            epsilon_tk: np.ndarray = self.y(k) - self.H @ x_tk
            delta = (delta + depsilon_tk.T * self.R ** (-1) * epsilon_tk -
                     0.5 * epsilon_tk.T * self.R ** (-1) * dR * self.R ** (-1) * epsilon_tk)
            d_identification_criteria = d_identification_criteria + delta
        return d_identification_criteria

    def estimates(self):
        n = 5
        θ_1 = np.zeros((2, n))
        θ_2 = np.zeros((2, n))
        for i in range(n):
            mmp_χ = lambda t: self.identification_criteria(t)
            res_1 = minimize(mmp_χ, np.array([-1, 1]), method='SLSQP', bounds=[(-2, -0.05), (0.01, 1.5)])
            θ_1[:, i] = res_1.x

            mmp_grad = lambda t: self.grad_identification_criteria(t)
            res_2 = minimize(mmp_χ, np.array([-1, 1]), jac=mmp_grad, method='SLSQP', bounds=[(-2, -0.05), (0.01, 1.5)])
            θ_2[:, i] = res_2.x
        return θ_1, θ_2
