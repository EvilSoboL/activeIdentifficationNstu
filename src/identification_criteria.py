import numpy as np


class IdentificationCriteria:
    """2 уровень сложности, 2 вариант"""

    def __init__(self, F: np.ndarray, Psi: np.ndarray, R: np.ndarray, H: np.ndarray):
        self.F: np.ndarray = F
        self.Psi: np.ndarray = Psi
        self.R: np.ndarray = R
        self.H: np.ndarray = H

    def get_identification_criteria(self, N: int, m: int, v: int) -> np.ndarray:
        """
        Критерий идентификации.
        """

        return N * m * v * np.log(2 * np.pi) + N * v * np.log(np.linalg.det(self.R))

    @staticmethod
    def set_ki() -> int:
        """
        Количество подачи сигнала на вход системы.
        """
        return 1

    @staticmethod
    def set_qi() -> int:
        """
        Количество входных сигналов.
        """
        return 1

    @staticmethod
    def get_x_i() -> np.ndarray:
        """
        Действительное значение вектора x.
        """
        return np.array([[1], [1]])

    @staticmethod
    def get_u_i() -> np.ndarray:
        """
        Детерминированный вектор управления.
        """
        return np.array([[1]])

    def get_x_i_k_plus(self) -> np.ndarray:  # Шаг 6
        """
        ?
        """
        x_i: np.ndarray = self.get_x_i()  # Шаг 5
        u_i: np.ndarray = self.get_u_i()  # Шаг 4

        return np.dot(self.F, x_i) + np.dot(self.Psi, u_i)

    def get_y_k_plus(self) -> np.ndarray:
        """
        Вектор выхода с шумами.
        """
        x_i_k_plus: np.ndarray = self.get_x_i_k_plus()
        y: np.ndarray = np.dot(self.H, x_i_k_plus) + np.random.normal(0, 1) * self.R

        return y

    def get_epsilon_k_plus(self) -> np.ndarray:  # Шаг 8
        """
        Вектор обновления в момент времени t(k+1)
        """
        x_i_k_plus: np.ndarray = self.get_x_i_k_plus()
        y_k_plus: np.ndarray = self.get_y_k_plus()

        return y_k_plus + np.dot(self.H, x_i_k_plus)

    def get_delta(self) -> np.ndarray:  # Шаг 9
        epsilon_k_plus: np.ndarray = self.get_epsilon_k_plus()

        return np.dot(epsilon_k_plus.T * self.R ** (-1), epsilon_k_plus)

    def get_identification_criteria_with_delta(self, N: int, m: int, v: int):  # Шаг 12, 14
        identification_criteria: np.ndarray = self.get_identification_criteria(N, m, v)
        delta: np.ndarray = self.get_delta()

        return 0.5 * (identification_criteria + delta)
