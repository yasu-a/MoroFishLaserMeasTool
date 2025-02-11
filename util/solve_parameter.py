import numpy as np
from sympy import symbols, Eq, solve


# noinspection PyPep8Naming
def solve_equations_camera(points: np.ndarray) -> np.ndarray:
    # 行列の初期化
    A = np.zeros((len(points) + sum(1 for _, _, _, _, v in points if v is not None), 11))
    B = np.zeros((len(points) + sum(1 for _, _, _, _, v in points if v is not None), 1))

    # 行列AとBの構築
    row_index = 0
    for x, y, z, u, v in points:
        A[row_index, 0:4] = [x, y, z, 1]
        A[row_index, 8:11] = [-u * x, -u * y, -u * z]
        B[row_index, 0] = u
        row_index += 1

    for x, y, z, u, v in points:
        if v is not None:
            A[row_index, 4:8] = [x, y, z, 1]
            A[row_index, 8:11] = [-v * x, -v * y, -v * z]
            B[row_index, 0] = v
            row_index += 1

    # 行列Aの擬似逆行列を計算
    A_pseudo_inv = np.linalg.pinv(A)

    # パラメータの解を計算
    solution_matrix = np.dot(A_pseudo_inv, B)

    solution_matrix = solution_matrix.ravel()
    solution_matrix = np.append(solution_matrix, 1)
    solution_matrix = solution_matrix.reshape(3, 4)

    return solution_matrix


# def solve_camera_parameters(self) -> CameraParameterSolution | None:
#     points = self._project.image_points.points_array
#     try:
#         solution = self._solve_equations_matrix(points)
#     except np.linalg.LinAlgError:
#         return None
#     else:
#         return CameraParameterSolution(
#             solution=solution,
#         )


def solve_equations_laser(points: np.ndarray) -> np.ndarray:
    # 未知数
    b11, b12, b13 = symbols('b11 b12 b13')

    # 連立方程式の作成
    equations = [
        Eq(x * b11 + y * b12 + z * b13 + 1, 0)
        for x, y, z in points
    ]

    # 連立方程式の解
    solution = solve(equations, (b11, b12, b13))

    return np.array([
        solution[b11].evalf(),
        solution[b12].evalf(),
        solution[b13].evalf(),
        1.0,
    ])
