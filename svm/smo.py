import numpy as np
from scipy.optimize import fsolve


def get_parabolic_function(a, b) -> callable:
    def df(alpha):
        return -a * alpha + b

    return df


def smo(
        train_data: np.ndarray,
        gamma: float
) -> np.ndarray:
    n_samples = len(train_data)
    alphas = np.random.normal(loc=0.0, scale=1.0, size=(1, n_samples))
    train_data_alphas = np.hstack((train_data, alphas))

    while True:
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                i_sample = train_data_alphas[i]
                j_sample = train_data_alphas[j]

                alpha_1, y1 = i_sample[-1], i_sample[-2]
                alpha_2, y2 = j_sample[-1], j_sample[-2]

                const = _find_const(
                    train_data=train_data_alphas,
                    n_samples=n_samples,
                    except_i=i,
                    except_j=j,
                )

                alpha_1 = y1 * (const - alpha_2 * y2)
                train_data_alphas[i][-1] = alpha_1

                a, b = _find_a_and_b(
                    train_data=train_data_alphas,
                    n_samples=n_samples,
                    except_i=j,
                    gamma=gamma,
                )
                new_alpha_2 = _find_parabola_extremum(a, b)

                train_data_alphas[j][-1] = new_alpha_2

                const = _find_const(
                    train_data=train_data_alphas,
                    n_samples=n_samples,
                    except_i=i,
                    except_j=j,
                )

                alpha_2 = y2 * (const - alpha_1 * y1)
                train_data_alphas[j][-1] = alpha_2

                a, b = _find_a_and_b(
                    train_data=train_data_alphas,
                    n_samples=n_samples,
                    except_i=j,
                    gamma=gamma,
                )
                new_alpha_1 = _find_parabola_extremum(a, b)

                train_data_alphas[i][-1] = new_alpha_1

    return train_data_alphas[:, -1]


def _find_const(
        train_data: np.ndarray,
        n_samples: int,
        except_i,
        except_j
) -> float:
    const = 0.0
    for k in range(n_samples):
        if k in (except_i, except_j):
            continue

        k_sample = train_data[k]
        alpha, y = k_sample[-1], k_sample[-2]
        const += alpha * y

    return const


def _find_a_and_b(
        train_data: np.ndarray,
        n_samples: int,
        except_i: int,
        gamma: float,
) -> tuple[float, float]:
    a = 0.
    b = 0.

    for k in range(n_samples):
        if k == except_i:
            continue

        k_sample = train_data[k]
        alpha_1, y_1 = k_sample[-1], k_sample[-2]
        b += alpha_1

        for l in range(n_samples):
            l_sample = train_data[l]
            alpha_2, y_2 = l_sample[-1], l_sample[-2]

            l_sample_features = l_sample[:-2]
            k_sample_features = k_sample[:-2]
            a += (alpha_1 * alpha_2 * y_1 * y_2 *
                  np.e ** (-1 * gamma * (np.linalg.norm(k_sample_features - l_sample_features) ** 2)))

    return a, b


def _find_parabola_extremum(
        a: float,
        b: float
):
    parabolic_function = get_parabolic_function(a, b)
    start_guess = 1.0
    alpha_zero = fsolve(parabolic_function, start_guess)
    return alpha_zero[0]
