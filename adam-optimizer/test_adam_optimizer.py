import numpy as np
import sys
import os
import importlib.util

# Добавляем текущую директорию в path для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Импортируем модуль с дефисом в имени через importlib
spec = importlib.util.spec_from_file_location("adam_optimizer_module", 
                                               os.path.join(current_dir, "adam-optimizer.py"))
adam_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adam_optimizer)
adam_step = adam_optimizer.adam_step


def test_zero_gradient_no_change():
    """grad=0 → param не меняется"""
    param = 5.0
    grad = 0.0
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert param_new == param, f"При нулевом градиенте param не должен меняться"
    print("✓ test_zero_gradient_no_change passed")


def test_t1_bias_correction():
    """t=1 → значительная bias correction"""
    param = 0.0
    grad = 1.0
    m = 0.0
    v = 0.0
    t = 1
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t, lr, beta1, beta2)
    
    # m_new = 0.9 * 0 + 0.1 * 1 = 0.1
    expected_m_new = 0.1
    # v_new = 0.999 * 0 + 0.001 * 1 = 0.001
    expected_v_new = 0.001
    # m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
    expected_m_hat = 1.0
    # v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
    expected_v_hat = 1.0
    # param_new = 0 - 0.001 * 1.0 / (sqrt(1.0) + 1e-8) ≈ -0.001
    expected_param_new = -lr * expected_m_hat / (np.sqrt(expected_v_hat) + 1e-8)
    
    assert np.isclose(m_new, expected_m_new), f"m_new: {m_new} != {expected_m_new}"
    assert np.isclose(v_new, expected_v_new), f"v_new: {v_new} != {expected_v_new}"
    assert np.isclose(param_new, expected_param_new, rtol=1e-5), f"param_new: {param_new} != {expected_param_new}"
    print("✓ test_t1_bias_correction passed")


def test_scalar_input():
    """Скалярные входы"""
    param = 1.0
    grad = 0.5
    m = 0.1
    v = 0.01
    t = 2
    lr = 0.01
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t, lr)
    
    assert isinstance(param_new, np.ndarray), "param_new должен быть np.ndarray"
    assert param_new.shape == (), "Скаляр должен остаться скаляром (0-d array)"
    print("✓ test_scalar_input passed")


def test_array_input_1d():
    """1D массив входов"""
    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])
    m = np.zeros(3)
    v = np.zeros(3)
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert param_new.shape == (3,), f"Форма param_new: {param_new.shape} != (3,)"
    assert m_new.shape == (3,), f"Форма m_new: {m_new.shape} != (3,)"
    assert v_new.shape == (3,), f"Форма v_new: {v_new.shape} != (3,)"
    print("✓ test_array_input_1d passed")


def test_array_input_2d():
    """2D массив входов (например, веса матрицы)"""
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])
    m = np.zeros((2, 2))
    v = np.zeros((2, 2))
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert param_new.shape == (2, 2), f"Форма param_new: {param_new.shape} != (2, 2)"
    assert m_new.shape == (2, 2), f"Форма m_new: {m_new.shape} != (2, 2)"
    assert v_new.shape == (2, 2), f"Форма v_new: {m_new.shape} != (2, 2)"
    print("✓ test_array_input_2d passed")


def test_moment_update():
    """Проверка обновления моментов"""
    param = 0.0
    grad = 2.0
    m = 1.0
    v = 0.5
    t = 1
    beta1 = 0.9
    beta2 = 0.999
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t, beta1=beta1, beta2=beta2)
    
    # m_new = 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
    expected_m_new = beta1 * m + (1 - beta1) * grad
    # v_new = 0.999 * 0.5 + 0.001 * 4.0 = 0.4995 + 0.004 = 0.5035
    expected_v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    
    assert np.isclose(m_new, expected_m_new), f"m_new: {m_new} != {expected_m_new}"
    assert np.isclose(v_new, expected_v_new), f"v_new: {v_new} != {expected_v_new}"
    print("✓ test_moment_update passed")


def test_bias_correction_t10():
    """t=10 → bias correction меньше чем при t=1"""
    param = 0.0
    grad = 1.0
    m = 0.0
    v = 0.0
    t = 10
    beta1 = 0.9
    beta2 = 0.999
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t, beta1=beta1, beta2=beta2)
    
    # m_new = 0.1 * 1 = 0.1
    # m_hat = 0.1 / (1 - 0.9^10) = 0.1 / (1 - 0.3487) ≈ 0.1536
    expected_m_new = (1 - beta1) * grad
    expected_m_hat = expected_m_new / (1 - beta1 ** t)
    
    # v_new = 0.001 * 1 = 0.001
    # v_hat = 0.001 / (1 - 0.999^10) ≈ 0.001 / 0.00995 ≈ 0.1005
    expected_v_new = (1 - beta2) * (grad ** 2)
    expected_v_hat = expected_v_new / (1 - beta2 ** t)
    
    assert np.isclose(m_new, expected_m_new), f"m_new: {m_new} != {expected_m_new}"
    assert np.isclose(v_new, expected_v_new), f"v_new: {v_new} != {expected_v_new}"
    print("✓ test_bias_correction_t10 passed")


def test_numerical_stability():
    """Численная стабильность с eps"""
    param = 0.0
    grad = 0.0
    m = 0.0
    v = 0.0
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert not np.isnan(param_new), "param_new не должен быть NaN"
    assert not np.isinf(param_new), "param_new не должен быть Inf"
    print("✓ test_numerical_stability passed")


def test_large_gradient():
    """Большой градиент"""
    param = 0.0
    grad = 1000.0
    m = 0.0
    v = 0.0
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert not np.isnan(param_new), "param_new не должен быть NaN"
    assert not np.isinf(param_new), "param_new не должен быть Inf"
    # При большом градиенте обновление должно быть значительным
    assert param_new < 0, "param должен уменьшиться при положительном градиенте"
    print("✓ test_large_gradient passed")


def test_large_array_vectorized():
    """Векторизация на большом массиве"""
    param = np.random.randn(10000)
    grad = np.random.randn(10000)
    m = np.zeros(10000)
    v = np.zeros(10000)
    t = 1
    
    param_new, m_new, v_new = adam_step(param, grad, m, v, t)
    
    assert param_new.shape == (10000,), f"Форма param_new: {param_new.shape}"
    assert np.all(np.isfinite(param_new)), "Все элементы должны быть конечными"
    print("✓ test_large_array_vectorized passed")


def test_return_types():
    """Проверка типов возврата"""
    param = 1.0
    grad = 0.5
    m = 0.0
    v = 0.0
    t = 1
    
    result = adam_step(param, grad, m, v, t)
    
    assert isinstance(result, tuple), "Должен возвращать tuple"
    assert len(result) == 3, "Должен возвращать 3 элемента"
    param_new, m_new, v_new = result
    assert isinstance(param_new, np.ndarray), "param_new должен быть np.ndarray"
    assert isinstance(m_new, np.ndarray), "m_new должен быть np.ndarray"
    assert isinstance(v_new, np.ndarray), "v_new должен быть np.ndarray"
    print("✓ test_return_types passed")


def test_learning_rate_effect():
    """Влияние learning rate"""
    param = 0.0
    grad = 1.0
    m = 0.0
    v = 0.0
    t = 1
    
    param_new_small, _, _ = adam_step(param, grad, m, v, t, lr=0.001)
    param_new_large, _, _ = adam_step(param, grad, m, v, t, lr=0.1)
    
    # Больший lr → большее обновление
    assert abs(param_new_large) > abs(param_new_small), "Больший lr должен давать большее обновление"
    print("✓ test_learning_rate_effect passed")


def test_beta_parameters():
    """Влияние beta1 и beta2"""
    param = 0.0
    grad = 1.0
    m = 0.0
    v = 0.0
    t = 1
    
    # Стандартные значения
    param_new_std, m_new_std, v_new_std = adam_step(param, grad, m, v, t)
    
    # Другие значения beta
    param_new_alt, m_new_alt, v_new_alt = adam_step(param, grad, m, v, t, beta1=0.5, beta2=0.5)
    
    # Разные beta дают разные результаты
    assert not np.isclose(m_new_std, m_new_alt), "Разные beta1 должны давать разные m_new"
    assert not np.isclose(v_new_std, v_new_alt), "Разные beta2 должны давать разные v_new"
    print("✓ test_beta_parameters passed")


if __name__ == "__main__":
    test_zero_gradient_no_change()
    test_t1_bias_correction()
    test_scalar_input()
    test_array_input_1d()
    test_array_input_2d()
    test_moment_update()
    test_bias_correction_t10()
    test_numerical_stability()
    test_large_gradient()
    test_large_array_vectorized()
    test_return_types()
    test_learning_rate_effect()
    test_beta_parameters()
    
    print("\n✅ Все тесты пройдены!")
