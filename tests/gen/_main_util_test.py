import pytest

from gen._main_util import _eval_circuit_param_expression


def test_eval_circuit_param_expression_allows_d():
    assert _eval_circuit_param_expression('5', d=3) == 5
    assert _eval_circuit_param_expression('d * 3', d=3) == 9
    assert _eval_circuit_param_expression("{'x': d + 1}", d=3) == {'x': 4}


def test_eval_circuit_param_expression_blocks_builtins():
    with pytest.raises((NameError, TypeError)):
        _eval_circuit_param_expression("__import__('os').getcwd()", d=3)
    with pytest.raises((NameError, TypeError)):
        _eval_circuit_param_expression("open('/etc/hostname')", d=3)
