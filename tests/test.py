import pytest
import sympy
import numpy as np
import torch as T
from main import SympyToTorch


def next_state_transition(state_vars, action_var):
    x, y, z = state_vars
    action = action_var

    tmp = x ** 3
    tmp2 = y ** 2
    tmp3 = 2 * z * action

    return tmp + tmp2 * tmp3

def test_parse_sympy_tree():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    action = sympy.Symbol('action')

    next_state = next_state_transition([x, y, z], action)
    parser = SympyToTorch(next_state)
    parsed_value = parser.parse()
    assert (len(parsed_value) == 3)
    assert(parsed_value[0] == [x,T.pow,3])
    assert(parsed_value[1] == T.add)
    assert(parsed_value[2] == [2, T.mul, action, T.mul, z, T.mul, [y, T.pow, 2]])


def test_torch_tree_1():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    action = sympy.Symbol('action')

    next_state = next_state_transition([x, y, z], action)
    parser = SympyToTorch(next_state)
    values = {'x': T.tensor([1]), 'y': T.tensor([2]), 'z': T.tensor([3]), 'action': T.tensor([4])}
    # torch_tree = parser.convert_to_torch(values)

    partial = [x, T.pow, 2]
    torch_tree = parser._convert_to_torch(partial)(values)
    assert torch_tree == T.pow(values['x'], 2)

def test_torch_tree_2():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    action = sympy.Symbol('action')

    next_state = next_state_transition([x, y, z], action)
    parser = SympyToTorch(next_state)
    values = {'x': T.tensor([1]), 'y': T.tensor([2]), 'z': T.tensor([3]), 'action': T.tensor([4])}
    # torch_tree = parser.convert_to_torch(values)

    partial = [[x, T.pow, 2], T.add, y]
    torch_tree = parser._convert_to_torch(partial)(values)
    assert torch_tree == T.add(T.pow(values['x'], 2), values['y'])

def test_torch_tree_3():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    action = sympy.Symbol('action')

    next_state = next_state_transition([x, y, z], action)
    parser = SympyToTorch(next_state)
    values = {'x': T.tensor([1]), 'y': T.tensor([2]), 'z': T.tensor([3]), 'action': T.tensor([4])}
    # torch_tree = parser.convert_to_torch(values)
    partial = [2, T.mul, action, T.mul, z, T.mul, [y, T.pow, 2]]
    torch_tree = parser._convert_to_torch(partial)(values)
    assert torch_tree == T.mul(T.mul(values['z'], T.mul(2, values['action'])), T.pow(values['y'], 2))

def test_torch_tree_4():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    action = sympy.Symbol('action')
    device = "cuda" if T.cuda.is_available() else "cpu"
    print(device)
    next_state = next_state_transition([x, y, z], action)
    parser = SympyToTorch(next_state, device)
    values = {'x': T.tensor([1]), 'y': T.tensor([2]), 'z': T.tensor([3]), 'action': T.tensor([4])}
    torch_tree = parser.convert_to_torch(values)
    assert torch_tree == T.add(T.pow(values['x'], 3), T.mul(T.mul(T.mul(2, values['action']), values['z']), T.pow(values['y'], 2))).to(device)

def test_trig_1():
    x = sympy.Symbol('x')
    next_state_transition_fn = lambda x: sympy.sin(x)
    next_state = next_state_transition_fn(x)
    parser = SympyToTorch(next_state)
    values = {'x': T.tensor([1.0])}
    torch_tree = parser.convert_to_torch(values)
    assert torch_tree == T.sin(values['x'])

def test_trig_2():
    x = sympy.Symbol('x')
    next_state_transition_fn = lambda x: sympy.sin(x) * sympy.cos(x) * x
    next_state = next_state_transition_fn(x)
    parser = SympyToTorch(next_state)
    values = {'x': T.tensor([1.0])}
    torch_tree = parser.convert_to_torch(values)
    assert torch_tree == T.mul(T.mul(T.cos(values['x']), T.sin(values['x'])), values['x'])
