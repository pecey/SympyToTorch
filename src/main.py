import sympy
import torch as T


class SympyToTorch():
    def __init__(self, sympy_parse_tree, device = "cpu"):
        self.sympy_parse_tree = sympy_parse_tree
        self.device = device

    def check_op(self, op):
        if op.is_Add:
            return T.add, True
        if op.is_Mul:
            return T.mul, True
        if op.is_Pow:
            return T.pow, True
        if op == sympy.sin:
            return T.sin, False
        if op == sympy.cos:
            return T.cos, False

    def parse(self):
        return self.__parse(self.sympy_parse_tree)

    def __parse(self, parse_tree):
        if not parse_tree.args:
            return parse_tree
        parsed_args = [self.__parse(arg) for arg in parse_tree.args]
        parsed_fn, is_binary_fn = self.check_op(parse_tree.func)
        equation = []
        for arg in parsed_args:
            equation.append(arg)
            equation.append(parsed_fn)
        if is_binary_fn:
            equation.pop()
        return equation

    def _parse_values(self, value, vars):
        if isinstance(value, sympy.Symbol):
            return vars[value.name].to(self.device)
        return T.tensor(float(value)).to(self.device)

    def _convert_to_torch(self, expression):
        def torch_partial(vars):
            if not isinstance(expression, list):
                return self._parse_values(expression, vars)
            if len(expression) == 1:
                return self._parse_values(expression[0], vars)
            simplified_expression = [self._convert_to_torch(subexpression)(vars) if isinstance(subexpression, list) else subexpression for subexpression in expression]
            op = simplified_expression[1]
            values = [self._parse_values(simplified_expression[idx], vars) for idx in range(0, len(simplified_expression), 2)]
            if len(values) == 1:
                return op(values[0])
            accumulator = op(values[0], values[1])
            for idx in range(2, len(values)):
                accumulator = op(accumulator, values[idx])
            return accumulator
        return torch_partial

    def convert_to_torch(self, args):
        parsed_tree = self.parse()
        return self._convert_to_torch(parsed_tree)(args)
