"""Numerical tests without executing notebook-style file IO and plotting."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy.interpolate import interp1d


def load_numeric_functions():
    path = Path(__file__).resolve().parents[1] / 'solvers' / 'Duan2006.py'
    tree = ast.parse(path.read_text())
    names = {'pressure_data', 'temperature_data', 'saturation_pressure_interp'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    scope = {'np': np, 'interp1d': interp1d}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class PotassiumContract(unittest.TestCase):
    def test_equal_nacl_and_kcl_for_this_parameterization(self):
        scope = load_numeric_functions()  # deliberately no global m_k
        calculate = scope['calculate_m_co2']
        for temperature in [323.15, 373.15, 423.15]:
            for pressure in [25., 150., 500.]:
                for molality in [0., .5, 1., 2., 4.]:
                    with self.subTest(T=temperature, P=pressure, m=molality):
                        sodium = calculate(pressure, temperature, 0, molality, molality, 0, 0, 0)
                        potassium = calculate(pressure, temperature, 0, molality, 0, molality, 0, 0)
                        np.testing.assert_allclose(potassium, sodium, rtol=1e-13)

    def test_potassium_increment_matches_salt_term(self):
        scope = load_numeric_functions()
        scope['m_k'] = 0  # Existing example state must not override m_K.
        calculate = scope['calculate_m_co2']
        pressure, temperature, chloride = 150., 373.15, 2.
        reference = calculate(pressure, temperature, 0, chloride, 0, 0, 0, 0)
        coefficient = (-2 * scope['calculate_lambda_co2_na'](pressure, temperature)
                       - chloride * scope['calculate_xi_co2_na_cl'](pressure, temperature))
        for potassium in [.5, 1., 2., 4.]:
            with self.subTest(m_K=potassium):
                actual = calculate(pressure, temperature, 0, chloride, 0, potassium, 0, 0)
                np.testing.assert_allclose(actual / reference,
                                           np.exp(coefficient * potassium), rtol=1e-13)

    def test_global_potassium_does_not_change_result(self):
        scope = load_numeric_functions()
        calculate = scope['calculate_m_co2']
        expected = calculate(150, 373.15, 0, 2, 0, 2, 0, 0)
        for irrelevant_global in [0, 1, 100]:
            scope['m_k'] = irrelevant_global
            self.assertEqual(calculate(150, 373.15, 0, 2, 0, 2, 0, 0), expected)

if __name__ == '__main__':
    unittest.main()
