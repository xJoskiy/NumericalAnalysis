from enum import unique
import tabulate as tb
import math
import scipy


class integral:
    def __init__(self, f, weight, left, right, nodes):
        self.f = f
        self.weight = weight
        self.left = left
        self.right = right
        self.nodes = nodes

    def calc_moments(self):
        res = []
        
        for k in range(len(self.nodes)):
            f = lambda x: self.weight(x) * x ** k
            moment, _ = scipy.integrate.quad(f, self.left, self.right)
            res.append(moment)
        return res

    def calc_coeffs(self):
        matrix = []
        for k in range(len(self.nodes)):
            f = lambda x: x ** k
            line = [f(node) for node in self.nodes]
            matrix.append(line)

        moments = self.calc_moments()
        res = list(scipy.linalg.solve(matrix, moments))
        return res

    def integrate(self):
        res = 0
        coeffs = self.calc_coeffs()
        
        for k in range(len(self.nodes)):
            res += coeffs[k] * self.f(self.nodes[k])
        
        return res



def main():
    print("Приближенное вычисление интегралов при помощи ИКФ. Вариант 2\n")
    print("Вычисление интеграла от функции p(x) * f(x), где p(x) = x ^ (1 / 4), f(x) = sin(x)\n")

    f = lambda x: math.sin(x)
    p = lambda x: x ** (1 / 4)
    phi = lambda x: f(x) * p(x)
    
    left = float(input("Введите левый конец отрезка: ") or 0)
    right = float(input("Введите правый конец отрезка: ") or 10)

    num_nodes = int(input("Введите количество узлов ИКФ: ") or 5)
    while True:
        nodes: List[float] = list(map(float, input(f"Введите {num_nodes} различных узлов ИКФ через пробел: ").split()))
        unique_nodes = list(set(nodes))
        if len(unique_nodes) != len(nodes) or len(nodes) != num_nodes:
            print("Неправильный ввод")
            continue
        else:
            break

    kf = integral(f, p, left, right, nodes)

    print("Моменты весовой фукнции:")
    moments = [f"{mom:.1f}" for mom in kf.calc_moments()]
    mom_tab = (["k"] + [k for k in range(len(nodes))], ["∫x^k * p(x)"] + moments)
    print(tb.tabulate(mom_tab, tablefmt="fancy_grid", numalign="center", stralign="center"))

    print()
    print("Коэффициенты A_k:")
    coeffs = [f"{mom:.1f}" for mom in kf.calc_coeffs()]
    coeff_tab = (["x_k"] + nodes, ["A_k"] + coeffs)
    print(tb.tabulate(coeff_tab, tablefmt="fancy_grid", numalign="center"))

    precise_val, _ = scipy.integrate.quad(phi, left, right)
    calc_val = kf.integrate()
    abs_error = abs(calc_val - precise_val)
    rel_error = abs_error / abs(precise_val)

    print(f"Точное значение интеграла: {precise_val:.12f}")
    print(f"Вычисленное значение интеграла: {calc_val}")
    print(f"Абсолютная погрешность: {abs_error}")
    print(f"Относительная погрешность: {rel_error}")
    print()
    
    poly = lambda x: 5 * x ** (num_nodes - 1) + 3
    weight = lambda _: 1
    kf = integral(poly, weight, left, right, nodes)
    poly_calc_val = kf.integrate()
    poly_precise_val, _ = scipy.integrate.quad(poly, left, right)

    poly_abs_error = abs(poly_calc_val - poly_precise_val)
    poly_rel_error = poly_abs_error / abs(poly_precise_val)
    print(f"Вычисление интграла от многочлена x ^ {num_nodes - 1} + 3, - степени {num_nodes - 1}")
    print(f"Точное значение интеграла от : {poly_precise_val}")
    print(f"Вычисленное значение интеграла {num_nodes - 1}: {poly_calc_val}")
    print(f"Абсолютная погрешность: {poly_abs_error}")
    print(f"Относительная погрешность: {poly_rel_error}")

# params usage: cat params | python3 main.py
if __name__ == "__main__":
    main()