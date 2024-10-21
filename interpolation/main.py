import matplotlib.pyplot as plt
import numpy as np
import math
import tabulate as tb
import itertools
import operator


class interpol:
    def __init__(self, f, start, end, num_nodes):
        self.f = f
        self.nodes = list(np.linspace(start, end, num_nodes))
        self.interp_nodes = None
        self.degree = None

    # prints table with function value in given number of points
    def print_nodes(self):
        self.print_table(self.nodes)

    # prints table with sorted points used for interpolation polynom
    def print_interp_nodes(self, x):
        points = self.get_knn(x, self.degree + 1)
        self.print_table(points)

    # prints table with function value in given points
    def print_table(self, points):
        vals = [self.f(t) for t in points]
        table = (['x'] + points, ['y'] + vals)
        print(tb.tabulate(table, floatfmt=".2f", tablefmt="fancy_grid", numalign="center"))

    def draw(self):
        f_vals = [self.f(t) for t in self.nodes]
        plt.plot(self.nodes, f_vals, 'o-', linewidth=1.5, label="ln(1 + x)")

        L_vals = [self.L_poly(t) for t in self.nodes]
        plt.plot(self.nodes, L_vals, 'o-', linewidth=1.5, label="L(x)")

        plt.axhline(0, color='gray', linewidth=1.5)
        plt.axvline(0, color='gray', linewidth=1.5)
        plt.legend()
        plt.show(block=False)

    def get_knn(self, x: float, k: int):
        dist_and_points = [(t, abs(x - t)) for t in self.nodes]
        dist_and_points.sort(key=lambda x: x[1])
        return [tup[0] for tup in dist_and_points[:k]]

    def interpolate(self, x: float) -> float:
        self.interp_nodes = self.get_knn(x, self.degree + 1)
        return self.L_poly(x)

    def mul(self, x, nodes_, exclude: int = None):
        nodes = nodes_.copy()
        if exclude is not None:
            nodes.pop(exclude)
        diffs = [x - t for t in nodes]
        res = 1
        if len(diffs) > 1:
            *_, res = itertools.accumulate(diffs, operator.mul)
        return 1 if len(diffs) == 0 else res

    def fund_poly(self, x, nodes, exclude):
        left = self.mul(x, nodes, exclude)
        right = self.mul(nodes[exclude], nodes, exclude)
        return left / right

    def L_poly(self, x):
        nodes = self.interp_nodes
        fund_vals = [self.f(nodes[i]) * self.fund_poly(x, nodes, i) for i in range(self.degree + 1)]
        *_, res = itertools.accumulate(fund_vals, operator.add)
        return res


def main():
    print("\n""Программа для алгебраического интерполирования функции ln(1 + x). Вариант №2\n")
    f = lambda x: math.log1p(x)
    while True:
        left = float(input("Введите левый конец отрезка: "))
        right = float(input("Введите правый конец отрезка: "))
        num_nodes = int(input("Введите M - число значений в таблице: "))
        while (degree := int(input(f"Введите N - степень интерполяционного многочлена (N < {num_nodes}): "))) >= num_nodes:
            print("Введено недопустимое значение N")

        inter = interpol(f, left, right, num_nodes)
        inter.print_nodes()

        while True:
            x = float(input("Введите точку интерполяции: "))

            while True:
                inter.degree = degree
                print("\n"f"Узлы интерполяции, ближайшие к точке интерполяции {x}:")
                inter.print_interp_nodes(x)

                val = inter.interpolate(x)

                print(f"Значение интерполяционного многочлена в точке x: {val}\n")
                print(f"Абсолютная погрешность: {abs(val - f(x))}\n")
                plt.close()
                inter.draw()
                print("\n""Доступные опции:\n"
                        "1 - Ввести новую степень многочлена\n"
                        "2 - Изменить параметры таблицы\n"
                        "3 - Ввести новую точку интерполяции\n"
                        "4 - Выйти\n")
                choice = int(input("Выберите опцию: "))

                if choice == 4:
                    return
                if choice == 1:
                    while (degree := int(input(f"Введите N - степень интерполяционного многочлена (N < {num_nodes}): "))) >= num_nodes:
                        print("Введено недопустимое значение N")
                if choice in (2, 3):
                    break

            if choice == 2:
                break

        print()


# params usage: cat params | python3 main.py
if __name__ == "__main__":
    main()

