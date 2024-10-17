import matplotlib.pyplot as plt
import numpy as np
import math
import tabulate as tb
import itertools
import operator


class interpol:
    def __init__(self, f, start, end, num_nodes, degree):
        self.f = f
        self.nodes = list(np.linspace(start, end, num_nodes))
        self.degree = degree
        self.interp_nodes = None

    # prints table with function value in given number of points
    def print_nodes(self):
        self.print_table(self.nodes)

    # prints table with sorted points used for interpolation polynom
    def print_interp_nodes(self, x: float):
        points = self.get_knn(x, self.degree)
        points.sort()
        self.print_table(points)
    
    # prints table with function value in given points
    def print_table(self, points):
        vals = [self.f(t) for t in points]
        table = (['x'] + points, ['y'] + vals)
        print(tb.tabulate(table, floatfmt=".2f"))

    def draw(self):
        f_vals = [self.f(t) for t in self.nodes]
        plt.plot(self.nodes, f_vals, 'o-', linewidth=1.5, label="ln(1 + x)")

        L_vals = [self.interpolate(t) for t in self.nodes]
        plt.plot(self.nodes, L_vals, 'o-', linewidth=1.5, label="L(x)")

        plt.axhline(0, color='gray', linewidth=1.5)
        plt.axvline(0, color='gray', linewidth=1.5)
        plt.legend()
        plt.show(block=False)
    
    def get_knn(self, x: float, k: int):
        dist_and_points = [(float(t), abs(x - float(t))) for t in self.nodes]
        dist_and_points.sort(key=lambda x: x[1])
        return [tup[0] for tup in dist_and_points[:k]]

    def interpolate(self, x: float) -> float:
        if self.interp_nodes is None:
            self.interp_nodes = self.get_knn(x, self.degree)
        return self.L_poly(x)

    def mul(self, x, nodes_, exclude: int = None):
        nodes = nodes_.copy()
        if exclude is not None:
            nodes.pop(exclude)
        diffs = [x - t for t in nodes]
        *_, prod = itertools.accumulate(diffs, operator.mul)
        return prod

    def fund_poly(self, x, nodes, exclude):
        left = self.mul(x, nodes, exclude)
        right = self.mul(nodes[exclude], nodes, exclude)
        if right == 0:
            print(exclude)
        return left / right
        
        
    def L_poly(self, x):
        nodes = self.interp_nodes
        fund_vals = [self.f(nodes[i]) * self.fund_poly(x, nodes, i) for i in range(self.degree)]
        *_, res = itertools.accumulate(fund_vals, operator.add)
        return res



def main():
    print("\n""Программа для алгебраического интерполирования функции ln(1 + x), вариант №2\n")
    f = lambda x: math.log1p(x)
    while True:
        left = float(input("Введите левый конец отрезка: "))
        right = float(input("Введите правый конец отрезка: "))
        num_nodes = int(input("Введите число значений M в таблице: "))
        while (degree := int(input("Введите степень интерполяционного многочлена N < M: "))) > num_nodes:
            print("Введено недопустимое значение N")

        inter = interpol(f, left, right, num_nodes, degree)
        inter.print_nodes()
        start = True

        while True:
            x = input("Введите точку интерполяции или \"exit\" - для выхода: ")
            print()
            if x == "exit":
                plt.close()
                break
            x = float(x)
            
            if start:
                print("\n""Узлы интерполяции, ближайшие к точке интерполяции:")
                inter.print_interp_nodes(x)
                start = False

            val = inter.interpolate(x)
            print(f"Значение интерполяционного многочлена в точке x: {val}\n")
            print(f"Абсолютная погрешность: {abs(val - f(x))}\n")
            inter.draw()

        if int(input("Хотите повторить ввод? 1 - Да, 0 - Нет: ")) == 0:
            break
        print()


# params usage: cat params | python3 main.py
if __name__ == "__main__":
    main()

