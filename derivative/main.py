from typing import Callable, List
import matplotlib.pyplot as plt
import tabulate as tb
import math


class function:
    def __init__(self, f: Callable[[float], float], f_str):
        self.f_str = f_str
        self.f: Callable[[float], float] = f
        self.df: Callable[[float], float] = None
        self.d2f: Callable[[float], float] = None

    def __call__(self, x) -> float:
        return self.f(x)


class diff:
    def __init__(self, f: Callable[[float], float], start, step, num_vals):
        self.f: Callable[[float], float] = f
        self.start = start
        self.step = step
        self.num_vals = num_vals
        self.nodes = [self.start + self.step * i for i in range(self.num_vals)]
        self.f_vals = [self.f(t) for t in self.nodes]

    def print_f(self):
        table = (['x'] + self.nodes, ['y'] + self.f_vals)
        print(tb.tabulate(table, floatfmt=".3f", stralign='center', tablefmt="fancy_grid"))

    def print_res_table(self):
        table_labels = ("№", "x", "f(x)", "f'(x) ± O(h^2)", "Погрешность O(h^2)",
                    "f'(x) ± O(h^4)", "Погрешность O(h^4)",
                    "f\"(x)", "Погрешность")

        table = [[i + 1, self.nodes[i], self.f_vals[i], 
                  *self.first_deriv_h2(i, self.step, self.nodes, self.f_vals),
                  *self.first_deriv_h4(i, self.step, self.nodes, self.f_vals),
                  *self.second_deriv(i, self.step, self.nodes, self.f_vals)] for i in range(len(self.nodes))]

        print(tb.tabulate(table, headers=table_labels, floatfmt=("e",".3f"), tablefmt="fancy_grid", numalign='center'))

    def first_deriv_h2(self, index: int, step, nodes, vals):
        if index == 0:
            df = (-3 * vals[index] + 4 * vals[index + 1] - vals[index + 2]) / (2 * step)
        elif index == len(nodes) - 1:
            df = (3 * vals[index] - 4 * vals[index - 1] + vals[index - 2]) / (2 * step)
        else:
            df = (vals[index + 1] - vals[index - 1]) / (2 * step)

        return df, abs(df - self.f.df(nodes[index]))

    def first_deriv_h4(self, index: int, step, nodes, vals):
        if index == 0:
            df = (-25 * vals[0] + 48 * vals[1] - 36 * vals[2]
                  + 16 * vals[3] - 3 * vals[4]) / (12 * step)

        elif index == 1:
            df = (-3 * vals[0] - 10 * vals[1] + 18 * vals[2]
                - 6 * vals[3] + vals[4]) / (12 * step)

        elif index == len(nodes) - 2:
            df = (3 * vals[-1] + 10 * vals[-2] - 18 * vals[-3]
                + 6 * vals[-4] - vals[-5]) / (12 * step)

        elif index == len(nodes) - 1:
            df = (25 * vals[-1] - 48 * vals[-2] + 36 * vals[-3]
                - 16 * vals[-4] + 3 * vals[-5]) / (12 * step)

        else:
            df = (vals[index - 2] - 8 * vals[index - 1] + 8 * vals[index + 1]
                - vals[index + 2]) / (12 * step)

        return df, abs(df - self.f.df(nodes[index]))

    def second_deriv(self, index, step, nodes, vals):
        if index == 0:
            d2f = (2 * vals[0] - 5 * vals[1] + 4 * vals[2] - vals[3]) / (step ** 2)
        elif index == len(nodes) - 1:
            d2f = (2 * vals[-1] - 5 * vals[-2] + 4 * vals[-3] - vals[-4]) / (step ** 2)
        else:
            d2f = (vals[index + 1] - 2 * vals[index] + vals[index - 1]) / (step ** 2)
        return d2f, abs(d2f - self.f.d2f(nodes[index]))

    def runge(self, index):
        cur_x = self.nodes[index]
        doubled_nodes = [self.start + self.step / 2 * i for i in range(self.num_vals * 2 - 1)]
        doubled_vals = [self.f(t) for t in doubled_nodes]

        J1_df, err_J1_df = self.first_deriv_h2(index, self.step, self.nodes, self.f_vals)
        J2_df, err_J2_df = self.first_deriv_h2(index * 2, self.step / 2, doubled_nodes, doubled_vals)
        J_df = (4 * J2_df - J1_df) / 3

        line_df = [[cur_x, self.f.df(cur_x), J1_df,
                    err_J1_df, J2_df, err_J2_df, J_df,
                    abs(J_df - self.f.df(cur_x))]]

        labels_df = ["x", "f'(x)", "J(h)", "Погрешность", "J(h/2)", "Погрешность", "J", "Погрешность"]

        print(f"Уточнённые значения первой производной в точке x = {self.nodes[index]}:")
        print(tb.tabulate(line_df, headers=labels_df, numalign="center", floatfmt=(".3f", "e"), tablefmt="fancy_grid"))

        J1_d2f, err_J1_d2f = self.second_deriv(index, self.step, self.nodes, self.f_vals)
        J2_d2f, err_J2_d2f = self.second_deriv(index * 2, self.step / 2, doubled_nodes, doubled_vals)
        J_d2f = (4 * J2_d2f - J1_d2f) / 3
        
        labels_d2f = ["x", "f\"(x)", "J(h)", "Погрешность", "J(h/2)", "Погрешность", "J", "Погрешность"]

        line_d2f = [[cur_x, self.f.d2f(cur_x), J1_d2f,
                    err_J1_d2f, J2_d2f, err_J2_d2f, J_d2f,
                    abs(J_d2f - self.f.d2f(cur_x))]]

        print("\n"f"Уточнённые значения второй производной в точке x = {cur_x}:")
        print(tb.tabulate(line_d2f, headers=labels_d2f, numalign="center", floatfmt=(".3f", "e"), tablefmt="fancy_grid"))

    def draw(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_size_inches(w=8, h=7)

        fig.suptitle('Точные и вычисленные значения соответственно')
        self.draw_exact(ax1)
        self.draw_calc(ax2)

    def draw_exact(self, plot):
        plot.set_title("Точные значения")

        plot.plot(self.nodes, self.f_vals, 'o-', linewidth=1.5, label=f"f(x) = {self.f.f_str}")

        df_vals = [self.f.df(x) for x in self.nodes]
        plot.plot(self.nodes, df_vals, 'o-', linewidth=1.5, label=f"f'(x) = {self.f.df.f_str}")

        d2f_vals = [self.f.d2f(x) for x in self.nodes]
        plot.plot(self.nodes, d2f_vals, 'o-', linewidth=1.5, label=f"f\"(x) = {self.f.d2f.f_str}")

        plot.axhline(0, color='gray', linewidth=1.5)
        plot.axvline(0, color='gray', linewidth=1.5)
        plot.legend()
        plt.show(block=False)

    def draw_calc(self, plot):
        plot.set_title("Вычисленные значения")

        plot.plot(self.nodes, self.f_vals, 'o-', linewidth=1.5, label=f"f(x) = {self.f.f_str}")

        df_vals = [self.first_deriv_h4(i, self.step, self.nodes, self.f_vals)[0] for i in range(len(self.nodes))]
        plot.plot(self.nodes, df_vals, 'o-', linewidth=1.5, label=f"f'(x) = {self.f.df.f_str}")

        d2f_vals = [self.second_deriv(i, self.step, self.nodes, self.f_vals)[0] for i in range(len(self.nodes))]
        plot.plot(self.nodes, d2f_vals, 'o-', linewidth=1.5, label=f"f\"(x) = {self.f.d2f.f_str}")

        plot.axhline(0, color='gray', linewidth=1.5)
        plot.axvline(0, color='gray', linewidth=1.5)
        plot.legend()
        plt.show(block=False)


def main():
    f1 = function(lambda x: math.log1p(x), "ln(1 + x)")
    f1.df = function(lambda x: 1 / (x + 1), "1 / (x + 1)")
    f1.d2f = function(lambda x: -1 / ((x + 1) ** 2), "-1 / (x + 1)^2", )

    f2 = function(lambda x: math.exp(4.5 * x), "e^4.5x")
    f2.df = function(lambda x: 4.5 * math.exp(4.5 * x), "4.5 * e^4.5x")
    f2.d2f = function(lambda x: 20.25 * math.exp(4.5 * x), "20.25 * e^4.5x")

    print("Программа для нахождения производных таблично-заданной функции по формулам численного дифференцирования. Вариант №2")

    choice = 0
    while True:
        print(f"\n1: {f1.f_str}\n2: {f2.f_str}\n")
        func_num = int(input("Выберите функцию для поиска производной: "))

        while True:
            num_vals = int(input("Введите m - количество значений в таблице (m >= 5): "))
            start = float(input("Введите начальное значение x: "))
            step = float(input("Введите шаг h: "))
            funcs = [f1, f2]

            prog = diff(funcs[func_num - 1], start, step, num_vals)
            print("\n"f"Таблица значений функции {funcs[func_num - 1].f_str}:\n")
            prog.print_f()
            print("\n"f"Поиск производных функции {funcs[func_num - 1].f_str}:\n")
            prog.print_res_table()
            plt.close()
            prog.draw()

            if choice == 3:
                node_num = int(input("\n""Выберите номер узла, для уточнения производных: "))
                prog.runge(node_num - 1)

            print("\n""Опции на выбор:\n"
                "1 - Выбрать другие функцию и параметры таблицы\n"
                "2 - Изменить параметры таблицы\n"
                "3 - Уточнить производную по Рунге-Ромберту\n"
                "4 - Выйти\n")
            choice = int(input("Выберите опцию: "))

            if choice != 2:
                break

        if choice == 4:
            break

# params usage: cat params | python3 main.py
if __name__ == '__main__':
    main()
