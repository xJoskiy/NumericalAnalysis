from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
import math
# import time


class interval:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end
    
    def len(self) -> float:
        return self.end - self.start

    def advance(self, step) -> None:
        self.start += step
        self.end += step
        
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def split(self):
        mid = self.mid()
        return [interval(self.start, self.mid()), interval(mid, self.end)] 
    
    def __str__(self, precision: int = 3) -> str:
        return f"[{round(self.start, precision)}, {round(self.end, precision)}]"


class root:
    def __init__(self, val, abs_dif, it, method, sect_len=0):
        self.abs_dif = abs_dif
        self.iterations = it
        self.val = val
        self.method = method
        self.sect_len = sect_len

    def __str__(self):
        res = f"{self.method}:\n\n" \
                f"Приближённое решение x = {self.val}\n\n" \
                f"Число итераций метода: {self.iterations}\n\n" \
                f"Модуль невязки: {self.abs_dif}\n\n"
        res += f"Длина последнего промежутка: {self.sect_len}\n" if self.sect_len > 0 else ""
        return res



class root_finder:
    def __init__(self, f: Callable[[float], float], df, section: interval, steps_count: float, eps: float) -> None:
        self.f: Callable[[float], float] = f
        self.df: Callable[[float], float] = df
        self.section: interval = section
        self.step: float = section.len() / steps_count
        self.eps: float = eps
        
    def get_root_sections(self) -> List[interval]:
        res: List[interval] = []
        start = self.section.start
        end = self.section.end
        cur_sect = interval(start, start + self.step)

        while cur_sect.end <= end:
            val = self.calc_prod(cur_sect)
            if val < 0:
                res.append(interval(cur_sect.start, cur_sect.end))
            cur_sect.advance(self.step)

        return res
            
    def calc_prod(self, section: interval) -> float:
        return self.f(section.start) * self.f(section.end)
    
    def bisection(self, section: interval) -> List[root]:
        i: int = 0
        cur_section = section
        while cur_section.len() >= 2 * self.eps:
            left, right = cur_section.split()
            cur_section = left if self.calc_prod(left) < 0 else right
            i += 1

        root_val = cur_section.mid()
        abs_dif = abs(self.f(root_val))

        return root(root_val, abs_dif, i, "Метод бисекции", cur_section.len())
    

    def newton(self, section: interval) -> List[root]:
        x_cur = section.mid()
        i: int = 1
        while True:
            x_next = x_cur - self.f(x_cur) / self.df(x_cur)
            if abs(x_cur - x_next) <= self.eps:
                abs_dif = abs(self.f(x_next))
                return root(x_next, abs_dif, i, "Метод Ньютона")
            i += 1
            x_cur = x_next
    
    def newton_enhanced(self, section: interval):
        x_cur = section.mid()
        const = self.df(x_cur)
        i: int = 1
        while True:
            x_next = x_cur - self.f(x_cur) / const
            if abs(x_cur - x_next) <= self.eps:
                abs_dif = abs(self.f(x_next))
                return root(x_next, abs_dif, i, "Модифицированный метод Ньютона")
            i += 1
            x_cur = x_next
    
    def secant(self, section: interval):
        a = section.start
        b = section.end
        i: int = 1
        while True:
            a = a - (b - a) * self.f(a) / (self.f(b) - self.f(a))
            b = b - (a - b) * self.f(b) / (self.f(a) - self.f(b))
            if abs(a - b) <= self.eps:
                abs_dif = abs(self.f(b))
                return root(b, abs_dif, i, "Метод секущих")
            i += 1

    def draw(self) -> None:
        start = self.section.start
        end =  self.section.end
        x = np.linspace(start, end, 100)
        y = [self.f(t) for t in x]
        plt.plot(x, y, color='red')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.grid()
        plt.show(block=False)
        plt.pause(0.001)


def main():
    # [A, B] = [-5 ,10]
    f = lambda x: pow(2, -x) - math.sin(x)
    df = lambda x: -pow(2, -x) * math.log(2) - math.cos(x)
    print("Программа для нахождения корней трансцендетного уравнения: 2^(-x) - sin(x) = 0\n")
    print("Доступные методы: \n"\
        "Метод половинного деления (бисекции)\n"
        "Метод Ньютона\n"\
        "Усовершенствованный метод Ньютона\n"
        "Метод секущих\n")

    while True:
        left = float(input("Введите левый конец отрезка: "))
        right = float(input("Введите правый конец отрезка: "))
        eps = float(input("Введите точность (например 1e-6): "))
        precision = 0

        while True:
            steps = int(input("Введите количество разбиений отрезка: "))
            precision = len(str(steps)) - 1
            rt = root_finder(f, df, interval(left, right), steps, eps)
            root_sections = rt.get_root_sections()

            print("Для заданной функции были найдены следующие отрезки перемены знака:\n")
            [print(section.__str__(precision)) for section in root_sections]
            doContinue = int(input("\n""Хотите изменить количество разбиений отрезка: 1 - Да, 0 - Нет: "))
            if doContinue == 0:
                break

        rt.draw()
        print()
        [print(f"{i + 1}.", section.__str__(precision)) for i, section in enumerate(rt.get_root_sections())]
        while (sect_num := int(input("\n""Укажите отрезок для уточнения корней (0 - выход): "))) != 0:
            cur_sect = root_sections[sect_num - 1]
            print(f"------------------------------------------------------\n")
            print("Точность: ", eps, end="\n\n")

            print(rt.bisection(cur_sect))
            print(f"------------------------------------------------------\n")
            # time.sleep(1.5)
            print(rt.newton(cur_sect))
            print(f"------------------------------------------------------\n")
            # time.sleep(1.5)
            print(rt.newton_enhanced(cur_sect))
            print(f"------------------------------------------------------\n")
            # time.sleep(1.5)
            print(rt.secant(cur_sect))
            print(f"------------------------------------------------------\n")
            [print(f"{i + 1}. {section.__str__(precision)}") for i, section in enumerate(rt.get_root_sections())]
        
        doContinue = int(input("Хотите повторить ввод: 1 - Да, 0 - Нет: "))
        if doContinue == 0:
            break
        
    
# params usage: cat params | python3 main.py
if __name__ == "__main__":
    main()
