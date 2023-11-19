import numpy as np
import matplotlib.pyplot as plt


class Tx:

    def __init__(self, x, y, T, Tc, B, fc):
        self.x = x
        self.y = y
        self.T = T
        self.Tc = Tc
        self.B = B
        self.S = B / Tc
        self.fc = fc

    def transmit(self):
        pass


class Rx:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def receave(self):
        pass


class Target:
    def __init__(self, x0, y0, vx, vy):
        self.x = x0
        self.y = y0
        self.vx = vx
        self.vy = vy

    def reflect(self):
        pass

    def move(self):
        pass


class Board:

    def __init__(self):
        self.rx = []
        self.tx = []
        self.channels = []

    def add(self, ltx, lrx, T, Tc, B):
        for elt in ltx:
            self.rx.append(Tx(elt[0], elt[1], T, Tc, B))

        for elt in lrx:
            self.rx.append(Rx(elt[0], elt[1]))
        pass


class Simulation:

    def __init__(self, T, Te, Tc, fc, B):
        self.board = Board
        self.targets = []
        self.Tc = Tc
        self.Te = Te
        self.B = B
        self.fc = fc
        self.S = B / Tc
        self.c = 3 * 10 ** 8
        self.T = T
        self.cnt = 0
        self.τs = np.zeros((len(self.board.rx), len(self.board.tx) * len(self.targets)))

    def add(self, ltargets):
        for elt in ltargets:
            self.targets.append(Target(elt[0], elt[1], elt[2]))

    def move(self):
        for trg in self.targets:
            trg.x += trg.vx*self.Tc
            trg.y += trg.vy*self.Tc


    def process(self):
        i = 0
        for rx in self.board.rx:
            j = 0
            for tx in self.board.tx:
                for trg in self.targets:
                    τ = (np.sqrt((rx.x - trg.x) ** 2 + (rx.y - trg.y) ** 2) + np.sqrt(
                        (trg.x - tx.x) ** 2 + (trg.y - tx.y) ** 2)) / self.c
                    self.τs[i, j] = τ
                    j += 1
            i += 1

    def record(self):
        nrx = len(self.board.rx)
        n = self.Tc / self.Te
        xif = np.zeros((nrx + 1, n))
        t = np.arange(self.cnt*self.Tc, self.cnt*self.Tc + self.Tc + self.Te, self.Te)
        xif[0, 0:n] = t
        for i in range(1, nrx + 1):
            for τ in self.τs[i]:
                f = τ * self.S
                phi = 2 * np.pi * np.fc * τ
                xif[i, 0:n] += np.exp(2 * np.pi * f * t + phi)

        return xif

    def clearτs(self):
        self.self.τs = np.zeros((len(self.board.rx), len(self.board.tx) * len(self.targets)))

    def simulation(self):
        with open('record.txt', 'w') as file:
            for k in range(self.T):
                self.process()
                xif = self.record()
                self.clearτs()
                self.move()
                self.cnt +=1
                print(xif)
                # for j in range(len(xif[0])):
                #     for i in range(len(xif)):
                #         file.write(xif[i,j])
                #     file.write('\n')

        return


