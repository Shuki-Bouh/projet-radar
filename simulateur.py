import numpy as np
import matplotlib.pyplot as plt
from parametre import *

class Tx:

    def __init__(self, x, y, virtual=False):
        self.x = x
        self.y = y
        self.virtual = virtual


class Rx:

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Target:
    def __init__(self, x0, y0, vx, vy):
        self.x = x0
        self.y = y0
        self.vx = vx
        self.vy = vy


class Board:

    def __init__(self):
        self.rx = []
        self.tx = []
        self.channels = []

    def add(self, ltx, lrx):  #ltx et lrx de la forme ltx = [[x1, y1], [x2, y2], ...]
        for elt in ltx:
            self.tx.append(Tx(elt[0], elt[1]))

        for elt in lrx:
            self.rx.append(Rx(elt[0], elt[1]))


class Simulation:

    def __init__(self, T, Tc, Te, fc, B, Mimo=False):  # attention on veut Te divisible par Tc et Tc divisible par Te
        self.board = Board()
        self.targets = []
        self.Tc = Tc
        self.Te = Te
        self.B = B
        self.fc = fc
        self.S = B / Tc
        self.c = 3 * 10 ** 8
        self.T = T
        self.Mimo = Mimo
        if Mimo:
            # self.board.add()
            pass
        self.nchan = len(self.board.rx)

    def addTrg(self, ltargets):  #ltargets de la forme ltargets = [[x1, y1, vx1, vy1], [x2, y2, vx2, vy2], ...]
        for elt in ltargets:
            self.targets.append(Target(elt[0], elt[1], elt[2], elt[3]))

    def move(self):
        for trg in self.targets:
            trg.x += trg.vx*self.Tc
            trg.y += trg.vy*self.Tc

    def process(self):  # rempli self.τs avec tout les τ associés
        self.nchan = len(self.board.rx)
        τs = np.zeros((self.nchan, len(self.board.tx) * len(self.targets)))
        i = 0
        for rx in self.board.rx:
            j = 0
            for tx in self.board.tx:
                for trg in self.targets:
                    τ = (np.sqrt((rx.x - trg.x) ** 2 + (rx.y - trg.y) ** 2) + np.sqrt(
                        (trg.x - tx.x) ** 2 + (trg.y - tx.y) ** 2)) / self.c
                    τs[i, j] = τ
                    j += 1
            i += 1
        return τs

    def record(self):
        ns = int(self.Tc / self.Te)  # nombre de samples par chirp, dimension x
        nc = int(self.T / self.Tc)  # nombre de de chirps, dimension y
        nrx = self.nchan  # nombre de channels, dimension z
        xif = np.zeros((ns, nc, nrx), dtype=complex)
        for j in range(nc):
            t = np.linspace(j*self.Tc, (j+1)*self.Tc, ns)
            τs = self.process()
            self.move()
            for k in range(nrx):
                for τ in τs[k]:
                    f = τ * self.S
                    phi = 2 * np.pi * self.fc * τ
                    arg = 2 * np.pi * f * t + phi
                    xif[0:ns+1, j, k] += np.exp(arg*1j)

        return xif

    def simulation(self):
        with open('record.txt', 'w') as file:
            xif = self.record()
            ns = len(xif)
            nc = len(xif[0])
            nchan = len(xif[0, 0])
            for k in range(nchan):
                for j in range(nc):
                    for i in range(ns):
                        file.write(f"{xif[i, j, k]} ")
                    file.write('\n')
                file.write('\n')

        return xif


