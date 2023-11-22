import numpy as np
import parametre as prm
class Tx:

    def __init__(self, x, y, activated=False):
        self.x = x
        self.y = y
        self.activated = activated


class Rx:

    def __init__(self, x, y, activated=False):
        self.x = x
        self.y = y
        self.activated = activated


class Target:
    def __init__(self, r, theta, phi, vr):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.vr = vr


class Board:

    def __init__(self, nrx=1, Mimo=False):
        self.rx = [Rx(0,0,True), Rx(prm.λ,0), Rx(3*prm.λ/2,0), Rx(2*prm.λ,0)]
        self.tx = [Tx(0,0,True), Tx(2*prm.λ,0)]
        self.nrx = nrx
        self.Mimo = Mimo
        self.nchan = 1
        self.ntx = 1
        if self.nrx == 2:
            self.rx[1].activated = True
            self.nchan = 2
        elif self.nrx == 4:
            for elt in self.rx:
                elt.activated = True
            self.nchan = 4
        if self.Mimo :
            self.tx[1].activated = True
            self.nchan = len(self.rx) * len(self.tx)
            self.nchan = len(self.tx)

class Simulation:

    def __init__(self, nrx=1, Mimo=False):  # attention on veut Te divisible par Tc et Tc divisible par Te
        self.board = Board(nrx, Mimo)
        self.targets = []
        self.Tc = prm.Tc
        self.Te = 1/prm.fs
        self.B = prm.B
        self.fc = prm.fc
        self.S = prm.B / prm.Tc
        self.c = prm.c
        self.T = prm.Tc * prm.Nc
        self.λ = prm.λ

    def addTrg(self, ltargets):  #ltargets de la forme ltargets = [[r, theta, phi, vr], [...], ...]
        for elt in ltargets:
            self.targets.append(Target(elt[0], elt[1], elt[2], elt[3]))

    def move(self):
        for trg in self.targets:
            trg.r += trg.vr*self.Tc

    def process(self):  # rempli self.τs avec tout les τ associés
        τs = np.zeros((self.board.nchan, self.board.ntx * len(self.targets)))
        print(np.shape(τs))
        d = self.λ/2
        i = 0
        for rx in self.board.rx:
            if rx.activated:
                j, k = 0, 0
                for tx in self.board.tx:
                    if tx.activated:
                        for trg in self.targets:
                            τ = (2*trg.r + i*d*np.sin(trg.theta) + k*4*d*np.sin(trg.theta)) / self.c
                            print(τ)
                            τs[i, j] = τ
                            j += 1
                        k += 1
                i += 1
        return τs

    def record(self):
        ns = int(self.Tc / self.Te)  # nombre de samples par chirp, dimension x
        nc = int(self.T / self.Tc)  # nombre de de chirps, dimension y
        nrx = self.board.nchan  # nombre de channels, dimension z
        xif = np.zeros((ns, nc, nrx), dtype=complex)
        for j in range(nc):
            t = np.linspace(j*self.Tc, (j+1)*self.Tc, ns)
            τs = self.process()
            self.move()
            for k in range(nrx):
                for τ in τs[k]:
                    # print(τ)
                    f = τ * self.S
                    phi = 2 * np.pi * self.fc * τ
                    arg = 2 * np.pi * f * t + phi
                    xif[0:ns+1, j, k] += np.exp(arg*1j)

        return xif

    def simulation(self):
        with open('record.txt', 'w') as file:
            xif = self.record()
            ns = prm.Ns
            nc = prm.Nc
            nchan = self.board.nchan
            for k in range(nchan):
                for j in range(nc):
                    for i in range(ns):
                        file.write(f"{xif[i, j, k]} ")
                    file.write('\n')
                file.write('\n')

        return xif
