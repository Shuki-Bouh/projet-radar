import simulateur as sml
import numpy as np
import matplotlib.pyplot as plt
import signalprocessing as tds
import parametre as prm

if __name__ == "__main__":

    simulation = sml.Simulation(nrx=2)

    ltargets = [[1, 0, 0, 0], [2, np.pi/4, 0, 0]]
    simulation.addTrg(ltargets)
    print(simulation.board.rx)
    print(simulation.board.tx)
    print(simulation.targets)

    print(simulation.process())

    xif = simulation.simulation()

    tds.range_calculation(xif)

