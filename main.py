import simulateur as sml
import matplotlib.pyplot as plt
import signalprocessing as tds
import parametre as prm

if __name__ == "__main__":

    simulation = sml.Simulation()

    ltx = [[0, 0]]
    lrx = [[-1, 0]]
    simulation.board.add(ltx, lrx)

    ltargets = [[0, 5, 0, 0]]
    simulation.addTrg(ltargets)
    print(simulation.board.rx)
    print(simulation.board.tx)
    print(simulation.targets)

    xif = simulation.simulation()

    tds.range_calculation(xif)

