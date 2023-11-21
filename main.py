# import struct
#
# print(struct.unpack('f', b'\xdb\x0fI@'))

import simulateur as sml

if __name__ == "__main__":

    Tc = 10 ** -3
    Te = Tc / 1000
    T = 64 * Tc
    fc = 77 * 10 ** 9
    B = 4 * 10 ** 9
    simulation = sml.Simulation(T, Tc, Te, fc, B)

    ltx = [[0, 0]]
    lrx = [[-1, 0], [1, 0]]
    simulation.board.add(ltx, lrx)

    ltargets = [[0, 5, 0, 0]]
    simulation.addTrg(ltargets)
    print(simulation.board.rx)
    print(simulation.board.tx)
    print(simulation.targets)

    ts = simulation.process()
    print(ts)

    xif = simulation.simulation()
    print(xif)

