
tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g','h', '1', '2', '3', '4', '5', '6', '7', '8', 'x', '1/2-1/2', '1-0', '0-1', '.']

m = 0

dictionary = {}

for token in tokens:
    dictionary[token] = m
    m+=1

text = ['1.c4 e5 2.Nc3 Nc6 3.e3 Nf6 4.a3 Be7 5.Nf3 O-O 6.Be2 d6 7.d4 exd4 8.Nxd4 Nxd4 9.Qxd4 Be6 10.Nd5 c5 11.Nxe7+ Qxe7 12.Qh4 d5 13.cxd5 Bxd5 14.f3 Qe6 15.O-O Nd7 16.Bd2 f5 17.Rac1 Rac8 18.Rfe1 Ne5 19.Bc3 Ng6 20.Qf2 Bb3 21.Bf1 a6 22.Qg3 Qe7 23.Bd3 Rc6 24.Qf2 Re6 25.g3 h5 26.h4 b5 27.f4 Bd5 28.Be2 Kf7 29.Bxh5 Rh8 30.Rcd1 Bb3 31.Bf3 Bxd1 32.Rxd1 Rd8 33.Rd5 Kg8 34.Rxf5 Rxe3 35.Bd5+ Rxd5 36.Rxd5 Qe4 37.Rd1 Re2 38.Re1 Nxf4 39.Rxe2 Nxe2+ 40.Kh2 Nd4 41.Qf4 Qxf4 42.gxf4 Nc6 43.Kg3 b4 44.Bd2 a5 45.Kf3 c4 46.Ke4 1/2-1/2 ','1.Nf3 c5 2.g3 g6 3.Bg2 Bg7 4.c4 Nc6 5.Nc3 e6 6.d3 Nge7 7.Bd2 O-O 8.a3 b6 9.Rb1 Bb7 10.b4 cxb4 11.axb4 Ne5 12.Nh4 Bxg2 13.Nxg2 d5 14.f4 N5c6 15.b5 Na5 16.cxd5 Nxd5 17.Nxd5 Qxd5 18.O-O Nb7 19.Qb3 Qd7 20.Rfc1 Rfc8 21.Qa4 Nc5 22.Qa3 Bf8 23.Qa1 Ne4 24.Be3 Bg7 25.Rxc8+ Rxc8 26.Qa6 Nc3 27.Re1 Nxb5 28.Qa4 Rc7 29.Kf2 Nc3 30.Qxd7 Rxd7 31.Nh4 Nd5 32.Nf3 Nxe3 33.Kxe3 Rc7 34. Rb1 Kf8 35.d4 Ke8 36.Kd3 Kd8 37.e4 Kc8 38.Rb3 Kb7 39.Ne5 f6 40.Nc4 Rd7 41. Na5+ Ka8 42.Nc4 Rd8 43.h3 Bf8 44.Ke3 Rb8 45.e5 fxe5 46.fxe5 b5 47.Na5 b4 48.Ke4 Rb5 49.Rf3 Rxa5 50.Rxf8+ Kb7 51.Rf7+ Kc8 52.Rf8+ Kc7 53.Rf7+ Kc6 54.Re7 b3 55.Rxe6+ Kd7 56.Rd6+ Kc7 57.Rf6 b2 58.Rf7+ Kb6 59.Rf6+ Kb5 60. Rf1 Ra1 0-1 ']

output = []

for line in text:
    x = []
    if '1/2-1/2' in line[-10:]:
        x.append(dictionary['1/2-1/2'])
        for char in line[:-10]:
            if char in dictionary.keys():
                x.append(dictionary[char])
    if '1-0' in line[-6:]:
        x.append(dictionary['1-0'])
        for char in line[:-6]:
            if char in dictionary.keys():
                x.append(dictionary[char])
    if '0-1' in line[-6:]:
        x.append(dictionary['0-1'])
        for char in line[:-6]:
            if char in dictionary.keys():
                x.append(dictionary[char])
    output.append(x)

print(output)
print(dictionary)