import math

def round_down(x):
    return int(math.floor(x / 10.0)) * 10


print(round_down(1985))