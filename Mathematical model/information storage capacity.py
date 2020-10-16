import math


def comb(n, m):
    c = math.factorial(n)//(math.factorial(n - m) * math.factorial(m))
    return c


def condition1(x):
    H = math.log((dendrites ** x), 2)
    return round(H)


def condition2(x):
    MSBs = round((x * MSB_synapses_rate)/2)
    boutons = x - MSBs
    H = math.log((comb(boutons, MSBs) * (dendrites ** boutons)), 2)
    return round(H)


def condition3(x):
    MSBs = round((x * MSB_synapses_rate)/2)
    boutons = x - MSBs
    H = math.log((comb(boutons, MSBs) * (dendrites ** (boutons - MSBs)) * ((dendrites + comb(dendrites, 2)) ** MSBs)), 2)
    return round(H)


def conditionA(x):
    add = round(x * plasticity)
    deltaH = math.log(comb(x, add) * ((dendrites + comb(dendrites, 2)) ** add), 2) - math.log((dendrites ** add), 2)
    return round(deltaH)


def conditionB(x):
    add = round(x * plasticity)
    deltaH = math.log(comb(x, add), 2)
    return round(deltaH)


dendrites, MSB_synapses_rate, plasticity = 9, 0.06, 0.1  # parameters

synapses = [100, 1000, 10000, 100000, 1000000]  # model size
cond1, cond2, cond3, condA, condB = [], [], [], [], []
for x in synapses:
    cond1.append(condition1(x))
    cond2.append(condition2(x))
    cond3.append(condition3(x))
    condA.append(conditionA(x))
    condB.append(conditionB(x))
print('H for condition1:', cond1)
print('H for condition2:', cond2)
print('H for condition3:', cond3)
print('ΔH for conditionA:', condA)
print('ΔH for conditionB:', condB)
