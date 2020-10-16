import math


def comb(n, m):
    return math.factorial(n)//(math.factorial(n - m) * math.factorial(m))


def coeff(t, n, m, p):
    if p >= t:
        return comb(n, t) * comb(m, p - t)
    else:
        return 0


'''
SIMPLE: simple synapses (1-to-1 synapses)
COMPLEX: complex synapses (1-to-2 synapses)
'''


COMPLEX_start_ratio, COMPLEX_end_ratio, turnover_rate, num_synapses_start = 0.018, 0.014, 0.15, 1200  # MSS + bouton turnover
# COMPLEX_start_ratio, COMPLEX_end_ratio, turnover_rate, num_synapses_start = 0.050, 0.068, 0.10, 1200  # MSB + spine turnover


MSBorMSS_start = round(num_synapses_start * (COMPLEX_start_ratio/2))
COMPLEX_start = MSBorMSS_start * 2
SIMPLE_start = num_synapses_start - COMPLEX_start

num_formation = round(num_synapses_start * turnover_rate)

MSBorMSS_end = round(num_synapses_start * (COMPLEX_end_ratio/2))
COMPLEX_end = MSBorMSS_end * 2
SIMPLE_end = SIMPLE_start + MSBorMSS_start - MSBorMSS_end

num_elimination = num_formation + MSBorMSS_start - MSBorMSS_end

# print(SIMPLE_start)
# print(MSBorMSS_start)
# print(COMPLEX_start)
# print(num_elimination)
# print(num_formation)
# print(SIMPLE_end)
# print(MSBorMSS_end)
# print(COMPLEX_end)


names = globals()
for k in range(1, min(MSBorMSS_end, num_formation) + 1):
    names['N' + str(k)] = 0

N = 0
for i in range(min(num_elimination, COMPLEX_start) + 1):  # The number of complex synapses eliminated
    # print('i:', i)
    for j in range(math.ceil(i / 2), min(i, MSBorMSS_start) + 1):  # The number of MSBs/MSSs where synaptic elimination occurs; j <= i
        # print('j:', j)
        if SIMPLE_start + j >= SIMPLE_end:
            a = comb(SIMPLE_start, num_elimination - i) * (comb(MSBorMSS_start, j) * comb(j, j * 2 - i) * (2 ** (j * 2 - i)))
            b = comb(SIMPLE_start - (num_elimination - i) + (j * 2 - i) + ((num_elimination - i) + (i - j)),
                         (SIMPLE_start - (num_elimination - i) + (j * 2 - i) + ((num_elimination - i) + (i - j))) - SIMPLE_end)
            N += a * b
            c = [coeff(k, (SIMPLE_start - (num_elimination - i)), ((num_elimination - i) + (i - j) + (j * 2 - i)),
                           ((SIMPLE_start - (num_elimination - i) + (j * 2 - i) + ((num_elimination - i) + (i - j))) - SIMPLE_end))
                for k in range(1, min(MSBorMSS_end, num_formation) + 1)]
            for k in range(1, min(MSBorMSS_end, num_formation) + 1):
                names['N' + str(k)] += a * c[k - 1]

N_add = 0
for k in range(1, min(MSBorMSS_end, num_formation) + 1):
    N_add += k * names.get('N' + str(k))
n_add = N_add / N
n_compete = num_formation - n_add
print('n_add:', n_add, ' n_add(%):', n_add / num_formation)
print('n_compete:', n_compete, ' n_compete(%):', n_compete / num_formation)
