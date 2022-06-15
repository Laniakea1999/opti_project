
from re import I

import random

def positive_mod(x, mod):
    m = x % mod
    if (m < 0):
        m = m + mod
    return m

def print_city(args=(3, 5)):
    cities, suburbs = args
    i = 0
    n_suburbs = [suburbs for c in range(cities)]#[random.randint(0, suburbs) for c in range(cities)]

    city_ids = []

    print(cities + sum(n_suburbs))
    for c in range(cities):
        city = i
        city_ids.append(city)
        i += 1
        for s in range(n_suburbs[c]):
            print(f"{city},{i}")
            i += 1

    for idx, c in enumerate(city_ids):
        print(f"{c},{city_ids[positive_mod(idx + 1, cities)]}")


if (__name__ == "__main__"):
    print_city()