n = 10

def positive_mod(x, mod):
    m = x % mod
    if (m < 0):
        m += mod
    return m


for i in range(n):
    print(f"{i},{positive_mod(i-1, n)},{positive_mod(i+1, n)}")