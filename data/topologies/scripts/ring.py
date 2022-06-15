
def print_ring(n=20):
    print(f"{n}")
    for i in range(n - 1):
        print(f"{i},{i+1}")
    print(f"{n - 1},{0}")

if (__name__ == '__main__'):
    print_ring()