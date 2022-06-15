def print_fully_connected(n=10):
    print(sum(range(n)))
    for i in range(n):
        for j in range(i, n):
            if (j != i):
                print(f"{i},{j}")

if (__name__ == "__main__"):
    print_fully_connected()