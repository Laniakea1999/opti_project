def print_star(n=20):
    print(f"{n - 1}")
    for i in range(1, n):
        print(f"0,{i}")

if (__name__ == '__main__'):
    print_star