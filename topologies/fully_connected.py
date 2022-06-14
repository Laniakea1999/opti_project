n = 10
for i in range(n):
    s = f"{i}"
    for j in range(n):
        s += f",{j}" if (j != i) else ""
    print(s)