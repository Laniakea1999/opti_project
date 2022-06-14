n = 10

zero_string = "0"
for i in range(1, n):
    zero_string += f",{i}"
    print(f"{i},0")
print(zero_string)