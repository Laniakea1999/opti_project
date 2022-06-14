import random

cities=3
suburbs=7


i = 0
city_map = {}
for j in range(cities):
    city = i
    city_map.update({city: f"{i}"})
    i += 1
    for k in range(random.randint(0, suburbs)):
        print(f"{i},{city}")
        city_map[city] += f",{i}"
        i += 1

for c in city_map.keys():
    for c2 in city_map.keys():
        city_map[c] += f",{c2}" if c2 != c else ""
    print(city_map[c])
