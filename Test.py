import Main

# RR 
ratios = [
    (100, 100, 100),
    (200, 50, 50),
    (50, 200, 50)
    (50, 50, 200)
]

for postb in range(-4, 5):
    for ratio in ratios:
        new_ratio = Main.run(ratio, postb)
        print(*new_ratio, sep = ' : ', end=', ')
        for i in range(3):
            print(new_ratio[i]-ratio[i], end=', ')
        print()