import random


def random_color():
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    while True:
        yield random.choice(colors)


# Example usage:
color_generator = random_color()
for _ in range(5):
    print(next(color_generator))