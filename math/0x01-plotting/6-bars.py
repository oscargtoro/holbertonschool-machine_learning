#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

apples = np.array([x for x in fruit[0, :]])
bananas = np.array([x for x in fruit[1, :]])
oranges = np.array([x for x in fruit[2, :]])
peaches = np.array([x for x in fruit[3, :]])
labels = ['Farrah', 'Fred', 'Felicia']
width = 0.5
fig, ax = plt.subplots()
ax.set_title('Number of Fruit per Person')
ax.bar(labels, apples, width, label='apples', color='red')
ax.bar(labels, bananas, width, bottom=apples, label='bananas', color='yellow')
ax.bar(labels, oranges, width, bottom=bananas + apples,
       label='oranges', color='orange')
ax.bar(labels, peaches, width, bottom=bananas + apples + oranges,
       label='peaches', color='#ffe5b4')
ax.legend()
ax.set_ylabel('Quantity of Fruit')
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(0, 80, 10))
plt.show()
