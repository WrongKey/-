# -*- coding: utf-8 -*-
from random import randint
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Perceptron(object):
    def __init__(self, eta=1):
        self.eta = eta

    @classmethod
    def classify(cls, item, weight, bias):
        return int((np.dot(item[0], weight)+ bias))

    def find_unclassified_set(self, training_set, weight, bias):
        unclassified_set = []
        for item in training_set:
            classify_result = self.classify(item, weight, bias)
            if classify_result * item[1] <= 0:
                unclassified_item = [copy(item)]
                unclassified_set.append(unclassified_item)
        return unclassified_set

    def optmise(self, weight, bias, unclassified_set):
        factor = unclassified_set[randint(0, len(unclassified_set) - 1)][0]
        return self.eta * np.dot(factor[0], factor[1]) + weight, self.eta * factor[1] + bias

    def train(self, training_set):
        weight = np.zeros(len(training_set[0][0]))
        bias = 0
        history = []
        while True:
            unclassified_set = self.find_unclassified_set(training_set, weight, bias)
            if len(unclassified_set) == 0:
                break
            weight, bias = self.optmise(weight, bias, unclassified_set)
            history.append([copy(weight), bias])
        return weight, bias, history

if __name__ == "__main__":
    PERCEPTRON = Perceptron(1)
    TRAINING_SET = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1], [[5, 2], -1]])
    WEIGHT, BIAS, HISTORY = PERCEPTRON.train(TRAINING_SET)

    fig = plt.figure()
    ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6))
    line, = ax.plot([], [], 'g', lw=1)
    label = ax.text([], [], '')

    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in TRAINING_SET:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])

        plt.plot(x, y, 'bo', x_, y_, 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Perceptron')
        return line, label


    # animation function.  this is called sequentially
    def animate(i):
        w = HISTORY[i][0]
        b = HISTORY[i][1]
        if w[1] == 0:
            return line, label
        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text("w = " + str(HISTORY[i][0]) + " \n b = " + str(HISTORY[i][1]))
        label.set_position([x1, y1])
        return line, label

    print(HISTORY)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(HISTORY),
                                   interval=500, repeat=False, blit=False)
    plt.show()
