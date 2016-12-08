# -*- coding: utf-8 -*-
from random import randint
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Perceptron(object):
    def __init__(self, eta=1):
        self.eta = eta
        self.alpha = []
        self.bias = 0
        self.gram_matrix = []

    def check(self, item, i, labels):
        total = 0
        res = np.dot(self.alpha * labels, self.gram_matrix[i])
        return (res + self.bias) * labels[i]

    def find_unclassified_data(self, training_set):
        for i in range(len(training_set)):
            if self.check(training_set[i], i, np.array(training_set[:, 1])) <= 0:
                return i
        return None

    def optmise(self, unclassified_index, attrs, labels, history):
        self.alpha[unclassified_index] = self.alpha[unclassified_index] + self.eta
        self.bias = self.bias + labels[unclassified_index] * self.eta
        history.append([np.dot(self.alpha * labels, attrs), copy(self.alpha), self.bias])

    def init_gram(self, training_set):
        self.gram_matrix = np.empty((len(training_set), len(training_set)))
        for i in range(len(training_set)):
            for j in range(len(training_set)):
                self.gram_matrix[i][j] = np.dot(training_set[i][0], training_set[j][0])

    def train(self, training_set):
        self.init_gram(training_set)
        self.alpha = np.zeros(len(training_set))
        attrs, labels = self.splitTrainingSet(training_set)
        history = [[np.dot(self.alpha * labels, attrs), copy(self.alpha), self.bias]]
        while True:
            unclassified_index = self.find_unclassified_data(training_set)
            if unclassified_index is None:
                break
            self.optmise(unclassified_index, attrs, labels, history)
        return self.alpha, self.bias, history

    def splitTrainingSet(self, training_set):
        labels = np.array(training_set[:, 1])
        attrs = np.empty((len(training_set), 2), np.float)
        for i in range(len(training_set)):
            attrs[i] = training_set[i][0]
        return attrs, labels

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
        b = HISTORY[i][2]
        if w[1] == 0:
            return line, label
        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text("alpha = " + str(HISTORY[i][1]) + "\nw = " + str(HISTORY[i][0]) + " \nb = " + str(HISTORY[i][2]))
        label.set_position([x1, y1])
        return line, label

    print(HISTORY)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(HISTORY),
                                   interval=500, repeat=False, blit=False)
    plt.show()
