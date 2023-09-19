import csv
import numpy as np
import time

NUM_STATES = 312020
NUM_ACTIONS = 9

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95

# init Q matrix
Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)


def read(file):  # returns array of arrays, each inner array is a row of data
    data = []
    with open(file, newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header row
        for row in reader:
            line = [int(x) for x in row]
            data.append(line)
        return data


def Q_learning(Q, data):
    prev_mean, curr_mean = -10e10, 0
    count = 0
    while count < 500 and round(curr_mean, 2) != round(prev_mean, 2):  # continue until significant increase in mean of Q-matrix
        for i in range(len(data)):
            s, a, r, sp = data[i][0] - 1, data[i][1] - 1, data[i][2] - 1, data[i][3] - 1  # subtract 1 to get index
            Q[s][a] += LEARNING_RATE * (r + DISCOUNT_RATE * (max_Q(Q, sp)) - Q[s][a])  # Q learning equation
            prev_mean = curr_mean
            curr_mean = Q.mean()
        count += 1 # count puts a limit on runtime just in case the large dataset does not converge fast enough
    return Q


def max_Q(Q, sp):  # helper function finds best Q in a row of the matrix given a state
    maxQ = -10e10
    for a in range(NUM_ACTIONS):
        if Q[sp, a] > maxQ:
            maxQ = Q[sp, a]
    return maxQ


def write_pol(pol, filename):
    with open(filename, 'w') as f:
        for i in pol:
            f.write(str(i) + '\n')


def main():
    start_time = time.time()
    data = read('large.csv')
    Q_final = Q_learning(Q, data)
    P = np.argmax(Q_final, axis=1) + 1
    print(Q_final)
    print("print P")
    print(P)
    end_time = time.time()
    print(end_time-start_time)
    return write_pol(P, "large.policy")


if __name__ == '__main__':
    main()
