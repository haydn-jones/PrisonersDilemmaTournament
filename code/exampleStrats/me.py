import numpy as np

DEFECT = 0
COOP   = 1

def strategy(history, memory):
    stepNum = history.shape[1]
    sched = [COOP, DEFECT, COOP, DEFECT]

    if stepNum < len(sched):
        return sched[stepNum], None

    probs, phi = getPhi(history)

    if phi > 0.35:
        choice = COOP
    else:
        choice = DEFECT

    return choice, None

def getPhi(history):
    meC, themC = history[0, :-1], history[1, 1:]
    
    cTable = np.zeros((2, 2))
    for m, t in zip(meC, themC):
        cTable[m, t] += 1

    A = cTable[0, 0]
    B = cTable[0, 1]
    C = cTable[1, 0]
    D = cTable[1, 1]

    denom = np.sqrt((A+B)*(C+D)*(A+C)*(B+D))
    if denom == 0:
        phi = 0
    else:
        phi = (A*D-B*C) / denom

    probs = cTable / cTable.sum(axis=-1)[:, None]
    return probs, phi
