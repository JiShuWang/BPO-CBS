import pandas as pd

Combination = {}
Data = {
    "BPR": {},
    "LearningChain": {},
    "BPOCBS": {}
}
Performance = {
    "BPR": {},
    "LearningChain": {},
    "BPOCBS": {},
}
Regulation = {
    "BPR": {},
    "LearningChain": {},
    "BPOCBS": {},
}


def BPDScoring(t, l):
    # Reading the dataset
    BPR = pd.read_csv("../Data/BPD_BPR.csv").values.tolist()
    LearningChain = pd.read_csv("../Data/BPD_LearningChain.csv").values.tolist()
    BPOCBS = pd.read_csv("../Data/BPD_BPO-CBS.csv").values.tolist()

    BPR.sort(key=lambda x: (x[0], x[1]))
    LearningChain.sort(key=lambda x: (x[0], x[1]))
    BPOCBS.sort(key=lambda x: (x[0], x[1]))

    # Constructing the data structure
    for index in range(len(BPR)):
        tar = round(BPR[index][0] / 5) * 5
        blocksize = BPR[index][1]
        latency, BPRlatency, LearningChainlatency, BPOCBSlatency = BPR[index][2], BPR[index][-2], LearningChain[index][-2], BPOCBS[index][-2]
        throughput, BPRthroughput, LearningChainthroughput, BPOCBSthroughput = BPR[index][3], BPR[index][-1], LearningChain[index][-1], BPOCBS[index][-1]
        if tar not in Combination:
            Combination.setdefault(tar, {})
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain", "BPOCBS"]:
                globals()["Data"][methods].setdefault(tar, {})
                globals()["Data"][methods][tar].setdefault(blocksize,
                                                           [locals()[methods + str("latency")], locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
                globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                   None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
        if blocksize not in Combination[tar]:
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain", "BPOCBS"]:
                globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
        else:  # Saving the better data for the duplicate
            if throughput > Combination[tar][blocksize][1]:
                Combination[tar][blocksize] = [latency, throughput]
            elif latency < Combination[tar][blocksize][0]:
                Combination[tar][blocksize] = [latency, throughput]

    # Finding the min and max latency and throughput
    for tar in Combination:
        for blocksize in Combination[tar]:
            for methods in ["BPR", "LearningChain"]:
                pred_latency, pred_throughput = globals()["Data"][methods][tar][blocksize][0], globals()["Data"][methods][tar][blocksize][1]
                if pred_latency < globals()["Performance"][methods][tar][0]:  # lower latency
                    globals()["Performance"][methods][tar][0] = pred_latency
                if pred_latency > globals()["Performance"][methods][tar][1]:  # higher latency
                    globals()["Performance"][methods][tar][1] = pred_latency
                if pred_throughput < globals()["Performance"][methods][tar][2]:  # lower throughput
                    globals()["Performance"][methods][tar][2] = pred_throughput
                if pred_throughput > globals()["Performance"][methods][tar][3]:  # higher throughput
                    globals()["Performance"][methods][tar][3] = pred_throughput
        for i in range(10, tar + 1, 5):
            for blocksize in Combination[i]:
                pred_latency, pred_throughput = Data["BPOCBS"][i][blocksize][0], Data["BPOCBS"][i][blocksize][1]
                if pred_latency < Performance["BPOCBS"][tar][0]:  # lower latency
                    Performance["BPOCBS"][tar][0] = pred_latency
                if pred_latency > Performance["BPOCBS"][tar][1]:  # higher latency
                    Performance["BPOCBS"][tar][1] = pred_latency
                if pred_throughput < Performance["BPOCBS"][tar][2]:  # lower throughput
                    Performance["BPOCBS"][tar][2] = pred_throughput
                if pred_throughput > Performance["BPOCBS"][tar][3]:  # higher throughput
                    Performance["BPOCBS"][tar][3] = pred_throughput

    # Scoring and getting the optimal BCP and traffic corresponding to actual performance, thus evaluate the effectiveness of BPO methods
    for tar in Combination:
        for methods in ["BPR", "LearningChain"]:
            for blocksize in Combination[tar]:  # Only can regulate the BCP in BPR and LearningChain
                pred_latency, pred_throughput = globals()["Data"][methods][tar][blocksize][0], globals()["Data"][methods][tar][blocksize][1]
                min_latency, max_latency, min_throughput, max_throughput = globals()["Performance"][methods][tar][0], globals()["Performance"][methods][tar][1], \
                                                                           globals()["Performance"][methods][tar][2], globals()["Performance"][methods][tar][3]
                score = t * (pred_throughput - min_throughput) / (
                        max_throughput - min_throughput) + l * (
                                pred_latency - max_latency) / (
                                min_latency - max_latency)
                if score > globals()["Performance"][methods][tar][4]:
                    globals()["Performance"][methods][tar][4], globals()["Performance"][methods][tar][5], globals()["Performance"][methods][tar][6] = score, \
                                                                                                                                                      Combination[tar][blocksize][
                                                                                                                                                          0], \
                                                                                                                                                      Combination[tar][blocksize][1]
        for i in range(10, tar + 1, 5):  # Global scoring: not only can regulate the BCP, but also can regulate the transaction traffic in BPO-CBS
            for blocksize in Combination[i]:
                pred_latency, pred_throughput = Data["BPOCBS"][i][blocksize][0], Data["BPOCBS"][i][blocksize][1]
                min_latency, max_latency, min_throughput, max_throughput = Performance["BPOCBS"][tar][0], Performance["BPOCBS"][tar][1], Performance["BPOCBS"][tar][2], \
                                                                           Performance["BPOCBS"][tar][3]
                score = t * (pred_throughput - min_throughput) / (max_throughput - min_throughput) + l * (
                        pred_latency - max_latency) / (
                                min_latency - max_latency)
                if score > Performance["BPOCBS"][tar][4]:
                    Performance["BPOCBS"][tar][4], Performance["BPOCBS"][tar][5], Performance["BPOCBS"][tar][6] = score, \
                                                                                                                  Combination[i][blocksize][0], \
                                                                                                                  Combination[i][blocksize][1]


if __name__ == '__main__':
    BPDScoring(1, 0)
