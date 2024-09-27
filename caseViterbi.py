import csv
import random
import time
import tracemalloc

from onlineViterbi import OnlineViterbi
from standardViterbi import StandardViterbi

B = -2000000  # lower bound for log probabilities

if __name__ == '__main__':
    K = 3  # num of Hidden States
    M = 3  # num of Observation Symbols
    T = 100  # num of time instances

    A = [[0.7, 0.2, 0.1],  # Transition Probability Matrix
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    E = [[0.8, 0.1, 0.1],  # Emission Matrix
         [0.3, 0.4, 0.3],
         [0.2, 0.6, 0.2]]

    initial = [0.6, 0.3, 0.1]  # Initial distribution

    observations = [0] * T
    previous = 0

    online_viterbi = OnlineViterbi(K, T)
    standard_viterbi = StandardViterbi(K, T)

    online_viterbi_time, standard_viterbi_time = 0, 0

    online_viterbi.initialization(0, initial)

    # 开始跟踪内存分配
    tracemalloc.start()

    # 打开 CSV 文件用于写入数据
    with open('viterbi_performance.csv', mode='w', newline='') as csv_file:
        fieldnames = ['iteration', 'observations', 'standard_time', 'online_time', 'nodes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        for i in range(T * 100):
            count = i % T
            observations[count] = int((previous + (2 * random.random()) % 2) % K)
            previous = observations[count]

            start_time = time.time()
            online_viterbi.update(count, observations[count], A, E)
            end_time = time.time()

            update_time = end_time - start_time
            online_viterbi_time += update_time

            if count == T - 1:
                start_time = time.time()
                online_viterbi.traceback_last_part()
                end_time = time.time()
                online_viterbi_time += end_time - start_time

                start_time = time.time()
                standard_viterbi.viterbi(observations, initial, A, E)
                end_time = time.time()
                standard_viterbi_time = end_time - start_time

                # 打印并记录结果
                print("observations: ", observations)
                print(standard_viterbi.optimalPath[0:min(T, 1000)] == online_viterbi.decoded_stream[0:min(T, 1000)])

                print("标准Viterbi运行时间: {:.8f}秒".format(standard_viterbi_time))
                print("在线Viterbi运行时间: {:.8f}秒".format(online_viterbi_time))

                print("在线Viterbi存储的节点个数: ", online_viterbi.node_list.size)

                # 写入 CSV 文件中的一行
                writer.writerow({
                    'iteration': i // T,
                    'observations': observations,  # 可根据需要调整输出的观测值格式
                    'standard_time': standard_viterbi_time,
                    'online_time': online_viterbi_time,
                    'nodes': online_viterbi.node_list.size
                })

                online_viterbi.initialization(0, initial)
                online_viterbi_time = 0

            time.sleep(0.01)
