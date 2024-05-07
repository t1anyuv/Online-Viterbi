import unittest
import time
import random
from onlineViterbi import OnlineViterbi
from standardViterbi import StandardViterbi


class TestViterbi(unittest.TestCase):
    def test_viterbi(self):
        K = 4  # num of Hidden States
        M = 4  # num of Observation Symbols
        T = 1000  # num of time instances

        A = [[0.96, 0.04, 0.0, 0.0],  # Transition Probability Matrix
             [0, 0.95, 0.05, 0.0],
             [0.0, 0.0, 0.85, 0.15],
             [0.1, 0.0, 0.0, 0.9]]

        E = [[0.6, 0.2, 0.0, 0.2],  # Emission Matrix
             [0.1, 0.8, 0.1, 0.0],
             [0.0, 0.14, 0.76, 0.1],
             [0.1, 0.0, 0.1, 0.8]]

        initial = [0.25, 0.25, 0.25, 0.25]  # Initial distribution

        observations = [0] * T
        previous = 0

        online_viterbi = OnlineViterbi(K, T)
        standard_viterbi = StandardViterbi(K, T)

        online_viterbi.initialization(0, initial)

        for i in range(100 * 60 * 60):

            count = i % T
            observations[count] = int((previous + (2 * random.random()) % 2) % K)
            previous = observations[count]

            online_viterbi.update(count, observations[count], A, E)

            if count == T - 1:
                online_viterbi.traceback_last_part()

                standard_viterbi.viterbi(observations, initial, A, E)

                self.assertEqual(standard_viterbi.optimalPath[0:min(T, 1000)],
                                 online_viterbi.decoded_stream[0:min(T, 1000)])

                online_viterbi.initialization(0, initial)
                time.sleep(1)

            time.sleep(0.01)


if __name__ == '__main__':
    unittest.main()
