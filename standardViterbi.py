from auxiliary import Auxiliary, B


class StandardViterbi:
    """
    A class for implementing the standard Viterbi algorithm.

    Attributes:
        K (int): Number of Hidden States.
        T (int): Number of Time Instances.
        scores (list): 2D list to store scores.
        path (list): 2D list to store paths.
        optimalPath (list): List to store the optimal path.
    """
    def __init__(self, K, T):
        """
        Initializes the StandardViterbi object.

        Args:
            K (int): Number of Hidden States.
            T (int): Number of Time Instances.
        """
        self.K = K
        self.T = T
        self.scores = [[0] * T for _ in range(K)]
        self.path = [[0] * T for _ in range(K)]
        self.optimalPath = [0] * T

    def initialization(self, observations, initial, A, E):
        """
        Initializes the scores and paths for the Viterbi algorithm.

        Args:
            observations (list): Observations at each time instance.
            initial (list): Initial distribution.
            A (list): Transition Probability Matrix.
            E (list): Emission Matrix.
        """
        for j in range(self.K):
            max_val = B
            max_index = 0
            for i in range(self.K):
                aux = Auxiliary.bounded_log_sum(Auxiliary.bounded_log(initial[i]),
                                                Auxiliary.bounded_log(A[i][j]),
                                                Auxiliary.bounded_log(E[j][observations[0]]))
                if aux > max_val:
                    max_val = aux
                    max_index = i
            self.scores[j][0] = max_val
            self.path[j][0] = max_index

    def recursion(self, observations, A, E):
        """
        Performs the recursion step of the Viterbi algorithm.

        Args:
            observations (list): Observations at each time instance.
            A (list): Transition Probability Matrix.
            E (list): Emission Matrix.
        """
        for t in range(1, self.T):
            for j in range(self.K):
                max_val = B
                max_index = 0
                for i in range(self.K):
                    aux = Auxiliary.bounded_log_sum(self.scores[i][t - 1],
                                                    Auxiliary.bounded_log(A[i][j]),
                                                    Auxiliary.bounded_log(E[j][observations[t]]))
                    if aux > max_val:
                        max_val = aux
                        max_index = i
                self.scores[j][t] = max_val
                self.path[j][t] = max_index

    def termination(self):
        """
        Performs the termination step of the Viterbi algorithm.
        """
        max_val = B
        max_index = 0
        for j in range(self.K):
            if self.scores[j][self.T - 1] > max_val:
                max_val = self.scores[j][self.T - 1]
                max_index = j
        self.optimalPath[self.T - 1] = max_index
        for t in range(self.T - 2, -1, -1):
            self.optimalPath[t] = self.path[self.optimalPath[t + 1]][t + 1]

    def viterbi(self, observations, initial, A, E):
        """
        Executes the Viterbi algorithm.

        Args:
            observations (list): Observations at each time instance.
            initial (list): Initial distribution.
            A (list): Transition Probability Matrix.
            E (list): Emission Matrix.
        """
        self.initialization(observations, initial, A, E)
        self.recursion(observations, A, E)
        self.termination()
