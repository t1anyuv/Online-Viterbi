from pyllist import dllist

from auxiliary import Auxiliary, B


class OnlineViterbi:
    """
    Online Viterbi algorithm implementation.

    Attributes:
        K (int): Number of Hidden States.
        T (int): Number of Time Instances.
        prob_list (dllist): Doubly linked list to store probability lists.
        state_list (dllist): Doubly linked list to store state lists.
        node_list (dllist): Doubly linked list for survivor memory.
        root (dllist node or None): Convergence point.
        prev_root (dllist node or None): Previous convergence point.
        delta_t (int or None): Distance between `root` and `prev_root`.
        decoded_stream (list): Solution path.
    """
    def __init__(self, K, T):
        """
        Initializes the OnlineViterbi object.

        Args:
            K (int): Number of Hidden States.
            T (int): Number of Time Instances.
        """
        self.K = K
        self.T = T
        self.prob_list = dllist()
        self.state_list = dllist()
        self.node_list = dllist()
        self.root = None
        self.prev_root = None
        self.delta_t = None
        self.decoded_stream = []

    def clear_all_lists(self):
        """
            Clears all linked lists (prob_list, state_list, node_list).
        """
        Auxiliary.clear_dllist(self.prob_list)
        Auxiliary.clear_dllist(self.state_list)
        Auxiliary.clear_dllist(self.node_list)

    def initialization(self, starting_state, initial):
        """
        Initializes the online Viterbi algorithm.

        Args:
            starting_state (int): Starting state.
            initial (list): Initial distribution.

        """
        self.root = None
        self.prev_root = None
        self.decoded_stream.clear()
        self.clear_all_lists()

        initial_prob = [Auxiliary.bounded_log(prob) for prob in initial]
        initial_state = [starting_state] * self.K

        self.prob_list.append(initial_prob)
        self.state_list.append(initial_state)

    def compress(self, current_time):
        """
        Compresses the node list.

        Args:
            current_time (int): Current time instance.

        """
        current = self.node_list.last
        while current is not None:
            state, time, parent, num_children = current.value
            temp = None

            if num_children == 0 and time != current_time:
                if parent is not None:
                    parent.value[3] = parent.value[3] - 1
            else:
                while parent is not None and parent.value[3] == 1:
                    current.value[2] = current.value[2].value[2]
                    parent = current.value[2]

            current = current.prev

    def free_dummy_nodes(self, current_time):
        """
        Frees dummy nodes from the node list.

        Args:
            current_time (int): Current time instance.

        """
        current = self.node_list.last
        while current is not None:
            state, time, parent, num_children = current.value
            temp = current.prev

            if num_children <= 0 and time != current_time:
                self.node_list.remove(current)

            current = temp

    def find_new_root(self):
        """
        Finds the new root in the node list.

        Returns:
            bool: True if root has changed based on time delta between previous root and new root, False otherwise.
        """
        # first make sure path has merged
        if self.root is None:
            last = self.node_list.last
            traced_root = [None] * self.K
            leaf = last
            for i in range(self.K):
                current = leaf
                while current is not None:
                    temp = current
                    current = current.value[2]
                    if current is None:
                        traced_root[i] = temp
                leaf = leaf.prev

            result = False
            if len(traced_root) > 0:
                result = all(elem == traced_root[0] for elem in traced_root)

            if not result:
                return False

        # find new root
        current = self.node_list.last
        aux = None
        time = current.value[1]

        self.delta_t = current.value[1]

        while current is not None:
            if current.value[3] >= 2:
                aux = current

            current = current.value[2]

        if aux is not None:
            if self.root is None:
                self.root = aux
                self.delta_t = self.delta_t - aux.value[1]
                if self.delta_t == 0:
                    return False
                else:
                    return True
            else:
                if aux != self.root:
                    self.delta_t = self.delta_t - aux.value[1]
                    if self.delta_t == 0:
                        return False
                    else:
                        self.prev_root = self.root
                        self.root = aux
                        return True
        else:
            return False

    def traceback(self):
        """
        Traces back through the node list to find the decoded stream.
        """
        interim_decoded_stream = []
        p_col = self.prob_list.last
        s_col = self.state_list.last

        output = self.root.value[0]  # state
        # print("{} ".format(output), end='')
        interim_decoded_stream.append(output)

        for i in range(self.delta_t):
            # Find column corresponding to root
            s_col = s_col.prev if s_col else None
            p_col = p_col.prev if p_col else None

        if self.prev_root is None:
            depth = self.root.value[1]
        else:
            depth = (self.root.value[1] - self.prev_root.value[1] - 1)

        for _ in range(depth):
            output = s_col.value[output]
            interim_decoded_stream.append(output)
            S = s_col
            s_col = s_col.prev
            self.state_list.remove(S)
            P = p_col
            p_col = p_col.prev
            self.prob_list.remove(P)

        while p_col is not None:
            S = s_col
            s_col = s_col.prev
            self.state_list.remove(S)
            P = p_col
            p_col = p_col.prev
            self.prob_list.remove(P)

        interim_decoded_stream.reverse()
        self.decoded_stream.extend(interim_decoded_stream)

    def traceback_last_part(self):
        """
        Traces back the last part of the node list to find the decoded stream.
        """
        interim_decoded_stream = []
        p_col = self.prob_list.last
        s_col = self.state_list.last

        output = p_col.value.index(max(p_col.value))
        interim_decoded_stream.append(output)

        if self.root is None:
            depth = (self.T - 1)
        else:
            depth = (self.T - 1) - self.root.value[1] - 1

        for i in range(depth):
            output = s_col.value[output]
            interim_decoded_stream.append(output)
            S = s_col
            s_col = s_col.prev

        interim_decoded_stream.reverse()
        self.decoded_stream.extend(interim_decoded_stream)

    def update(self, t, observation, A, E):
        """
        Updates the online Viterbi algorithm with the given observation.

        Args:
            t (int): Time instance.
            observation (int): Observation at time t.
            A (list): Transition Probability Matrix.
            E (list): Emission Matrix.

        """
        p_col = self.prob_list.last
        s_col = self.state_list.last
        last_node = self.node_list.last

        pCol = [B] * self.K
        sCol = [0] * self.K

        for j in range(self.K):
            max_val = B
            max_index = 0

            for i in range(self.K):
                aux = Auxiliary.bounded_log_sum(p_col.value[i], Auxiliary.bounded_log(A[i][j]),
                                                Auxiliary.bounded_log(E[j][observation]))
                if aux > max_val:
                    max_val = aux
                    max_index = i

            pCol[j] = max_val
            sCol[j] = max_index

            if t == 0:
                parent_node = None
            else:
                temp = self.K - max_index - 1
                parent_node = last_node
                while temp > 0:
                    parent_node = parent_node.prev
                    temp -= 1
                parent_node.value[3] = parent_node.value[3] + 1

            self.node_list.append([j, t, parent_node, 0])

        self.prob_list.append(pCol)
        self.state_list.append(sCol)

        self.compress(t)
        self.free_dummy_nodes(t)

        if self.find_new_root():
            self.traceback()

    def printProbList(self):
        """
        Prints the probability list.
        """
        prob = self.prob_list.last
        while prob is not None:
            print(prob)
            prob = prob.prev

        print("\n\n")

    def printStateList(self):
        """
        Prints the state list.
        """
        state = self.state_list.last
        while state is not None:
            print(state)
            state = state.prev

        print("\n\n")

    def printList(self):
        """
        Prints the node list.
        """
        node = self.node_list.last
        while node is not None:
            print(node)
            node = node.prev

        print("\n\n")
