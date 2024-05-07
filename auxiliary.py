from math import log

B = -2000000  # lower bound for log probabilities


class Auxiliary:
    @staticmethod
    def bounded_log(a):
        if a == 0:
            return B
        else:
            return log(a)

    @staticmethod
    def bounded_log_sum(log_a, log_b, *args):
        log_sum = log_a + log_b
        for ar in args:
            log_sum += ar
        if log_sum < B:
            return B
        else:
            return log_sum

    @staticmethod
    def printArray(array):
        for j in range(len(array)):
            print("{} ".format(array[j]), end='')

    @staticmethod
    def clear_dllist(dl_list):
        llist = dl_list.last
        while llist is not None:
            temp = llist.prev
            dl_list.remove(llist)
            llist = temp
