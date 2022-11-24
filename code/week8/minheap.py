import heapq


class BtmkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        num1 = elem[0]
        num1 = -num1
        num2 = elem[1]

        if len(self.data) < self.k:
            heapq.heappush(self.data, (num1, num2))
        else:
            topk_small = self.data[0][0]
            if elem[0] > topk_small:
                heapq.heapreplace(self.data, (num1, num2))

    def BtmK(self):
        return sorted([(-x[0], x[1]) for x in self.data])

    def getSmallest(self):
        return -heapq.nlargest(1, self.data)[0][0]

    def getSmallestV(self):
        return heapq.nlargest(1, self.data)[0][1]


# a = (100, [1,1])
# b = (200, [1,1])
# c = (300, [1,1])
# d = (400, [1,1])
# e = (50, [1,1])
# btm = BtmkHeap(3)
# btm.Push(a)
# print(btm.BtmK())
# print(btm.getSmallest())
# btm.Push(b)
# print(btm.BtmK())
# print(btm.getSmallest())
# btm.Push(c)
# print(btm.BtmK())
# print(btm.getSmallest())
# btm.Push(e)
# print(btm.BtmK())
# print(btm.getSmallest())
# btm.Push((23, [1,1]))
# print(btm.BtmK())
# print(btm.getSmallest())
