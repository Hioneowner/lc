'''




716. 最大栈
设计一个最大栈数据结构，既支持栈操作，又支持查找栈中最大元素。

实现 MaxStack 类：

MaxStack() 初始化栈对象
void push(int x) 将元素 x 压入栈中。
int pop() 移除栈顶元素并返回这个元素。
int top() 返回栈顶元素，无需移除。
int peekMax() 检索并返回栈中最大元素，无需移除。
int popMax() 检索并返回栈中最大元素，并将其移除。如果有多个最大元素，只要移除 最靠近栈顶 的那个。
示例：

输入
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
输出
[null, null, null, null, 5, 5, 1, 5, 1, 5]

解释
MaxStack stk = new MaxStack();
stk.push(5);   // [5] - 5 既是栈顶元素，也是最大元素
stk.push(1);   // [5, 1] - 栈顶元素是 1，最大元素是 5
stk.push(5);   // [5, 1, 5] - 5 既是栈顶元素，也是最大元素
stk.top();     // 返回 5，[5, 1, 5] - 栈没有改变
stk.popMax();  // 返回 5，[5, 1] - 栈发生改变，栈顶元素不再是最大元素
stk.top();     // 返回 1，[5, 1] - 栈没有改变
stk.peekMax(); // 返回 5，[5, 1] - 栈没有改变
stk.pop();     // 返回 1，[5] - 此操作后，5 既是栈顶元素，也是最大元素
stk.top();     // 返回 5，[5] - 栈没有改变
class MaxStack:

    def __init__(self):
        self.stack = []
        self.max_stack = [-10000001]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.max_stack.append(max(x, self.max_stack[-1]))

    def pop(self) -> int:
        self.max_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def peekMax(self) -> int:
        return self.max_stack[-1]

    def popMax(self) -> int:
        val = self.max_stack.pop()

        temp = [self.stack.pop()]
        while temp[-1] != val:
            self.max_stack.pop()
            temp.append(self.stack.pop())

        temp.pop()

        while temp:
            v = temp.pop()
            self.stack.append(v)
            self.max_stack.append(max(v, self.max_stack[-1]))

        return val







381. O(1) 时间插入、删除和获取随机元素 - 允许重复
解题思路： 这道题其实就是考一些语言特性，对于Python 语言我们可以直接用list来完成，插入使用append，删除使用remove，随机取数就随机一个int作为下标直接取。
import random
class RandomizedSet:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dict = {}
        self.list = []
    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True
    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            last_element, idx = self.list[-1], self.dict[val]
            self.list[idx], self.dict[last_element] = last_element, idx
            self.list.pop()
            del self.dict[val]
            return True
        return False
    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return choice(self.list)




273. 整数转换英文表示
示例 1：
输入：num = 123
输出：“One Hundred Twenty Three”

示例 2：
输入：num = 12345
输出：“Twelve Thousand Three Hundred Forty Five”

示例 3：
输入：num = 1234567
输出：“One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven”

示例 4：
输入：num = 1234567891
输出：“One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One”

class Solution:
    def numberToWords(self, num: int) -> str:
        to19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ' \
               'Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
        tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()

        def helper(num):
            if num < 20:
                return to19[num - 1:num]
            if num < 100:
                return [tens[num // 10 -2]] + helper(num % 10)
            if num < 1000:
                return [to19[num // 100 - 1]] + ["Hundred"] + helper(num % 100)
            for p, w in enumerate(["Thousand", "Million", "Billion"], 1):
                if num < 1000 ** (p + 1):
                    return helper(num // 1000 ** p) + [w] + helper(num % 1000 ** p)
        return " ".join(helper(num)) or "Zero"



'''