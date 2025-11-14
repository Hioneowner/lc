'''
238. 除自身以外数组的乘积
给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

示例:

输入: [1,2,3,4]
输出: [24,12,8,6]
1
2
提示： 题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。

进阶：

你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）

解题思路
思路：左右乘积列表

先看题目的提示，保证数组中任意元素的全部前缀后缀（甚至整个数组）的乘积都在 32 位整数范围内，那这里就可以不考虑数据溢出的问题。

再看说明，这个说明中表示，不能够使用除法。因为，如果要求除当前元素数组的乘积，只要先求得整个数组的乘积，除以当前元素，那么就是要求的答案。（当然，这里有个问题，如果当前元素是 0 的话，这里就要注意。不过题目不建议往这个方向考虑解决的方法，那么这里就不展开去说明了。）

现在看本篇幅使用的方法：左右乘积列表，这里需要先构造 left，right 两个数组，分别存储当前元素左侧的乘积以及右侧的乘积。

具体的构造方法：

初始化两个数组 left，right。其中 left[i] 表示 i 左侧全部元素的乘积。right[i] 表示 i 右侧全部元素的乘积。
开始填充数组。这里需要注意 left[0] 和 right[lenght-1] 的值（length 表示数组长度，right 数组是从右侧往左进行填充）。
对于 left 数组而言，left[0] 表示原数组索引为 0 的元素左侧的所有元素乘积，这里原数组第一位元素左侧是没有其他元素的，所以这里初始化为 1。而其他的元素则为：left[i] = left[i-1] * nums[i - 1]
同样的 right 数组，right[length-1] 表示原数组最末尾的元素右侧所有元素的乘积。但因为最末尾右侧没有其他元素，所以这里 right[length-1] 也初始化为 1。其他的元素则为：right[i]=right[i+1]*nums[i+1]
至于返回数组 output 数组，再次遍历原数组，索引为 i 的值则为：output[i] = left[i] * right[i]
主要在于 left 和 right 数组的构造
具体的实现代码见【code 1】 部分。

上面的方法实现之后，时间复杂度为 O(N)，空间复杂度也是 O(N) （N 表示数组的长度）。因为 构造 left 和 right 数组，两者的数组长度就是 N。

题目中的进阶部分，希望能够尝试使用常数空间复杂度完成本题。（这里不计输出数组的空间）

方法还是使用 左右乘积列表 的方法，但是这里不在单独构建 left 和 right 数组。直接在输出数组中进行构造。

具体的思路：

先将输出数组当成 left 数组进行构造
然后动态构造 right 数组计算结果
具体的做法：

先初始化 output 数组，先当成 left 数组进行构造，那么 output[i] 就表示 i 左侧所有元素的乘积。（具体的构造方法同上）
但是，这里我们不能够再单独构造 right 数组（前面说了，空间复杂度须为常数）。这里我们使用的方法是，在遍历输出答案的时候，维护一个变量 right，而变量 right 表示右侧元素的乘积。遍历更新 output[i] 的值为 output[i] * right，同时更新 right 的值为 right * nums[i] 表示遍历下一个元素右侧的元素乘积。
具体实现代码见【code 2】。
# code 1
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)

        left = [0] * length
        right = [0] * length
        output = [0] * length

        # 先填充 left 数组
        # left 数组表示遍历时 i 左侧的乘积
        # 因为数组第一个元素左侧没有其他元素，所以 left 数组第一个元素为 1
        # left 数组接下来的元素则为原数组的第一位元素与 left 数组第一位元素的乘积，依次类推
        left[0] = 1
        for i in range(1, length):
            left[i] = nums[i-1] * left[i-1]

        # 同样的 right 数组从右往左进行填充
        # 同样数组末尾元素右侧没有其他元素，所以末尾元素值为 1
        # 右边往左的元素则为原数组与 right 数组末尾往前一位元素的乘积，依次类推
        right[length-1] = 1
        for i in range(length-2, -1, -1):
            right[i] = nums[i+1] * right[i+1]

        # 重新遍历，输出 output 数组
        # output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积
        # 也就是 output[i] 的值为 left[i] * right[i]
        for i in range(length):
            output[i] = left[i] * right[i]

        return output


306. 累加数
描述：累加数 是一个字符串，组成它的数字可以形成累加序列。
一个有效的 累加序列 必须 至少 包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
给你一个只包含数字 ‘0’-‘9’ 的字符串，编写一个算法来判断给定输入是否是 累加数 。如果是，返回 true ；否则，返回 false 。
说明：累加序列里的数 不会 以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。
示例1
输入：“112358”
输出：true
解释：累加序列为: 1, 1, 2, 3, 5, 8 。1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8

示例2
输入：“199100199”
输出：true
解释：累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199

提示
1 <= num.length <= 35
num 仅由数字（0 - 9）组成
枚举所有可能出现的情况，按照满足题意的方式去组合字符串，最终判断是否能够组合出和原字符串相同的字符串，如果出现则返回True。
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        n = len(num)
        for i in range(1, n // 2 + 1):
            for j in range(i + 1, n // 3 * 2 + 1):
                a = int(num[:i])
                b = int(num[i:j])
                # 数值转字符串拼接
                res = f'{a}{b}'
                while len(res) < n:
                    a, b, res = b, a + b, res + str(a + b)
                if res == num:
                    return True
        return False




133. 克隆图
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。
class Node {
    public int val;
    public List<Node> neighbo rs;
}

# 广度优先搜索
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        from collections import deque

        marked = {}

        def bfs(node):
            if not node:
                return node
            # 克隆节点，放到哈希表中
            clone_node = Node(node.val, [])
            marked[node] = clone_node
            # 先将给定的节点入队
            queue = deque()
            queue.append(node)

            # 出队，开始遍历
            while queue:
                cur_node = queue.popleft()
                for neighbor in cur_node.neighbors:
                    # 如果邻接点不在哈希表中，克隆邻接点存入哈希表中，并将邻接点入队
                    if neighbor not in marked:
                        marked[neighbor] = Node(neighbor.val, [])
                        queue.append(neighbor)
                    # 更新当前节点的邻接列表，注意是克隆节点
                    marked[cur_node].neighbors.append(marked[neighbor])
            return clone_node

        return bfs(node)



118. 杨辉三角
解法1：
【解析】
① 首先初始化了一个列表“ret”，用来存放所有行的数字。
② 每一行的数字也以列表的形式存储，每一行的列表都初始化为row。
③ “i”表示某一行，即第1行, 第2行等等。j 表示特定行的某一个数字。
④ 根据性质①可知，杨辉三角每一行的开始和结尾都是1, 故当
j == 0：表示该行的第一个数字
j == i ：表示该行的最后一个数字
时候，直接给列表加1即可。
⑤ 除去每一行的首位元素末位元素直接为1外，其余所有元素根据性质③中的公式来计算即可。ret[i-1] : 定位到存放上一行所有数字的列表；ret[i - 1][j]：定位到第 i-1 行 的 第 j 个元素。同理ret[i - 1][j-1]
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ret = list()
        for i in range(numRows):
            row = list()
            for j in range(0, i + 1):
                if j == 0 or j == i:
                    row.append(1)
                else:
                    row.append(ret[i - 1][j] + ret[i - 1][j - 1])
            ret.append(row)
        return ret





986. 区间列表的交集
给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。
返回这 两个区间列表的交集 。
形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。

两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。
输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

示例 2：
输入：firstList = [[1,3],[5,9]], secondList = []
输出：[]

示例 3：
输入：firstList = [], secondList = [[4,8],[10,12]]
输出：[]

示例 4：
输入：firstList = [[1,7]], secondList = [[3,10]]
输出：[[3,7]]

思路：
先从最简单的两个区间的交集开始考虑，
假设我们有 L1 = [s1, e1], L2 = [s2, e2]，
那么它们的交集就应该是 [max(s1, s2), min(e1, e2)]。
下一步考虑对两个区间取了交集之后，到底谁该往后挪一步找下一个区间，
贪心告诉我们，应该移动 e 较小的那个 List, 因为长的 List 还有可能跟其他的 List 取到交集。
时间复杂度：O（N）
空间复杂度：O（1）
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        p1, p2 = 0, 0
        res = []
        while p1 < len(firstList) and p2 < len(secondList):
            interscetion = [max(firstList[p1][0], secondList[p2][0]), min(firstList[p1][1], secondList[p2][1])]
            if interscetion[0] <= interscetion[1]:
                res.append(interscetion)

            if firstList[p1][1] <= secondList[p2][1]:
                p1 += 1
            else:
                p2 += 1
        return res



233. 数字 1 的个数
给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
示例1：
输入：n = 13
输出：6

示例2：
输入：n = 0
输出：0
提示：
0 <= n <= 2 * 109
思路
动态规划
我首先想到的是枚举，但是毫无疑问超时了，数位dp看的是题解写的。

枚举
思路：dp[i]表示1~i里已经出现的1的总数,状态方程：dp[i] = dp[i - 1] + 这个数中出现的1的次数
class Solution:
    def cacul_1(self, num: int) -> int:
        """计算数字各位中1的个数"""
        ans = 0
        while num > 0:
            ans += 1 if num % 10 == 1 else 0
            num = num // 10
        return ans

    def countDigitOne(self, n: int) -> int:
        """计算1到n中所有数字的1的总数"""
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + self.cacul_1(i)
        return dp[n]




907. 子数组的最小值之和
给定一个整数数组 arr，找到 min(b) 的总和，其中 b 的范围为 arr 的每个（连续）子数组。
由于答案可能很大，因此 返回答案模 10^9 + 7 。

示例
示例 1：
输入：arr = [3,1,2,4]
输出：17
解释： 子数组为
[3]，[1]，[2]，[4]，[3,1]，[1,2]，[2,4]，[3,1,2]，[1,2,4]，[3,1,2,4]。 最小值为 3，1，2，4，1，1，2，1，1，1，和为 17。

示例 2：
输入：arr = [11,81,94,43,3]
输出：444

提示：
1 <= arr.length <= 3 * 104
1 <= arr[i] <= 3 * 104

思路
题目简而言之就是求给定数组的子数组的所有子数组中最小元素的和。
该问题我们可以转换为单调栈的问题，

我们需要找到每个元素 arr[i] 以该元素为最右且最小的子序列的数目 left[i]，以及以该元素为最左且最小的子序列的数目 right[i]，则以 arr[i] 为最小元素的子序列的数目合计为 left[i]×right[i]
只需要求出数组中当前元素 x（下标i） 左边第一个小于 x 的元素(下标j)以及右边第一个小于等于 x 的元素(下标k)
则以为arr[i]为最小元素对答案的贡献就是arr[i] * (i-j) * (k-i)，最后整合所有元素即可
MOD = 10 ** 9 + 7

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        monoStack = []
        # arr[i]左右两边小于该元素的元素数量和
        left = [0] * n
        right = [0] * n
        # 寻找左边
        for i, x in enumerate(arr):
            while monoStack and x <= arr[monoStack[-1]]:
                monoStack.pop()
            left[i] = i - (monoStack[-1] if monoStack else -1)
            monoStack.append(i)
        monoStack = []
        # 寻找右边
        for i in range(n - 1, -1, -1):
            while monoStack and arr[i] < arr[monoStack[-1]]:
                monoStack.pop()
            right[i] = (monoStack[-1] if monoStack else n) - i
            monoStack.append(i)
        ans = 0
        # 整合答案
        for l, r, x in zip(left, right, arr):
            ans = (ans + l * r * x) % MOD
        return ans



907. 子数组的最小值之和
给定一个整数数组 arr，找到 min(b) 的总和，其中 b 的范围为 arr 的每个（连续）子数组。
由于答案可能很大，因此 返回答案模 10^9 + 7 。

示例
示例 1：
输入：arr = [3,1,2,4]
输出：17
解释： 子数组为
[3]，[1]，[2]，[4]，[3,1]，[1,2]，[2,4]，[3,1,2]，[1,2,4]，[3,1,2,4]。 最小值为 3，1，2，4，1，1，2，1，1，1，和为 17。

示例 2：
输入：arr = [11,81,94,43,3]
输出：444

提示：
1 <= arr.length <= 3 * 104
1 <= arr[i] <= 3 * 104

思路
题目简而言之就是求给定数组的子数组的所有子数组中最小元素的和。
该问题我们可以转换为单调栈的问题，

我们需要找到每个元素 arr[i] 以该元素为最右且最小的子序列的数目 left[i]，以及以该元素为最左且最小的子序列的数目 right[i]，则以 arr[i] 为最小元素的子序列的数目合计为 left[i]×right[i]
只需要求出数组中当前元素 x（下标i） 左边第一个小于 x 的元素(下标j)以及右边第一个小于等于 x 的元素(下标k)
则以为arr[i]为最小元素对答案的贡献就是arr[i] * (i-j) * (k-i)，最后整合所有元素即可
MOD = 10 ** 9 + 7

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        monoStack = []
        # arr[i]左右两边小于该元素的元素数量和
        left = [0] * n
        right = [0] * n
        # 寻找左边
        for i, x in enumerate(arr):
            while monoStack and x <= arr[monoStack[-1]]:
                monoStack.pop()
            left[i] = i - (monoStack[-1] if monoStack else -1)
            monoStack.append(i)
        monoStack = []
        # 寻找右边
        for i in range(n - 1, -1, -1):
            while monoStack and arr[i] < arr[monoStack[-1]]:
                monoStack.pop()
            right[i] = (monoStack[-1] if monoStack else n) - i
            monoStack.append(i)
        ans = 0
        # 整合答案
        for l, r, x in zip(left, right, arr):
            ans = (ans + l * r * x) % MOD
        return ans






'''