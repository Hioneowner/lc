'''
108. 将有序数组转换为二叉搜索树

380. 常数时间插入、删除和获取随机元素
设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构。

insert(val)：当元素 val 不存在时，向集合中插入该项。
remove(val)：元素 val 存在时，从集合中移除该项。
getRandom：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。


import random
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.values = []
        self.index = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.index:
            return False
        self.values.append(val)
        self.index[val] = len(self.values) - 1
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.index:
            return False
        self.index[self.values[-1]] = self.index[val]
        self.values[-1], self.values[self.index[val]] = self.values[self.index[val]], self.values[-1]
        self.values.pop()
        self.index.pop(val)
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return self.values[random.randint(0, len(self.values) - 1)]

randomSet = RandomizedSet()
randomSet.insert(8)
randomSet.insert(9)
randomSet.insert(7)
randomSet.remove(9)
randomSet.insert(2)
randomSet.getRandom()
randomSet.remove(1)
randomSet.insert(2)
randomSet.getRandom()





863. 二叉树中所有距离为 K 的结点
554. 砖墙
题：画一条自顶向下的、穿过最少砖块的垂线。

法：这题极端情况，一条垂线穿过最多砖块的数目就是，整个砖墙的行数len(walls)。那么我们对每个位置都 建立index与砖块边缘数目的关系。
比如题目中给的例子：行数=len(walls)=6,列数=6，对于index=0和index=6是整个砖墙的边缘，不考虑。对于index=1时，在该位置结束的砖块数有3个，index=2时，
在该位置结束的砖块数仅有1个，index=3时，在该位置结束的砖块数有3个，index=4时，在该位置结束的砖块数有4个，是最多的情况，所以len(walls)-4，就是本题的答案。
def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        mark={}
        for x in wall:
            tmp=0
            for i in range(len(x)-1):
                tmp+=x[i]
                mark[tmp]=mark.get(tmp,0)+1
        print(mark)
        if not mark:return len(wall)
        return len(wall)-max(mark.values())




6. Z 字形变换
将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
示例1：
输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
P   A   H   N
A P L S I I G
Y   I   R


示例2：
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
P     I     N
A   L S   I G
Y A   H R
P     I

比如输入字符串为 “LEETCODEISHIRING” 行数为 3 时，排列如下：
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if s == "" or numRows == 1 :
            return s
        temp = ["" for _ in range(numRows)]
        flag = -1
        i = 0
        for c in s:
            temp[i] += c
            if i == 0 or i==numRows-1:
                flag = -flag
            i = i + flag
        return "".join(temp)





剑指 Offer 35. 复杂链表的复制
先用哈希表生成每个节点对应的新的节点，然后从头遍历原链表，根据next,random指针给新的链表设置next,random值。
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return
        dic={}
        cur=head
        while cur:
            dic[cur]=Node(cur.val)
            cur=cur.next
        cur=head
        while cur:
            dic[cur].next=dic.get(cur.next)
            dic[cur].random=dic.get(cur.random)
            cur=cur.next
        return dic[head]


130. 被围绕的区域
给定一个二维的矩阵，包含 ‘X’ 和 ‘O’（字母 O）。
找到所有被 ‘X’ 围绕的区域，并将这些区域里所有的 ‘O’ 用 ‘X’ 填充。
示例:
X X X X
X O O X
X X O X
X O X X

运行你的函数后，矩阵变为：
X X X X
X X X X
X X X X
X O X X
解释:
被围绕的区间不会存在于边界上，换句话说，任何边界上的 ‘O’ 都不会被填充为 ‘X’。 任何不在边界上，或不与边界上的 ‘O’ 相连的 ‘O’ 最终都会被填充为 ‘X’。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
#深度优先搜索（DFS）
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or len(board)==0:
            return

        def dfs(board, i, j):
            # 不处于矩阵内，或者如果已经标记的话，直接跳过
            if not (0<=i<m) or not (0<=j<n) or board[i][j] != 'O':
                return

            # 当确认为未标记的，并与边界 'O' 直接间接相连的 'O' 时，标记为 'M'
            board[i][j] = 'M'
            # 向四个方位扩散
            # 上下左右
            dfs(board, i-1, j)
            dfs(board, i+1, j)
            dfs(board, i, j-1)
            dfs(board, i, j+1)

        m = len(board)
        n = len(board[0])

        # 从边界的 'O' 开始搜索
        for i in range(m):
            for j in range(n):
                # 先确认 i，j 是否处于边界
                is_frontier = (i == 0 or j == 0 or i == m-1 or j == n-1)
                if is_frontier and board[i][j] == 'O':
                    dfs(board, i, j)

        # 遍历二维数组，将被标记为 'M' 的重新转换为 'O'，未标记的，则转换为 'X'
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'M':
                    board[i][j] = 'O'



剑指 Offer 32 - III. 从上到下打印二叉树 III
给定二叉树: [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [20,9],
  [15,7]
]

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        from collections import deque
        res, queue = [], deque()
        queue.append(root)
        while queue:
            tmp = []
            for _ in range(len(queue)):
                p = queue.popleft()
                tmp.append(p.val)
                if p.left:
                	queue.append(p.left)
                if p.right:
                	queue.append(p.right)
            res.append(tmp[::-1] if len(res) % 2 else tmp)
        return res


'''