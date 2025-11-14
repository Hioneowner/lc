'''
补充题2. 圆环回原点问题



91. 解码方法
一条包含字母 A-Z 的消息通过以下方式进行了编码：
‘A’ -> 1
‘B’ -> 2
…
‘Z’ -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

示例 1:
输入: “12”
输出: 2
解释: 它可以解码为 “AB”（1 2）或者 “L”（12）。

示例 2:
输入: “226”
输出: 3
解释: 它可以解码为 “BZ” (2 26), “VF” (22 6), 或者 “BBF” (2 2 6) 。
dp[i]表示到第i-1位时解码的方法数
两种情况：
1.s[i-1]单独解码，方法数为dp[i-1]
2.s[i-2:i]拼接成双字符解码，若10<=s[i-2:i]<26，双字符合格，解码的方法数位dp[i-2]，否则为0
综合两种情况，得到状态转移矩阵：
dp[i] = dp[i-1] + (dp[i-2] if 双字符合格 else 0)
为什么dp[i]表示的使i-1位？
例如 216，在判断第二位‘1’时，i-2<0了，状态转移矩阵不能用了，故在前加一位，即dp[0]为1

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1 if s[0]!='0' else 0
        for i in range(2,n+1):
            if s[i-1]!='0':
                dp[i] = dp[i-1]

            if 9< int(s[i-2:i])<27:
                dp[i] += dp[i-2]

        return dp[-1]






230. 二叉搜索树中第K小的元素
给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
说明：
你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。

示例 2:
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        inorder = list()
        self.inorderTra(root, inorder)
        print inorder

        return inorder[k-1]

    def inorderTra(self, node, path):
        if not node:
            return
        self.inorderTra(node.left, path)
        path.append(node.val)
        self.inorderTra(node.right, path)


572. 另一个树的子树
给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。

示例 1:
给定的树 s:
     3
    / \
   4   5
  / \
 1   2
给定的树 t：
   4
  / \
 1   2
返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。


示例 2:
给定的树 s：
     3
    / \
   4   5
  / \
 1   2
    /
   0

给定的树 t：
   4
  / \
 1   2
返回 false。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        return self.dfs(s, t)

    def dfs(self, c, t):
        # c 子树为空时，返回 False
        if not c:
            return False
        return self.is_same(c, t) or self.dfs(c.left, t) or self.dfs(c.right, t)

    def is_same(self, c, t):
        # 两个树都为空时，也认为是相同
        if (not c) and (not t):
            return True
        # 当其中一个树为空，但另外一个树不为空时，此时则为不同
        if (not c and t) or (c and not t):
            return False
        # 两个树都不为空，若值不同，也为不同
        if (c.val != t.val):
            return False
        # 上面的情况都不符合时，继续向下检查
        return self.is_same(c.left, t.left) and self.is_same(c.right, t.right)


114. 二叉树展开为链表
    1
   / \
  2   5
 / \   \
3   4   6

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6


# 非递归
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root:
            if root.left:
                pre_node = root.left
                # 同样先找到左子树的最右节点
                while pre_node.right:
                    pre_node = pre_node.right
                # 最右节点指向根节点的右子树
                pre_node.right = root.right
                # 根的右子树指向根的左子树，同时置空左子树
                root.right = root.left
                root.left = None
            root = root.right



剑指 Offer 62. 圆圈中最后剩下的数字
445. 两数相加 II
两数相加 II
给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。
你可以假设除了数字 0 之外，这两个数字都不会以零开头。
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[8,0,7]

输入：l1 = [0], l2 = [0]
输出：[0]

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverse(self, head):   # 链表反转
        pre = head
        cur = head.next
        pre.next = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)

        h = ListNode((l1.val + l2.val) % 10)   # 额外记下头结点，方便后续反转链表
        flag = (l1.val + l2.val) // 10
        p = h
        l1 = l1.next
        l2 = l2.next
        while l1 or l2 or flag:
            if l1 and l2:
                node = ListNode((l1.val + l2.val + flag) % 10)
                flag = (l1.val + l2.val + flag) // 10
                l1 = l1.next
                l2 = l2.next
            elif l1:
                node = ListNode((l1.val + flag) % 10)
                flag = (l1.val + flag) // 10
                l1 = l1.next
            elif l2:
                node = ListNode((l2.val + flag) % 10)
                flag = (l2.val + flag) // 10
                l2 = l2.next
            elif flag:
                node = ListNode(flag)
                flag = 0
            p.next = node
            p = node

        return self.reverse(h)





295. 数据流的中位数
剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
9. 回文数
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

例如，121 是回文，而 123 不是
输入：x = 121
输出：true

输入：x = -121
输出：false
解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。

输入：x = 10
输出：false
解释：从右向左读, 为 01 。因此它不是一个回文数
def isPalindrome(x: int) -> bool:
    """不满足进阶要求"""
    if x < 0:
        return False
    if 0 <= x <= 9:
        return True
    if x % 10 == 0:
        return False
    rev_x = int(''.join(list(str(x))[::-1]))
    if rev_x == x:
        return True
    else:
        return False




'''