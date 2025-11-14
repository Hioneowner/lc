'''
862. 和至少为 K 的最短子数组
要求是连续的子数组，所以自然就想到了双指针法，也就是寻找一个连续的区间。但是细思之后没有理清左右边界移动的逻辑，遂弃。
暴力解
反正先暴力解一下吧：
前缀和数组：构建数组sum[i]表示原数组前 i 个元素之和，比如sum[2] = A[0] + A[1] 。

例子3：
Input: A = [2,-1,2], K = 3
Output: 3
数组A对应的前缀和数组:
sum = [0, 2, 1, 3]










617. 合并二叉树
给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        # 递归，整体是怎么执行的，里边就怎么执行
        if not root1: return root2
        if not root2: return root1
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left,root2.left)
        root.right = self.mergeTrees(root1.right,root2.right)
        return root





剑指 Offer 07. 重建二叉树
题目描述
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。(事实上，返回的是二叉树的根节点）

LeetCode
类比LeetCode题目有105. 从前序与中序遍历序列构造二叉树
思路
前序 [1,2,4,7,3,5,6,8]
后序 [4,7,2,15,3,8,6]

前序遍历的第一个是根节点1，扫描中序遍历结果，就可以找到根节点1的位置。在中序遍历中，根节点1左边的数字位于左子树，1右边的节点位于右子树。这样就可以划分出根节点和对应的孩子。
同样地，在左子树和右子树当中，前序遍历的第一个节点是根节点。可以用上面同样的方法去构建，这样也就使用递归的方法来构建二叉树。
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        #递归终止条件 任何一种遍历为空
        if not preorder or not inorder:
            return None
        #前序第一个为root
        root = TreeNode(preorder[0])
        #找到中序遍历中root的位置
        index = inorder.index(preorder[0])
        # 递归调用 求左子树 右子树
        # 左子树为 前序中根节点后一个后面index个数 中序中从开始到根节点
        root.left = self.buildTree(preorder[1:index+1], inorder[:index])
        # 右子树为 前序中从index+1后面的 中序从根节点后一个开始到最后
        root.right = self.buildTree(preorder[index+1:], inorder[index+1:])
        return root



扩展2 前序后序建立二叉树
需要注意的是：前序和后序并不能够唯一确定一颗二叉树。
其先序序列为： 1 2 3 4 6 7 5
后序序列为：2 6 7 4 5 3 1
def constructFromPrePost(self, pre, post):
    if not pre: return None
    root = TreeNode(pre[0])
    if len(pre) == 1: return root

    L = post.index(pre[1]) + 1
    root.left = self.constructFromPrePost(pre[1:L+1], post[:L])
    root.right = self.constructFromPrePost(pre[L+1:], post[L:-1])
    return root





17. 电话号码的字母组合


706. 设计哈希映射
使用任何内建的哈希表库设计一个哈希映射

具体地说，你的设计应该包含以下的功能

put(key, value)：向哈希映射中插入(键,值)的数值对。如果键对应的值已经存在，更新这个值。
get(key)：返回给定的键所对应的值，如果映射中不包含这个键，返回-1。
remove(key)：如果映射中存在这个键，删除这个数值对。

示例：

MyHashMap hashMap = new MyHashMap();
hashMap.put(1, 1);
hashMap.put(2, 2);
hashMap.get(1); // 返回 1
hashMap.get(3); // 返回 -1 (未找到)
hashMap.put(2, 1); // 更新已有的值
hashMap.get(2); // 返回 1
hashMap.remove(2); // 删除键为2的数据
hashMap.get(2); // 返回 -1 (未找到)

注意：

所有的值都在 [1, 1000000]的范围内。
操作的总数目在[1, 10000]范围内。
不要使用内建的哈希库。
思路：

给定了值的范围，所以可以用桶排序的思想实现hashmap。
class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashmap = [-99999 for _ in range(1000005)]

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        self.hashmap[key] = value

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        if self.hashmap[key] != -99999:
            return self.hashmap[key]
        return -1

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        self.hashmap[key] = -99999




1095. 山脉数组中查找目标值
给你一个 山脉数组 mountainArr，请你返回能够使得 mountainArr.get(index) 等于 target 最小 的下标 index 值。

如果不存在这样的下标 index，就请返回 -1。
何为山脉数组？如果数组 A 是一个山脉数组的话，那它满足如下条件：

首先，A.length >= 3
其次，在 0 < i < A.length - 1 条件下，存在 i 使得：
A[0] < A[1] < … A[i-1] < A[i]
A[i] > A[i+1] > … > A[A.length - 1]
你将 不能直接访问该山脉数组，必须通过 MountainArray 接口来获取数据：

MountainArray.get(k) - 会返回数组中索引为k 的元素（下标从 0 开始）
MountainArray.length() - 会返回该数组的长度

输入：array = [1,2,3,4,5,3,1], target = 3
输出：2
解释：3 在数组中出现了两次，下标分别为 2 和 5，我们返回最小的下标 2。


#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
	#先找峰值，然后二分搜索左边，如果有直接返回结果，如果返回-1，再搜右边，直接返回结果即可
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        mid = self.findInMountainTop(mountain_arr)
        l_idx = self.left_index(mid, mountain_arr,target)
        if l_idx==-1:
            return self.right_index(mid, mountain_arr,target)
        else:
            return l_idx

    #二分搜索峰值的下标位置
    def findInMountainTop(self,mountain_arr):
        l = 0
        r = mountain_arr.length()-1
        while(l<r):
            mid = (l+r)//2
            if mountain_arr.get(mid)<mountain_arr.get(mid+1):
                l = mid+1
            else:
                r = mid
        return l
	 #二分搜索左边空间函数
    def left_index(self, mid, mountain_arr,target):
        l = 0
        r = mid
        while(l<r):
            mid = (l+r)//2
            if mountain_arr.get(mid)==target:
                return mid
            elif mountain_arr.get(mid)>target:
                r = mid
            else:
                l = mid+1
        return -1
    #二分搜索右边空间函数
    def right_index(self,mid,mountain_arr,target):
        l = mid
        r = mountain_arr.length()
        while(l<r):
            mid = (l+r)//2
            if mountain_arr.get(mid)==target:
                return mid
            elif mountain_arr.get(mid)<target:
                r = mid
            else:
                l = mid+1
        return -1


547. 省份数量（原朋友圈）
题目：

有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。
————————————————
版权声明：本文为CSDN博主「mingchen_peng」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/mingchen_peng/article/details/141106520









107. 二叉树的层次遍历 II
给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

例如：

给定二叉树 [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7

返回其自底向上的层次遍历为：

[
  [15,7],
  [9,20],
  [3]
]
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque

class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        ans = []

        stack = []

        queue = deque()
        queue.append(root)

        while queue:
            cnt = len(queue)

            tmp = []

            for _ in range(cnt):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                tmp.append(node.val)

            stack.append(tmp)

        while stack:
            ans.append(stack.pop())

        return ans





'''