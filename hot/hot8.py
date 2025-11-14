'''
442. 数组中重复的数据
给你一个长度为 n 的整数数组 nums ，其中 nums 的所有整数都在范围 [1, n] 内，且每个整数出现 一次 或 两次 。请你找出所有出现 两次 的整数，并以数组形式返回。
你必须设计并实现一个时间复杂度为 O(n) 且仅使用常量额外空间的算法解决此问题。


class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        dic = collections.Counter(nums)

        res = []
        for key, value in dic.items():
            if value == 2:
                res.append(key)

        return res


class Solution(object):

    def findDuplicates(self, nums):
        """
        主要的思路就是利用正负来保存第一次遍历到的数值的状态
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for i in range(len(nums)):
            index = nums[i] - 1  # 取到以这个值-1所在索引的值，判断这个值是否大于0
            if nums[index] > 0:  # 如果大于0，则说明这个值是第一次被遍历到，
                nums[index] = -nums[index]  # 则将这个值-1所在索引的值变负数，通过这个来记住已经遍历过一遍了
            else:  # 当这个值是负数，表示这个值第二次被遍历到了，所以它是重复的，故添加到res中即可
                res.append(nums[i])
        return res




10. 正则表达式匹配
329. 矩阵中的最长递增路径
347. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
示例：
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        result = []
        d = collections.Counter(nums).most_common(k)
        for e in d:
            result.append(e[0])
        return result

287. 寻找重复数
给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
输入：nums = [1,3,4,2,2]
输出：2
输入：nums = [3,1,3,4,2]
输出：3

class Solution(object):
    def findDuplicate(self, nums):
        if not nums:
            return None
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                return nums[i]
                break




887. 鸡蛋掉落
189. 轮转数组
给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释:
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        nums[:] = nums[n - k:] + nums[:n - k]



120. 三角形最小路径和
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
例如，给定三角形：
[
 [2],
 [3,4],
 [6,5,7],
 [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
解题误区
典型的动态规划问题，没有注意到题目相邻节点的信息。🤔智障的我用集合遍历整了半天，结果死在了这个用例：
输入：[[-1],[2,3],[1,-1,-3]]
输出：-2
预期：-1

题目的意思并不是直接求最小路径，隐含的时求在相邻节点的条件下，求最小路径
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[][] dp = new int[n+1][n+1];
        //倒数第二行，倒数第二列
        for(int i = n - 1; i >= 0; i--) {
            for(int j = triangle.get(i).size() - 1; j >= 0; j--) {
                dp[i][j] = Math.min(dp[i+1][j], dp[i+1][j+1]) + triangle.get(i).get(j);
            }
        }
        return dp[0][0];
    }
}





'''