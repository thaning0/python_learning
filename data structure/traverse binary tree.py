# def binary tree node
class TreeNode():
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
    
    def preorder_rec(self,root):
        if root == None:
            return
        print(root.val)
        self.preorder_rec(root.left)
        self.preorder_rec(root.right)  

    def preorder_ite(self,root):
        if not root:
            return
        stack = [root]
        preorder_list = []
        while stack:
            node = stack.pop()
            if node:
                preorder_list.append(node.val)
                stack.extend([node.right,node.left])
        return preorder_list


    def inorder_rec(self,root):
        if root==None:
            return
        self.inorder_rec(root.left)
        print(root.val)
        self.inorder_rec(root.right)

    def inorder_ite(self,root):
        if root == None:
            return
        stack = []
        node = root
        inorder_list=[]
        while node or stack:
            while node:
                stack.append(node)
                node=node.left
            node = stack.pop()
            inorder_list.append(node.val)
            node=node.right
        return inorder_list

    def postorder_rec(self,root):
        if root==None:
            return
        self.postorder_rec(root.left)
        self.postorder_rec(root.right)
        print(root.val)

    def postorder_ite(self,root):
        if root==None:
            return
        stack_in=[root]
        stack_out=[]
        postorder_list=[]
        while stack_in:
            node = stack_in.pop()
            if node.left:
                stack_in.append(node.left)
            if node.right:
                stack_in.append(node.right)
            stack_out.append(node)
        while stack_out:
            node = stack_out.pop()
            postorder_list.append(node.val)
        return postorder_list

    def levelorder(self,root):
        if root==None:
            return
        queue = [root]
        node=root
        leveloroder_list=[]
        while queue:
            node = queue.pop(0)
            leveloroder_list.append(node.val)
            if node.left != None:
                queue.append(node.left )
            if node.right != None:
                queue.append(node.right)
        return leveloroder_list
# an example
#             1
#          /     \
#         2       3
#       /   \   /   \
#      4    5   6    7
root = TreeNode(1)
node_2 = TreeNode(2)
node_3 = TreeNode(3)
node_4 = TreeNode(4)
node_5 = TreeNode(5)
node_6 = TreeNode(6)
node_7 = TreeNode(7)

root.left = node_2
root.right = node_3
node_2.left = node_4
node_2.right = node_5
node_3.left = node_6
node_3.right = node_7 

print(root.preorder_ite(root)) # 1->2->4->5->3->6->7
root.inorder_ite(root) # 4->2->5->1->6->3->7
root.postorder_ite(root) # 4->5->2->6->7->3->1
root.levelorder(root) # 1->2->3->4->5->6->7
