# function used to count the number of unique thresholds used by a tree produced by cart
def count_unique_splits_cart(tree):
    return len(set((x, th) for x, th in zip(tree.feature, tree.threshold) if x != -2))

# function used to count the number of unique thresholds used by a tree produced by DL8.5
def count_unique_splits_dl85(tree):
    def count_splits_recursive(sub_tree, nodes):
        if "feat" not in sub_tree:
            return set()

        nodes.add(sub_tree["feat"])
        nodes.union(count_splits_recursive(sub_tree["left"], nodes))
        nodes.union(count_splits_recursive(sub_tree["right"], nodes))
        
        return nodes

    return len(count_splits_recursive(tree, set()))