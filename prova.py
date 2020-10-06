import matplotlib.pyplot as plt
import numpy as np
import random
import math


def increase_red(img):
    img[:, :, 0] = img[:, :, 0] + 0.01
    return img
def decrease_red(img):
    img[:, :, 0] = img[:, :, 0] + 0.01
    return img

def increase_green(img):
    img[:, :, 1] = img[:, :, 1] + 0.01
    return img
def decrease_green(img):
    img[:, :, 1] = img[:, :, 1] + 0.01
    return img

def increase_blue(img):
    img[:, :, 2] = img[:, :, 2] + 0.01
    return img
def decrease_blue(img):
    img[:, :, 2] = img[:, :, 2] + 0.01
    return img

def operations():
    op = [increase_red, decrease_red, increase_green, decrease_green, increase_blue, decrease_blue]
    return op

def over(img):
    original_img = plt.imread('img.png')
    distance_r = abs(original_img[:, :, 0] - img[:, :, 0])
    distance_g = abs(original_img[:, :, 1] - img[:, :, 1])
    distance_b = abs(original_img[:, :, 2] - img[:, :, 2])
    total_dist = np.sum(distance_r) + np.sum(distance_g) + np.sum(distance_b)
    print(total_dist)
    return total_dist < 20000

def good_op(img):
    original_img = plt.imread('img.png')
    altered_img = plt.imread('im2.png')
    distance_r1 = abs(original_img[:, :, 0] - img[:, :, 0])
    distance_g1 = abs(original_img[:, :, 1] - img[:, :, 1])
    distance_b1 = abs(original_img[:, :, 2] - img[:, :, 2])
    distance_r2 = abs(original_img[:, :, 0] - altered_img[:, :, 0])
    distance_g2 = abs(original_img[:, :, 1] - altered_img[:, :, 1])
    distance_b2 = abs(original_img[:, :, 2] - altered_img[:, :, 2])
    total_dist1 = np.sum(distance_r1) + np.sum(distance_g1) + np.sum(distance_b1)
    total_dist2 = np.sum(distance_r2) + np.sum(distance_g2) + np.sum(distance_b2)
    return total_dist1 < total_dist2


class Node:
    """A node in the MCTS search tree."""
    def __init__(self, state, parent):
        """New node for the given parent node."""
        self.state = state
        self.parent = parent
        self.visits = 0
        self.total = 0
        if over(state):
            self.children = {}
        else:
            self.children = {op: None for op in operations()}

    def terminal(self):
        """True if the game is over (win or draw)."""
        return not self.children

    def expandable(self):
        """True if any child node is unexplored."""
        return any(n is None for n in self.children.values())

    def expand(self):
        """Expand one child node."""
        m = random.choice([k for k, v in self.children.items() if v is None])
        new_state = m(self.state)
        self.children[m] = Node(new_state, self)
        return self.children[m]

    def score(self):
        """Average score."""
        return self.total / max(1, self.visits)

    def ucb1(self):
        """Exploration criterion."""
        if self.visits == 0:
            return float("inf")
        exploration = 2 * math.sqrt(math.log(self.parent.visits) / self.visits)
        exploitation = self.score()
        return exploitation + exploration

    def choose_child(self):
        """Select the child to visit."""
        return max(self.children.values(), key=Node.ucb1)

    def backprop(self, value):
        """Propagate back the score obtained."""
        cur = self
        while cur is not None:
            cur.visits += 1
            cur.total += value
            cur = cur.parent


class Tree:
    """MCTS search tree."""
    def __init__(self, st=plt.imread('im2.png')):
        self.root = Node(st, None)

    def eval_moves(self):
        """Return a dict of operation/score pairs for the root position."""
        return {k: 1 - v.score() for (k, v) in self.root.children.items()}

    def search_step(self):
        """A single MCTS search step."""
        # 1) Select the node to expand.
        cur = self.root
        while not cur.expandable():
            if cur.terminal():
                if good_op(cur.state):
                    # the corrupted image is fixed.
                    cur.backprop(1)
                return
            cur = cur.choose_child()
        # 2) Create the new node.
        newnode = cur.expand()
        # 3) Perform a simulation.
        v = self.rollout(newnode)
        # 4) Propagate the result of the simulation.
        newnode.backprop(v)

    def rollout(self, node):
        """Make a random simulation from the given state."""
        im = node.state
        for i in range(100):
            if good_op(im):
                return 1
            im = random.choice(operations())(im)
        return 0

    def dump(self, file, node=None, moves=[], maxlevel=5):
        """Write to file a representation of the search tree."""
        if len(moves) == maxlevel:
            return
        if not moves:
            node = self.root
        prefix = " ".join(map(str, moves))
        if node is None:
            print(prefix, "...", "?", file=file)
            return
        val = (float("nan") if node.visits == 0 else node.total / node.visits)
        print(prefix, "...", node.visits, val, file=file)
        for m, c in node.children.items():
            self.dump(file, node=c, moves=moves + [m])

    def search(self, steps):
        """Perform the given number of search steps."""
        for _ in range(steps):
            self.search_step()

    def make_move(self, op, im):
        """Move down the root by one position."""
        assert op in self.root.children
        child = self.root.children[op]
        if child is None:
            child = Node(im, None)
        else:
            child.parent = None
        self.root = child

def play_tree(epochs):
    tree = Tree()
    for _ in range(epochs):
        op = random.choice(operations())
        if op is None:
            break
        tree.make_move(op, tree.root.state)
        if over(tree.root.state):
            print(_)
            print("Over")
            plt.imshow(tree.root.state)
            plt.show()
            break
        tree.search_step()
        #cur_time = time.time()
        evals = tree.eval_moves()
        evals = sorted(evals.items(), key=lambda x: -x[1])
        print()
        for k, v in evals:
            print(f"Operation {k}:  score {v:.2f}")
        tree.make_move(evals[0][0])

if __name__ == '__main__':
    im = plt.imread('img.png')
    im2 = plt.imread('im2.png')
    play_tree(100)