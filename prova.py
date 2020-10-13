import matplotlib.pyplot as plt
import numpy as np
import random
import math


def increase_red(img):
    """
    Increases the red channel of the input image.
    """
    img[:, :, 0] = img[:, :, 0] + 0.01
    return img
def decrease_red(img):
    """
    Decreases the red channel of the input image.
    """
    img[:, :, 0] = img[:, :, 0] - 0.01
    return img

def increase_green(img):
    """
    Increases the green channel of the input image.
    """
    img[:, :, 1] = img[:, :, 1] + 0.01
    return img
def decrease_green(img):
    """
    Decreases the green channel of the input image.
    """
    img[:, :, 1] = img[:, :, 1] - 0.01
    return img

def increase_blue(img):
    """
    Increases the blue channel of the input image.
    """
    img[:, :, 2] = img[:, :, 2] + 0.01
    return img
def decrease_blue(img):
    """
    Decreases the blue channel of the input image.
    """
    img[:, :, 2] = img[:, :, 2] - 0.01
    return img

def operations():
    """
    Returns a list of functions, relative to the operation that can be performed on an image.
    """
    op = [increase_red, decrease_red, increase_green, decrease_green, increase_blue, decrease_blue]
    return op

def distance(img):
    """
    Given a target image X and its corrupted version X', performs ||X - X'||^2
    """
    original_img = plt.imread('img.png')
    distance_r = abs(original_img[:, :, 0] - img[:, :, 0])
    distance_g = abs(original_img[:, :, 1] - img[:, :, 1])
    distance_b = abs(original_img[:, :, 2] - img[:, :, 2])
    total_dist = np.sum(distance_r) + np.sum(distance_g) + np.sum(distance_b)
    #print(pow(total_dist, 2))
    return pow(total_dist, 2)

def over(img):
    """
    Checks if the input image is very similar to the target image.
    """
    original_img = plt.imread('img.png')
    distance_r = abs(original_img[:, :, 0] - img[:, :, 0])
    distance_g = abs(original_img[:, :, 1] - img[:, :, 1])
    distance_b = abs(original_img[:, :, 2] - img[:, :, 2])
    total_dist = np.sum(distance_r) + np.sum(distance_g) + np.sum(distance_b)
    #print(total_dist)
    return total_dist < 10000

def orderOfMag(x):
    """
    Returns the order of magnitude of the input value
    """
    oom = 1
    while x >= 10:
        oom = oom*10
        x = x/10
    return oom

def alpha():
    """
    Returns the alpha coefficient
    """
    #return 1/(orderOfMag(n))
    return -0.00000000001

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
        return {k: v.score() for (k, v) in self.root.children.items()}

    def search_step(self):
        """A single MCTS search step."""
        # 1) Select the node to expand.
        cur = self.root
        while not cur.expandable():
            if cur.terminal():
                if over(cur.state):
                    # the corrupted image is fixed.
                    cur.backprop(1)
                else:
                    cur.backprop(np.exp(alpha()*(distance(cur.state))))
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
        new_op = None
        #plt.ion()
        for i in range(30):
            if over(im):
                return 1
            new_op = random.choice(operations())
            im = new_op(im)
            #plt.clf()
            #plt.imshow(im)
            #plt.pause(0.01)
        #plt.ioff()
        #print(distance(im))
        print(new_op, np.exp(alpha()*(distance(im))))
        return np.exp(alpha()*(distance(im)))

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
            child = Node(op(im), None)
        else:
            child.parent = None
        self.root = child

def play_tree(epochs):
    orig_im = plt.imread('img.png')
    corr_im = plt.imread('im2.png')
    tree = Tree()
    for _ in range(epochs):
        op = random.choice(operations())
        tree.make_move(op, tree.root.state)
        if over(tree.root.state):
            print(_)
            print("Image recovered")
            plt.figure(1)
            plt.imshow(orig_im)
            plt.figure(2)
            plt.imshow(tree.root.state)
            plt.figure(3)
            plt.imshow(corr_im)
            plt.show()
            break
        plt.ion()
        for i in range(40):
            if over(tree.root.state):
                break
            tree.search_step()
            tree.root.state = tree.root.state.clip(0,1)
            plt.clf()
            plt.imshow(tree.root.state)
            plt.pause(0.01)
        plt.ioff()
        evals = tree.eval_moves()
        evals = sorted(evals.items(), key=lambda x: -x[1])
        print(evals[0][0])
        print()
        for k, v in evals:
            print(f"Operation {k}:  score {v:.5f}")
        tree.make_move(evals[0][0], tree.root.state)
        if over(tree.root.state):
            print(_)
            print("Image recovered")
            plt.figure(1)
            plt.imshow(orig_im)
            plt.figure(2)
            plt.imshow(tree.root.state)
            plt.figure(3)
            plt.imshow(corr_im)
            plt.show()
            break


if __name__ == '__main__':
    im = plt.imread('img.png')
    im2 = plt.imread('im2.png')
    play_tree(100)