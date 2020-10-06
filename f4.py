import random
import math
import matplotlib
import matplotlib.pyplot as plt
import collections
import sys
import time


# Bitboard representation: first player stones, second player stones,
# side to move (0 or 1)
Position = collections.namedtuple("Position", ["p0", "p1", "stm"])


def _all_winning():
    """Winning bitboards (four consecutive positions). """
    b = 15
    for r in range(6):
        for c in range(4):
            yield b << (c + 7 * r)
    b = 1 | (1 << 7) | (1 << 14) | (1 << 21)
    for r in range(3):
        for c in range(7):
            yield b << (c + 7 * r)
    b = 1 | (1 << 8) | (1 << 16) | (1 << 24)
    for r in range(3):
        for c in range(4):
            yield b << (c + 7 * r)
    b = (1 << 3) | (1 << 9) | (1 << 15) | (1 << 21)
    for r in range(3):
        for c in range(4):
            yield b << (c + 7 * r)


WINNING = list(_all_winning())
FULL = (1 << 42) - 1


def make_move(pos, c):
    """Return the position pos after move c."""
    occ = pos[0] | pos[1]
    prev = None
    for i in range(35 + c, -1, -7):
        b = 1 << i
        if occ & b:
            break
        prev = b
    if prev is None:
        raise ValueError("Invalid move")
    if pos.stm == 0:
        return Position(pos[0] | prev, pos[1], 1)
    else:
        return Position(pos[0], pos[1] | prev, 0)


def full(pos):
    """Tell if the board is full."""
    return ((pos[0] | pos[1]) == FULL)


def won(pos):
    """Tell if last move was a win."""
    p = pos[1 - pos.stm]
    return any(p & c == c for c in WINNING)


def valid_moves(pos):
    """Return the list of valid moves in pos."""
    occ = pos[0] | pos[1]
    b = 1 << 35
    return [c for c in range(7) if not (occ & (b << c))]


def pos_to_coordinates(pos):
    """Used to draw the position."""
    xs = set((k // 7, k % 7) for k in range(42) if (1 << k) & pos[0])
    os = set((k // 7, k % 7) for k in range(42) if (1 << k) & pos[1])
    return xs, os


def draw_pos(pos):
    """Graphically draw the position."""
    xs, os = pos_to_coordinates(pos)
    plt.clf()
    rect = matplotlib.patches.Rectangle((-0.5,-0.5), 7 , 6,linewidth=1,edgecolor='r',facecolor='b')
    plt.gca().add_patch(rect)
    es = [(r, c) for r in range(6) for c in range(7)]
    plt.plot([p[1] for p in es], [p[0] for p in es], "w.", ms=70)
    plt.plot([p[1] for p in xs], [p[0] for p in xs], "r.", ms=70)
    plt.plot([p[1] for p in os], [p[0] for p in os], "y.", ms=70)
    choice = None
    def onclick(event):
        if event.button == 1:
            move = int(event.xdata + 0.5)
            if move in valid_moves(pos):
                nonlocal choice
                choice = move
                plt.close()
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return choice


class Node:
    """A node in the MCTS search tree."""
    def __init__(self, pos, parent):
        """New node for the given position and parent node."""
        self.pos = pos
        self.parent = parent
        self.visits = 0
        self.total = 0
        if won(pos) or full(pos):
            self.children = {}
        else:
            self.children = {m: None for m in valid_moves(pos)}

    def terminal(self):
        """True if the game is over (win or draw)."""
        return not self.children

    def expandable(self):
        """True if any child node is unexplored."""
        return any(n is None for n in self.children.values())

    def expand(self):
        """Expand one child node."""
        m = random.choice([k for k, v in self.children.items() if v is None])
        newpos = make_move(self.pos, m)
        self.children[m] = Node(newpos, self)
        return self.children[m]

    def score(self):
        """Average score."""
        return self.total / max(1, self.visits)

    def ucb1(self):
        """Exploration criterion."""
        if self.visits == 0:
            return float("inf")
        exploration = 2 * math.sqrt(math.log(self.parent.visits) / self.visits)
        # ucb1 will be invoked at the parent level to choose the next
        # move, so the score needs to be inverted because good for
        # child ==> bad for parent.
        exploitation = 1 - self.score()
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
            # In two-player zero-sum games, the value for the parent
            # is the opposite of that for the child
            value = 1 - value


class Tree:
    """MCTS search tree."""
    def __init__(self, pos=Position(0, 0, 0)):
        self.root = Node(pos, None)

    def eval_moves(self):
        """Return a dict of move/score pairs for the root position."""
        return {k: 1 - v.score() for (k, v) in self.root.children.items()}

    def search_step(self):
        """A single MCTS search step."""
        # 1) Select the node to expand.
        cur = self.root
        while not cur.expandable():
            if cur.terminal():
                if won(cur.pos):
                    # Game lost.
                    cur.backprop(0)
                else:
                    # Game drawn.
                    cur.backprop(0.5)
                return
            cur = cur.choose_child()
        # 2) Create the new node.
        newnode = cur.expand()
        # 3) Perform a simulation.
        v = self.rollout(newnode)
        # 4) Propagate the result of the simulation.
        newnode.backprop(v)

    def rollout(self, node):
        """Make a random simulation from the given position."""
        pos = node.pos
        v = 0
        while True:
            if won(pos):
                # Game lost.
                return v
            elif full(pos):
                # Game drawn.
                return 0.5
            m = random.choice(valid_moves(pos))
            pos = make_move(pos, m)
            v = 1 - v

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

    def make_move(self, m):
        """Move down the root by one position."""
        assert m in self.root.children
        child = self.root.children[m]
        if child is None:
            child = Node(make_move(self.root.pos, m), None)
        else:
            child.parent = None
        self.root = child


def play_humans():
    pos = Position(0, 0, 0)
    while True:
        print(pos)
        m = draw_pos(pos)
        if m is None:
            break
        pos = make_move(pos, m)
        if won(pos) or full(pos):
            break


def play_tree(seconds, player):
    tree = Tree()
    if player == 1:
        tree.make_move(3)
    while True:
        m = draw_pos(tree.root.pos)
        if m is None:
            break
        tree.make_move(m)
        if won(tree.root.pos):
            print("You won!")
            break
        if full(tree.root.pos):
            print("Draw.")
            break
        start_time = cur_time = time.time()
        while cur_time - start_time < seconds:
            tree.search_step()
            cur_time = time.time()
        evals = tree.eval_moves()
        evals = sorted(evals.items(), key=lambda x: -x[1])
        print()
        for k, v in evals:
            print(f"Move {k}:  score {v:.2f}")
        tree.make_move(evals[0][0])
        if won(tree.root.pos):
            print("You lost!")
            break
        if full(tree.root.pos):
            print("Draw.")
            break


if __name__ == "__main__":
    # play_humans()
    seconds = float(sys.argv[1] if len(sys.argv) > 1 else 1.0)
    color = (sys.argv[2] if len(sys.argv) > 2 else random.choice(["red", "yellow"]))
    player = {"red": 0, "yellow": 1}[color]
    play_tree(seconds, player)
    # tree = Tree()
    # v = tree.rollout(tree.root)
    # search_move(Position(67118124, 661584, 0))