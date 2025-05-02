import math
import chess
import torch
import collections
import numpy as np

from copy import copy, deepcopy
from typing import Callable, Tuple, Mapping, Iterable, Any

from core.model import ChessNet
from core.chess_base_v2 import ChessEnv
from core.coords_converter import ChessCoordsConverter

NUM_ACTIONS = 4864

class DummyNode(object):
    """A place holder to make computation possible for the root node."""

    def __init__(self):
        self.parent = None
        self.child_W = collections.defaultdict(float)
        self.child_N = collections.defaultdict(float)

class MCTSNode:
    def __init__(self, to_play: int, move: int = None, parent: Any = None) -> None:
        self.to_play = to_play
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.child_W = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.child_N = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.child_P = np.zeros(NUM_ACTIONS, dtype=np.float32)

        self.children: Mapping[int, MCTSNode] = dict()

        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))

    def child_Q(self):
        """Returns a 1D numpy.array contains mean action value for all child."""
        # Avoid division by zero
        child_N = np.where(self.child_N > 0, self.child_N, 1)

        return self.child_W / child_N

    @property
    def N(self):
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.move] = value

    @property
    def W(self):
        """The total value for current node is stored at parent's level."""
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.move] = value

    @property
    def Q(self):
        """Returns the mean action value Q(s, a)."""
        if self.parent.child_N[self.move] > 0:
            return self.parent.child_W[self.move] / self.parent.child_N[self.move]
        else:
            return 0.0

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, MCTSNode)
    
def best_child(node: MCTSNode, legal_actions: np.ndarray, c_puct_base: float, c_puct_init: float, child_to_play: int) -> MCTSNode:
    """Returns best child node with maximum action value Q plus an upper confidence bound U.
    And creates the selected best child node if not already exists.

    Args:
        node: the current node in the search tree.
        legal_actions: a 1D bool numpy.array mask for all actions,
                where `1` represents legal move and `0` represents illegal move.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        child_to_play: the player id for children nodes.

    Returns:
        The best child node corresponding to the UCT score.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')
    
    UCB_score = -node.child_Q() + node.child_U(c_puct_base=c_puct_base, c_puct_init=c_puct_init)
    UCB_score = np.where(legal_actions == 1, UCB_score, -9999)

    move = np.argmax(UCB_score)

    assert legal_actions[move] == 1

    if move not in node.children:
        node.children[move] = MCTSNode(to_play=child_to_play, move=move, parent=node)

    return node.children[move]

def _expand(node: MCTSNode, policy: np.ndarray) -> None:
    """Expand all legal actions.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for legal actions.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded: raise RuntimeError('This node has already expanded.')
    if not isinstance(policy, np.ndarray) or len(policy.shape) != 1 or policy.dtype != np.float32:
        raise ValueError(f'Expect `policy` to be a 1D float numpy.array, got {policy}')
    
    node.child_P = policy
    node.is_expanded = True

def _beckprop(node: MCTSNode, value: float) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value evaluated from current player's perspective.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """
    if not isinstance(value, float): 
        print(value)
        raise ValueError(f'Expect `value` to be a float type, got {type(value)}')
    
    while isinstance(node, MCTSNode):
        node.N += 1
        node.W += value
        node = node.parent
        value = -1 * value

def add_dirichlet_noise(node: MCTSNode, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.3) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """
    if not isinstance(node, MCTSNode) or not node.is_expanded:
        raise ValueError('Expect `node` to be expanded')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)

    node.child_P = node.child_P * (1 - eps) + noise * eps

def generate_search_policy(child_N: np.ndarray, temperature: float, legal_actions: np.ndarray) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        child_N: the visit number of the children nodes from the root node of the search tree.
        temperature: a parameter controls the level of exploration.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `temperature` is not float type or not in range (0.0, 1.0].
    """
    if not isinstance(temperature, float) or not 0 < temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range (0.0, 1.0], got {temperature}')
    
    child_N = legal_actions * child_N

    if temperature > 0:
        exp = max(1.0, min(5.0, 1.0 / temperature))
        child_N = np.power(child_N, exp)

    assert np.all(child_N >= 0) and not np.any(np.isnan(child_N))

    policy = child_N
    sums = np.sum(policy)
    if sums > 0: 
        policy /= sums
    
    return policy

def uct_search(env: ChessEnv, eval_func: Callable[[np.ndarray, np.ndarray], Tuple[Iterable[np.ndarray], Iterable[float]]],
               root_node: MCTSNode, c_puct_base: float, c_puct_init: float, num_sims: int = 800, root_noise: bool = False,
               warm_up: bool = False, deterministic: bool = False) -> Tuple[int, np.ndarray, float, float, MCTSNode]:
    if not isinstance(env, ChessEnv): 
        raise ValueError(f'Expect `env` to be a valid ChessEnv instance, got {env}')
    if not 1 <= num_sims:
        raise ValueError(f'Expect `num_sims` to a positive integer, got {num_sims}')
    if env._is_game_over():
        raise RuntimeError('Game is over.')
    
    if root_node is None:
        policy, value = eval_func(env._observation(), env.legal_actions)
        root_node = MCTSNode(to_play=int(env.to_play), parent=DummyNode())
        _expand(root_node, policy)
        _beckprop(root_node, value)

    assert root_node.to_play == int(env.to_play)

    root_legal_action = env.legal_actions
    converter = ChessCoordsConverter()

    if root_noise:
        add_dirichlet_noise(root_node, root_legal_action)

    while root_node.N < num_sims:
        node = root_node

        # Do not touch the actual environment
        sim_env = deepcopy(env)
        obs = sim_env._observation()
        done = sim_env._is_game_over()

        # 1. Select
        while node.is_expanded:
            node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, not sim_env.to_play)
            # Convert move index to chess move before stepping
            chess_move = converter.index_to_move(node.move)
            obs, reward, done = sim_env.step(chess_move)

            if done: break
        
        assert node.to_play == int(sim_env.to_play)

        if done:
            _beckprop(node, -reward)
            continue

        # 2. Expand and evaluation
        policy, value = eval_func(obs, sim_env.legal_actions)
        _expand(node, policy)

        # 3. Backprop 
        _beckprop(node, value)

    # Play - generate search policy action probability from the root node's child visit number.
    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1, root_legal_action)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        move = np.argmax(root_node.child_N)
    else:
        while move is None or root_legal_action[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    for move in root_node.children:
        next_root_node = root_node.children[move]

        N, W = copy(next_root_node.N), copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.move = None
        next_root_node.N = N
        next_root_node.W = W

        best_child_Q = -next_root_node.Q

    assert root_legal_action[move] == 1

    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)

def add_virtual_loss(node: MCTSNode) -> None:
    vloss = +1
    while isinstance(node, MCTSNode):
        node.losses_applied += 1
        node.W += vloss
        node = node.parent

def revert_virtual_loss(node: MCTSNode) -> None:
    vloss = -1
    while isinstance(node, MCTSNode):
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.W += vloss
        node = node.parent

def parallel_uct_search(env: ChessEnv, eval_func: Callable[[np.ndarray, np.ndarray], Tuple[Iterable[np.ndarray], Iterable[float]]],
                        root_node: MCTSNode, c_puct_base: float, c_puct_init: float, num_sims: int = 800, num_parallel: int = 8,
                        root_noise: bool = False, warm_up: bool = False, deterministic: bool = False) -> Tuple[int, np.ndarray, float, float, MCTSNode]:
    if not isinstance(env, ChessEnv): 
        raise ValueError(f'Expect `env` to be a valid ChessEnv instance, got {env}')
    if not 1 <= num_sims:
        raise ValueError(f'Expect `num_sims` to a positive integer, got {num_sims}')
    if env._is_game_over():
        raise RuntimeError('Game is over.')
    
    if root_node is None:
        policy, value = eval_func(env._observation(), env.legal_actions)
        root_node = MCTSNode(int(env.to_play), parent=DummyNode())
        _expand(root_node, policy)
        _beckprop(root_node, value)

    assert root_node.to_play == int(env.to_play)

    root_legal_actions = env.legal_actions
    converter = ChessCoordsConverter()

    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)

    while root_node.N < num_sims + num_parallel:
        leaves = []
        fail_safe = 0

        while len(leaves) < num_parallel and fail_safe < num_parallel * 2:
            fail_safe += 1
            node = root_node

            sim_env = deepcopy(env)
            obs = sim_env._observation()
            done = sim_env._is_game_over()

            # 1. Select
            while node.is_expanded:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, not sim_env.to_play)

                chess_move = converter.index_to_move(node.move)
                obs, reward, done = sim_env.step(chess_move)
                
                if done: break

            assert node.to_play == int(sim_env.to_play)

            if done:
                _beckprop(node, -reward)
                continue
            else:
                add_virtual_loss(node)
                leaves.append((node, obs, sim_env.legal_actions))

        if leaves:
            batched_nodes, batched_obs, batched_mask = map(list, zip(*leaves))
            policies, values = eval_func(batched_obs, batched_mask)

            print(len(batched_nodes), len(policies), len(values))

            for leaf, policy, value in zip(batched_nodes, policies, values):
                revert_virtual_loss(leaf)

                if leaf.is_expanded:
                    continue

                _expand(leaf, policy)
                _beckprop(leaf, value)

    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1, root_legal_actions)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic: 
        move = np.argmax(root_node.child_N)
    else:
        while move is None or root_legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    if move in root_node.children:
        next_root_node = root_node.children[move]

        N, W = copy(next_root_node.N), copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.move = None
        next_root_node.N = N
        next_root_node.W = W

        best_child_Q = -next_root_node.Q

    assert root_legal_actions[move] == 1

    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)