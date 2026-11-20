//! POMCP's search tree: history nodes, each with one action node per action, and action
//! nodes with one child history per observation seen.

use alloc::vec::Vec;

/// The id of the root history node.
pub(super) const ROOT: u32 = 0;

/// Marks a history node whose action nodes are not created yet.
const UNEXPANDED: u32 = u32::MAX;

/// A history: an action-observation sequence from the root belief.
#[derive(Debug, Clone)]
struct History {
    /// Simulations that passed through this history and chose an action here.
    visits: u64,
    /// The index of this history's first action node; its actions follow contiguously.
    /// [`UNEXPANDED`] until [`Tree::expand`].
    first_action: u32,
}

/// An action taken after a history.
#[derive(Debug, Clone)]
struct ActionNode<O> {
    /// Simulations that took this action here.
    visits: u64,
    /// The average discounted return of those simulations, from this history on.
    value: f64,
    /// The histories that follow, one per observation seen so far.
    // shortcut: linear scan with `==`, fine for small observation sets; switch to a hash map
    // keyed by observation if a domain has many distinct observations per action.
    children: Vec<(O, u32)>,
}

/// History and action nodes in two arenas.
#[derive(Debug, Clone)]
pub(super) struct Tree<O> {
    /// The number of actions per history.
    actions: usize,
    /// History nodes; [`ROOT`] first.
    histories: Vec<History>,
    /// Action nodes, grouped per history.
    action_nodes: Vec<ActionNode<O>>,
}

impl<O: PartialEq> Tree<O> {
    /// A tree with an expanded root, for `actions` actions per history (at least one).
    pub(super) fn new(actions: usize) -> Self {
        let mut tree = Self {
            actions,
            histories: Vec::new(),
            action_nodes: Vec::new(),
        };
        tree.add_history();
        tree.expand(ROOT);

        tree
    }

    /// The number of history nodes.
    pub(super) fn len(&self) -> usize {
        self.histories.len()
    }

    /// Whether `history` has its action nodes.
    pub(super) fn is_expanded(&self, history: u32) -> bool {
        self.histories[history as usize].first_action != UNEXPANDED
    }

    /// Creates the action nodes of `history`.
    ///
    /// # Panics
    ///
    /// Panics if the action arena outgrows `u32` indices.
    pub(super) fn expand(&mut self, history: u32) {
        let first = u32::try_from(self.action_nodes.len()).expect("at most u32::MAX action nodes");
        self.histories[history as usize].first_action = first;
        for _ in 0..self.actions {
            self.action_nodes.push(ActionNode {
                visits: 0,
                value: 0.0,
                children: Vec::new(),
            });
        }
    }

    /// The position (in emission order) of the action node `action` of `history`.
    pub(super) fn position(&self, history: u32, action: u32) -> usize {
        (action - self.histories[history as usize].first_action) as usize
    }

    /// The UCB1 choice at the expanded `history`: the first untried action, otherwise the
    /// highest `Q + c * sqrt(ln N / n)`, ties to the first. Counts the visit of `history`.
    pub(super) fn select(&mut self, history: u32, exploration: f64) -> u32 {
        let node = &mut self.histories[history as usize];
        node.visits += 1;
        let first = node.first_action;
        #[expect(
            clippy::cast_precision_loss,
            reason = "visit counts beyond 2^53 only lose precision in the exploration bonus"
        )]
        let log_visits = libm::log(node.visits as f64);

        let mut best = (first, f64::NEG_INFINITY);
        for (offset, entry) in self.action_nodes[first as usize..][..self.actions]
            .iter()
            .enumerate()
        {
            let index = first + u32::try_from(offset).expect("action positions fit u32");
            if entry.visits == 0 {
                return index;
            }

            #[expect(
                clippy::cast_precision_loss,
                reason = "visit counts beyond 2^53 only lose precision in the exploration bonus"
            )]
            let score = entry.value + exploration * libm::sqrt(log_visits / entry.visits as f64);
            if score > best.1 {
                best = (index, score);
            }
        }

        best.0
    }

    /// The child history of `action` for `observation`, added if new.
    ///
    /// # Panics
    ///
    /// Panics if the history arena outgrows `u32` ids.
    pub(super) fn child(&mut self, action: u32, observation: O) -> u32 {
        let children = &self.action_nodes[action as usize].children;
        if let Some(&(_, id)) = children.iter().find(|(seen, _)| *seen == observation) {
            return id;
        }

        let id = self.add_history();
        self.action_nodes[action as usize]
            .children
            .push((observation, id));

        id
    }

    /// Averages one more return into `action`.
    pub(super) fn record(&mut self, action: u32, value: f64) {
        let node = &mut self.action_nodes[action as usize];
        node.visits += 1;
        #[expect(
            clippy::cast_precision_loss,
            reason = "visit counts beyond 2^53 only lose precision in the running mean"
        )]
        let weight = 1.0 / node.visits as f64;
        node.value += (value - node.value) * weight;
    }

    /// The position and average return of the best tried root action, ties to the first
    /// emitted. The first action with value NaN if no simulation tried any.
    pub(super) fn best_root_action(&self) -> (usize, f64) {
        let first = self.histories[ROOT as usize].first_action as usize;
        let mut best = (0, f64::NAN);
        for (position, entry) in self.action_nodes[first..][..self.actions]
            .iter()
            .enumerate()
        {
            if entry.visits > 0 && (best.1.is_nan() || entry.value > best.1) {
                best = (position, entry.value);
            }
        }

        best
    }

    /// Appends an unexpanded history node and returns its id.
    fn add_history(&mut self) -> u32 {
        let id = u32::try_from(self.histories.len()).expect("at most u32::MAX histories");
        self.histories.push(History {
            visits: 0,
            first_action: UNEXPANDED,
        });

        id
    }
}

#[cfg(test)]
mod tests {
    use super::{ROOT, Tree};

    /// Untried actions come first, then the higher mean wins without exploration.
    #[test]
    fn select_tries_every_action_then_exploits() {
        let mut tree: Tree<u8> = Tree::new(2);

        let first = tree.select(ROOT, 0.0);
        tree.record(first, 1.0);
        let second = tree.select(ROOT, 0.0);
        assert_ne!(first, second);
        tree.record(second, 5.0);

        assert_eq!(tree.select(ROOT, 0.0), second);
        assert_eq!(tree.best_root_action(), (1, 5.0));
    }

    /// The same observation leads to the same child; a new one adds a history.
    #[test]
    fn children_are_shared_per_observation() {
        let mut tree: Tree<u8> = Tree::new(1);
        let action = tree.select(ROOT, 1.0);

        let a = tree.child(action, 7);
        assert_eq!(tree.child(action, 7), a);
        assert_ne!(tree.child(action, 8), a);
        assert_eq!(tree.len(), 3);
    }

    /// Without any simulation, the recommendation carries no value.
    #[test]
    fn untried_root_has_no_value() {
        let tree: Tree<u8> = Tree::new(3);
        let (position, value) = tree.best_root_action();

        assert_eq!(position, 0);
        assert!(value.is_nan());
    }
}
