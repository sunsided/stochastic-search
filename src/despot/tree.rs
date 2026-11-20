//! DESPOT's sparse belief tree: nodes with their scenarios and bounds, and the per-action
//! branches between them.

use alloc::vec::Vec;

/// Marks a node whose action branches are not created yet.
pub(super) const UNEXPANDED: u32 = u32::MAX;

/// The id of the root node.
pub(super) const ROOT: u32 = 0;

/// A belief node `b` of the tree.
///
/// Bound names follow Ye et al. (2017): `upper` is `U(b)` and `default_value` is `L0(b)`,
/// both averages over the node's scenarios of the discounted return from `b`; `mu` and `ell`
/// bound the regularized weighted discounted utility (RWDU) of the subtree, and already
/// carry the weight `|Phi_b| / K * gamma^depth`.
#[derive(Debug, Clone)]
pub(super) struct Node<S> {
    /// The depth `Delta(b)`: steps from the root.
    pub(super) depth: usize,
    /// The parent node; `None` at the root.
    pub(super) parent: Option<u32>,
    /// The scenarios passing through this node, with their states here. Never empty.
    pub(super) scenarios: Vec<(u32, S)>,
    /// `U(b)`: an upper bound on the scenarios' average return from here. At least
    /// `default_value`.
    pub(super) upper: f64,
    /// `L0(b)`: the default policy's average return from here over the scenarios.
    pub(super) default_value: f64,
    /// `l0(b) = |Phi_b| / K * gamma^depth * L0(b)`: the RWDU of following the default policy
    /// from here.
    pub(super) default_utility: f64,
    /// `mu(b)`: the upper bound on the best RWDU of the subtree. Never below `ell`.
    pub(super) mu: f64,
    /// `ell(b)`: the lower bound on the best RWDU of the subtree. Never below
    /// `default_utility`.
    pub(super) ell: f64,
    /// The index of the node's first branch; branches of all actions follow contiguously.
    /// [`UNEXPANDED`] until expanded.
    pub(super) first_branch: u32,
    /// Set by `MakeDefault`: the node is fixed to the default policy and never expanded or
    /// backed up from children again.
    pub(super) fixed: bool,
}

/// The branch of one action below a node.
#[derive(Debug, Clone)]
pub(super) struct Branch<O> {
    /// The summed reward of the action over the parent's scenarios, undiscounted.
    pub(super) reward_sum: f64,
    /// The child nodes, one per observation the scenarios produced. Scenarios whose step
    /// ended the episode have no child.
    pub(super) children: Vec<(O, u32)>,
}

/// The nodes and branches in two arenas.
#[derive(Debug, Clone)]
pub(super) struct Tree<S, O> {
    /// The nodes; [`ROOT`] first.
    pub(super) nodes: Vec<Node<S>>,
    /// The branches, grouped per node.
    pub(super) branches: Vec<Branch<O>>,
}

impl<S, O> Tree<S, O> {
    /// The node `id`.
    pub(super) fn node(&self, id: u32) -> &Node<S> {
        &self.nodes[id as usize]
    }

    /// The node `id`, mutably.
    pub(super) fn node_mut(&mut self, id: u32) -> &mut Node<S> {
        &mut self.nodes[id as usize]
    }

    /// The branches of the expanded node `id`, `actions` of them.
    pub(super) fn branches_of(&self, id: u32, actions: usize) -> &[Branch<O>] {
        let first = self.node(id).first_branch as usize;

        &self.branches[first..first + actions]
    }

    /// Appends `node` and returns its id.
    ///
    /// # Panics
    ///
    /// Panics if the tree outgrows `u32` ids.
    pub(super) fn push(&mut self, node: Node<S>) -> u32 {
        let id = u32::try_from(self.nodes.len()).expect("at most u32::MAX nodes");
        self.nodes.push(node);

        id
    }

    /// Sets `id` to the default policy (`MakeDefault`): `U = L0`, `mu = ell = l0`.
    pub(super) fn make_default(&mut self, id: u32) {
        let node = self.node_mut(id);
        node.upper = node.default_value;
        node.mu = node.default_utility;
        node.ell = node.default_utility;
        node.fixed = true;
    }
}
