//! Value sets and their operators: dominance, convex coverage set pruning, the bounded set
//! sum and the Hausdorff distance (Chen, Trevizan and Thiébaux 2023, Sections 2 and 3).

use alloc::vec::Vec;

mod planar;

use crate::mo::simplex::{dot, is_covered};

/// The action tag of a vector that no backup produced (a heuristic estimate, a goal's zero
/// vector, a dead end's bound).
pub(crate) const NO_ACTION: u32 = u32::MAX;

/// One cost vector of a value set, tagged with the action whose Q-value produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Point<const N: usize> {
    /// The expected cost vector; every component finite.
    pub(crate) vector: [f64; N],
    /// The action (an index into the action arena) whose Q-value contains the vector, or
    /// [`NO_ACTION`].
    pub(crate) action: u32,
}

/// A value: a set of cost vectors pruned to a convex coverage set, plus the paper's upper
/// bound `b` as a separate marker.
///
/// The paper keeps `b` inside the set (the operator `CS_B` never prunes it). Here the
/// vectors never contain `b`; [`ValueSet::bound`] lists the actions whose Q-value contains
/// it, so that `getActions` keeps them greedy exactly as in the paper.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ValueSet<const N: usize> {
    /// The vectors, all weakly dominated by `b` once produced by a backup. Pruned: no
    /// vector is weakly dominated by another or by a convex combination of others (up to
    /// the margin).
    pub(crate) points: Vec<Point<N>>,
    /// The actions whose Q-value contains `b`; [`NO_ACTION`] for a dead end. `b` belongs to
    /// the set exactly when this is not empty. Sorted, without duplicates.
    pub(crate) bound: Vec<u32>,
}

impl<const N: usize> ValueSet<N> {
    /// The value of a goal: the zero vector.
    pub(crate) fn goal() -> Self {
        Self {
            points: alloc::vec![Point {
                vector: [0.0; N],
                action: NO_ACTION,
            }],
            bound: Vec::new(),
        }
    }

    /// The value of a dead end: only `b`.
    pub(crate) fn dead_end() -> Self {
        Self {
            points: Vec::new(),
            bound: alloc::vec![NO_ACTION],
        }
    }

    /// Returns `true` if `b` belongs to the set.
    pub(crate) fn has_bound(&self) -> bool {
        !self.bound.is_empty()
    }

    /// The vectors of the set, without `b`.
    pub(crate) fn vectors(&self) -> impl Iterator<Item = &[f64; N]> {
        self.points.iter().map(|point| &point.vector)
    }
}

/// Returns `true` if `u` weakly dominates `v`: `u_i <= v_i` for every objective.
pub(crate) fn dominates<const N: usize>(u: &[f64; N], v: &[f64; N]) -> bool {
    u.iter().zip(v).all(|(a, b)| a <= b)
}

/// Reusable buffers of [`prune`].
#[derive(Debug, Default)]
pub(crate) struct Scratch<const N: usize> {
    /// The vectors a candidate is compared against.
    others: Vec<[f64; N]>,
    /// The simplex tableau.
    tableau: Vec<f64>,
}

/// Prunes `points` to a convex coverage set: removes every vector weakly dominated by
/// another (the first of equal vectors stays), then every vector that no weight makes
/// better than all remaining others by more than `margin`.
///
/// One objective keeps the first smallest value. Two objectives use a monotone chain over
/// the sorted front ([`planar::prune`], `O(m log m)`); more use one linear program per
/// vector ([`prune_by_programs`]).
pub(crate) fn prune<const N: usize>(
    points: &mut Vec<Point<N>>,
    margin: f64,
    scratch: &mut Scratch<N>,
) {
    match N {
        1 => keep_minimum(points),
        2 => planar::prune(points, margin),
        _ => prune_by_programs(points, margin, scratch),
    }
}

/// Keeps only the first vector with the smallest first component.
fn keep_minimum<const N: usize>(points: &mut Vec<Point<N>>) {
    let best = points
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| a.vector[0].total_cmp(&b.vector[0]).then(i.cmp(j)))
        .map(|(index, _)| index);
    if let Some(best) = best {
        points.swap(0, best);
        points.truncate(1);
    }
}

/// The general pruning: the dominance filter, then the covering program of
/// [`is_covered`] per vector.
///
/// The convex test removes candidates one at a time, worst component sum first, and tests
/// each against the vectors still kept, so of two nearly equal vectors one survives.
fn prune_by_programs<const N: usize>(
    points: &mut Vec<Point<N>>,
    margin: f64,
    scratch: &mut Scratch<N>,
) {
    remove_dominated(points);
    if points.len() <= 1 {
        return;
    }

    points.sort_by(|a, b| sum(&b.vector).total_cmp(&sum(&a.vector)));
    let mut index = 0;
    while index < points.len() {
        scratch.others.clear();
        scratch.others.extend(
            points
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != index)
                .map(|(_, point)| point.vector),
        );
        if is_covered(
            &points[index].vector,
            &scratch.others,
            margin,
            &mut scratch.tableau,
        ) {
            points.remove(index);
        } else {
            index += 1;
        }
    }
}

/// Removes every vector weakly dominated by another; of equal vectors the first stays.
fn remove_dominated<const N: usize>(points: &mut Vec<Point<N>>) {
    let mut index = 0;
    while index < points.len() {
        let v = points[index].vector;
        let beaten = points.iter().enumerate().any(|(other, point)| {
            other != index
                && dominates(&point.vector, &v)
                && (!same(&point.vector, &v) || other < index)
        });
        if beaten {
            points.remove(index);
        } else {
            index += 1;
        }
    }
}

/// Returns `true` if two vectors are equal in every component (bitwise equality of the
/// values, as dominance in both directions).
fn same<const N: usize>(u: &[f64; N], v: &[f64; N]) -> bool {
    dominates(u, v) && dominates(v, u)
}

/// The sum of a vector's components.
pub(crate) fn sum<const N: usize>(v: &[f64; N]) -> f64 {
    v.iter().sum()
}

/// The bounded set sum `acc (+)_B p * set`: every sum of a vector of `acc` and `p` times a
/// vector of `set` that `b` weakly dominates. Returns `true` if `b` belongs to the result:
/// some sum exceeds `b`, or `set` contains `b` (here `p * b` counts as `b`, see
/// `docs/design.md`). Tags are taken from `acc`.
pub(crate) fn add_scaled<const N: usize>(
    acc: &[Point<N>],
    set: &ValueSet<N>,
    probability: f64,
    bound: &[f64; N],
    out: &mut Vec<Point<N>>,
) -> bool {
    out.clear();
    let mut exceeded = set.has_bound();
    for left in acc {
        for right in set.vectors() {
            let mut vector = left.vector;
            for (component, add) in vector.iter_mut().zip(right) {
                *component += probability * add;
            }

            if dominates(&vector, bound) {
                out.push(Point {
                    vector,
                    action: left.action,
                });
            } else {
                exceeded = true;
            }
        }
    }

    exceeded
}

/// The Hausdorff distance between two values under the max-norm, with `b` counted as a
/// vector of a set that contains it. Zero for equal sets; infinite if exactly one is empty.
pub(crate) fn hausdorff<const N: usize>(a: &ValueSet<N>, b: &ValueSet<N>, bound: &[f64; N]) -> f64 {
    directed(a, b, bound).max(directed(b, a, bound))
}

/// `max_{u in a} min_{v in b} d(u, v)`.
fn directed<const N: usize>(a: &ValueSet<N>, b: &ValueSet<N>, bound: &[f64; N]) -> f64 {
    members(a, bound)
        .map(|u| {
            members(b, bound)
                .map(|v| distance(u, v))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// The vectors of `set`, with `b` if the set contains it.
fn members<'a, const N: usize>(
    set: &'a ValueSet<N>,
    bound: &'a [f64; N],
) -> impl Iterator<Item = &'a [f64; N]> {
    set.vectors().chain(set.has_bound().then_some(bound))
}

/// The max-norm distance of two vectors.
fn distance<const N: usize>(u: &[f64; N], v: &[f64; N]) -> f64 {
    u.iter()
        .zip(v)
        .map(|(a, b)| libm::fabs(a - b))
        .fold(0.0, f64::max)
}

/// The smallest scalarised value `w . v` over the vectors of `set`, and the vector's index.
pub(crate) fn best_under<const N: usize>(set: &ValueSet<N>, weight: &[f64; N]) -> Option<usize> {
    set.points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| dot(weight, &a.vector).total_cmp(&dot(weight, &b.vector)))
        .map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// Wraps vectors as untagged points.
    fn points<const N: usize>(vectors: &[[f64; N]]) -> Vec<Point<N>> {
        vectors
            .iter()
            .map(|&vector| Point {
                vector,
                action: NO_ACTION,
            })
            .collect()
    }

    /// The vertices of the lower-left convex hull of 2D points, by a monotone chain over the
    /// Pareto front. An oracle independent of the simplex.
    fn lower_left_hull(input: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut front: Vec<[f64; 2]> = input
            .iter()
            .copied()
            .filter(|v| !input.iter().any(|u| dominates(u, v) && !same(u, v)))
            .collect();
        front.sort_by(|a, b| a[0].total_cmp(&b[0]));
        front.dedup();

        let mut hull: Vec<[f64; 2]> = Vec::new();
        for point in front {
            while hull.len() >= 2 {
                let (a, b) = (hull[hull.len() - 2], hull[hull.len() - 1]);
                let cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
                if cross <= 1e-12 {
                    hull.pop();
                } else {
                    break;
                }
            }

            hull.push(point);
        }

        hull
    }

    /// Pruning random 2D sets leaves exactly the hull vertices.
    #[test]
    fn prune_matches_the_monotone_chain_hull() {
        let mut rng = SplitMix64::new(7);
        let mut scratch = Scratch::default();
        for _ in 0..300 {
            let count = 1 + (rng.next_u64() % 12) as usize;
            let vectors: Vec<[f64; 2]> = (0..count)
                .map(|_| {
                    [
                        f64::from(u8::try_from(rng.next_u64() % 20).unwrap()) / 2.0,
                        f64::from(u8::try_from(rng.next_u64() % 20).unwrap()) / 2.0,
                    ]
                })
                .collect();
            let mut planar = points(&vectors);
            prune(&mut planar, 1e-9, &mut scratch);
            let mut general = points(&vectors);
            prune_by_programs(&mut general, 1e-9, &mut scratch);

            for set in [planar, general] {
                let mut kept: Vec<[f64; 2]> = set.iter().map(|point| point.vector).collect();
                kept.sort_by(|a, b| a[0].total_cmp(&b[0]));
                assert_eq!(kept, lower_left_hull(&vectors), "input {vectors:?}");
            }
        }
    }

    /// Edge points and near twins are pruned; one twin survives.
    #[test]
    fn prune_drops_edge_points_and_one_twin() {
        let mut scratch = Scratch::default();
        let mut set = points(&[[2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [0.0, 2.0 + 1e-12]]);
        prune(&mut set, 1e-9, &mut scratch);
        assert_eq!(set.len(), 2);

        let mut twins = points(&[[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]]);
        prune(&mut twins, 1e-9, &mut scratch);
        assert_eq!(twins.len(), 1);

        let mut set = points(&[[2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [0.0, 2.0 + 1e-12]]);
        prune_by_programs(&mut set, 1e-9, &mut scratch);
        assert_eq!(set.len(), 2);

        let mut twins = points(&[[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]]);
        prune_by_programs(&mut twins, 1e-9, &mut scratch);
        assert_eq!(twins.len(), 1);
    }

    /// One objective keeps the minimum.
    #[test]
    fn prune_single_objective_keeps_the_minimum() {
        let mut scratch = Scratch::default();
        let mut set = points(&[[3.0], [1.0], [1.0 + 1e-12], [2.0]]);
        prune(&mut set, 1e-9, &mut scratch);
        assert_eq!(set.len(), 1);
        assert_eq!(set[0].vector, [1.0]);
    }

    /// The bounded sum drops vectors beyond `b` and reports it.
    #[test]
    fn bounded_sum() {
        let acc = points(&[[1.0, 0.0]]);
        let set = ValueSet {
            points: points(&[[2.0, 2.0], [20.0, 0.0]]),
            bound: Vec::new(),
        };
        let mut out = Vec::new();
        let exceeded = add_scaled(&acc, &set, 0.5, &[5.0, 5.0], &mut out);
        assert!(exceeded);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vector, [2.0, 1.0]);
    }

    /// The Hausdorff distance counts `b` as a member.
    #[test]
    fn hausdorff_with_bound() {
        let bound = [10.0, 10.0];
        let a = ValueSet {
            points: points(&[[1.0, 1.0]]),
            bound: Vec::new(),
        };
        let mut b = a.clone();
        assert_eq!(hausdorff(&a, &b, &bound), 0.0);

        b.bound.push(0);
        assert_eq!(hausdorff(&a, &b, &bound), 9.0);
        assert!(hausdorff(&a, &ValueSet::default(), &bound).is_infinite());
    }
}
