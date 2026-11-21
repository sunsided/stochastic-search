//! Convex coverage set pruning for two objectives: a monotone chain over the sorted front,
//! then the margin test against each vertex's two hull neighbours.
//!
//! With two objectives, the weights are `(t, 1 - t)` for `t` in `[0, 1]`, and a hull vertex
//! competes only with its neighbours on the hull: any other vector is beaten by a neighbour
//! under every weight where it could beat the vertex. So the margin of a vertex is
//! `max_t min(f_left(t), f_right(t))` with `f_u(t) = (t, 1 - t) . (u - v)`, which is the
//! linear program of [`best_weight`](crate::mo::simplex::best_weight) solved in closed form.

use alloc::vec::Vec;

use crate::mo::coverage::Point;

/// Prunes `points` (two objectives; components past the second are ignored) to the lower
/// left convex hull, then removes the vertex of smallest margin while it is at most
/// `margin`. Keeps the first of equal vectors. The result is sorted by the first objective.
pub(super) fn prune<const N: usize>(points: &mut Vec<Point<N>>, margin: f64) {
    pareto_front(points);
    lower_hull(points);
    while points.len() > 1 {
        let (index, gap) = (0..points.len())
            .map(|index| (index, vertex_margin(points, index)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least two vertices");
        if gap > margin {
            break;
        }

        points.remove(index);
    }
}

/// Sorts by the first objective (then the second, then input order) and keeps the vectors
/// whose second objective is strictly below that of every vector before them: the strict
/// Pareto front, first of equals kept.
fn pareto_front<const N: usize>(points: &mut Vec<Point<N>>) {
    // A stable sort keeps the input order among equal vectors.
    points.sort_by(|a, b| {
        a.vector[0]
            .total_cmp(&b.vector[0])
            .then(a.vector[1].total_cmp(&b.vector[1]))
    });
    let mut best = f64::INFINITY;
    points.retain(|point| {
        let keep = point.vector[1] < best;
        if keep {
            best = point.vector[1];
        }

        keep
    });
}

/// Keeps the vertices of the lower convex hull of a front sorted by the first objective
/// (and strictly decreasing in the second); points on an edge are dropped.
fn lower_hull<const N: usize>(points: &mut Vec<Point<N>>) {
    let mut kept = 0;
    for index in 0..points.len() {
        while kept >= 2 && !turns_left(&points[kept - 2], &points[kept - 1], &points[index]) {
            kept -= 1;
        }

        points.swap(kept, index);
        kept += 1;
    }

    points.truncate(kept);
}

/// Returns `true` if `a`, `b`, `c` make a strict left turn (`b` lies below the segment
/// from `a` to `c`).
fn turns_left<const N: usize>(a: &Point<N>, b: &Point<N>, c: &Point<N>) -> bool {
    let (a, b, c) = (&a.vector, &b.vector, &c.vector);

    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0
}

/// The margin of the vertex at `index` over its hull neighbours: the largest, over the
/// weights `(t, 1 - t)`, of the smallest weighted advantage over a neighbour.
fn vertex_margin<const N: usize>(points: &[Point<N>], index: usize) -> f64 {
    let v = &points[index].vector;
    let neighbours = [
        index.checked_sub(1).map(|left| points[left].vector),
        points.get(index + 1).map(|right| right.vector),
    ];
    // The advantage over `u` under `(t, 1 - t)`, as `a * t + b`.
    let lines: Vec<(f64, f64)> = neighbours
        .iter()
        .flatten()
        .map(|u| ((u[0] - v[0]) - (u[1] - v[1]), u[1] - v[1]))
        .collect();
    let worst = |t: f64| {
        lines
            .iter()
            .map(|&(a, b)| a * t + b)
            .fold(f64::INFINITY, f64::min)
    };

    let mut candidates = [0.0, 1.0, 0.0];
    if let [(a1, b1), (a2, b2)] = lines[..] {
        let slope = a1 - a2;
        if slope != 0.0 {
            candidates[2] = ((b2 - b1) / slope).clamp(0.0, 1.0);
        }
    }

    candidates
        .into_iter()
        .map(worst)
        .fold(f64::NEG_INFINITY, f64::max)
}
