//! The linear program behind convex coverage set pruning: the weight under which a vector
//! beats a set of others by the widest margin.
//!
//! For a vector `v` and others `u_1..u_m`, it solves
//!
//! ```text
//! maximize x  subject to  w . (u_j - v) >= x  for all j,  sum(w) = 1,  w >= 0
//! ```
//!
//! (Roijers and Whiteson 2017, Section 4.1.3). `v` belongs to the convex coverage set of the
//! others exactly when the optimum `x*` is positive. The program is brought into the
//! standard form `max c.z, A z <= b, z >= 0, b >= 0` by eliminating `w_N = 1 - sum(w_i)`
//! and shifting `x = y - M` with `M` larger than every `|u_j - v|`, so the origin is a basic
//! feasible solution and no phase one is needed. It is solved with a dense tableau and
//! Bland's rule, which cannot cycle. The programs are tiny (`N` variables, one row per other
//! vector), so a dense tableau is the fastest choice.

use alloc::vec::Vec;

/// Pivot elements and reduced costs smaller than this are treated as zero.
const PIVOT_TOLERANCE: f64 = 1e-12;

/// The artificial variable of [`is_covered`] counts as zero below this.
const COVER_TOLERANCE: f64 = 1e-10;

/// The weight under which `v` beats every vector in `others` by the widest margin, and that
/// margin `x* = min_j w . (u_j - v)`.
///
/// `tableau` is a reusable buffer. With no others, every weight works: returns the uniform
/// weight and an infinite margin.
pub(crate) fn best_weight<const N: usize>(
    v: &[f64; N],
    others: &[[f64; N]],
    tableau: &mut Vec<f64>,
) -> ([f64; N], f64) {
    if others.is_empty() {
        return (uniform(), f64::INFINITY);
    }

    let lp = Tableau::build(v, others, tableau);
    let lp = lp.solve();
    let weight = lp.weight::<N>();
    let margin = margin(&weight, v, others);

    (weight, margin)
}

/// Returns `true` if `v` beats no vector of `others` by more than `margin` under any
/// weight: if some convex combination `c` of `others` satisfies `c <= v + margin` in every
/// objective.
///
/// By the minimax theorem this is `max_w min_j w . (u_j - v) <= margin`, the test of
/// [`best_weight`] decided through its dual: `N + 1` rows instead of one per other vector,
/// so a pivot costs `O(N m)` instead of `O(m^2)`. Phase one of the simplex method on
/// `sum_j lambda_j (u_j - v) + s = margin`, `sum_j lambda_j + a = 1`: covered exactly when
/// the artificial `a` can be driven to zero.
pub(crate) fn is_covered<const N: usize>(
    v: &[f64; N],
    others: &[[f64; N]],
    margin: f64,
    tableau: &mut Vec<f64>,
) -> bool {
    if others.is_empty() {
        return false;
    }

    let lp = Tableau::build_covering(v, others, margin, tableau).solve();

    lp.objective() >= -COVER_TOLERANCE
}

/// `min_j w . (u_j - v)`: how much `v` beats the closest of `others` under `weight`.
pub(crate) fn margin<const N: usize>(weight: &[f64; N], v: &[f64; N], others: &[[f64; N]]) -> f64 {
    others
        .iter()
        .map(|u| dot(weight, u) - dot(weight, v))
        .fold(f64::INFINITY, f64::min)
}

/// The weight `1 / N` on every objective.
pub(crate) fn uniform<const N: usize>() -> [f64; N] {
    #[expect(
        clippy::cast_precision_loss,
        reason = "N is a handful of objectives, exact in f64"
    )]
    let share = 1.0 / N as f64;

    [share; N]
}

/// The scalar product of two vectors.
pub(crate) fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A dense simplex tableau in standard form, with its basis.
struct Tableau<'a> {
    /// Row-major coefficients: `rows` constraint rows, then the objective row; each row
    /// has `columns` entries, the right-hand side last.
    cells: &'a mut Vec<f64>,
    /// Constraint rows.
    rows: usize,
    /// Entries per row: structural variables, slacks, right-hand side.
    columns: usize,
    /// Structural variables: `w_1..w_{N-1}`, then `y`.
    structural: usize,
    /// The basic variable of each constraint row.
    basis: Vec<usize>,
}

impl<'a> Tableau<'a> {
    /// Builds the standard-form program for `v` against `others` (see the module docs).
    fn build<const N: usize>(v: &[f64; N], others: &[[f64; N]], cells: &'a mut Vec<f64>) -> Self {
        let free = N - 1;
        let structural = N;
        let rows = others.len() + usize::from(free > 0);
        let columns = structural + rows + 1;
        let shift = 1.0
            + others
                .iter()
                .flat_map(|u| u.iter().zip(v).map(|(a, b)| libm::fabs(a - b)))
                .fold(0.0, f64::max);

        cells.clear();
        cells.resize((rows + 1) * columns, 0.0);
        for (row, u) in others.iter().enumerate() {
            let last = u[free] - v[free];
            let line = &mut cells[row * columns..(row + 1) * columns];
            for i in 0..free {
                line[i] = -((u[i] - v[i]) - last);
            }

            line[free] = 1.0;
            line[structural + row] = 1.0;
            line[columns - 1] = last + shift;
        }

        if free > 0 {
            let row = others.len();
            let line = &mut cells[row * columns..(row + 1) * columns];
            line[..free].fill(1.0);
            line[structural + row] = 1.0;
            line[columns - 1] = 1.0;
        }

        cells[rows * columns + free] = -1.0;
        let basis = (structural..structural + rows).collect();

        Self {
            cells,
            rows,
            columns,
            structural,
            basis,
        }
    }

    /// Builds phase one of the covering program of [`is_covered`]: columns `lambda_j`, one
    /// slack per objective, the artificial `a`; the objective maximises `-a`.
    fn build_covering<const N: usize>(
        v: &[f64; N],
        others: &[[f64; N]],
        margin: f64,
        cells: &'a mut Vec<f64>,
    ) -> Self {
        let m = others.len();
        let rows = N + 1;
        let columns = m + N + 2;
        cells.clear();
        cells.resize((rows + 1) * columns, 0.0);
        for objective in 0..N {
            let line = &mut cells[objective * columns..(objective + 1) * columns];
            for (j, u) in others.iter().enumerate() {
                line[j] = u[objective] - v[objective];
            }

            line[m + objective] = 1.0;
            line[columns - 1] = margin;
        }

        let line = &mut cells[N * columns..(N + 1) * columns];
        line[..m].fill(1.0);
        line[m + N] = 1.0;
        line[columns - 1] = 1.0;

        // The objective row with the basic artificial eliminated: -a = sum(lambda) - 1.
        let objective = &mut cells[rows * columns..];
        objective[..m].fill(-1.0);
        objective[columns - 1] = -1.0;
        let basis = (m..=m + N).collect();

        Self {
            cells,
            rows,
            columns,
            structural: m,
            basis,
        }
    }

    /// The objective value of the current basis.
    fn objective(&self) -> f64 {
        self.cells[(self.rows + 1) * self.columns - 1]
    }

    /// Pivots until no reduced cost is negative, entering and leaving by Bland's rule.
    fn solve(mut self) -> Self {
        while let Some(entering) = self.entering() {
            let Some(leaving) = self.leaving(entering) else {
                // Unbounded cannot happen (y is bounded by every row); stop defensively.
                break;
            };

            self.pivot(leaving, entering);
        }

        self
    }

    /// The lowest-index column with a negative reduced cost, if any.
    fn entering(&self) -> Option<usize> {
        let objective = &self.cells[self.rows * self.columns..];

        (0..self.columns - 1).find(|&column| objective[column] < -PIVOT_TOLERANCE)
    }

    /// The row with the smallest ratio for `column`; ties go to the lowest basic variable.
    fn leaving(&self, column: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for row in 0..self.rows {
            let a = self.cells[row * self.columns + column];
            if a <= PIVOT_TOLERANCE {
                continue;
            }

            let ratio = self.cells[row * self.columns + self.columns - 1] / a;
            let better = best.is_none_or(|(current, best_ratio)| {
                ratio < best_ratio - PIVOT_TOLERANCE
                    || (ratio <= best_ratio + PIVOT_TOLERANCE
                        && self.basis[row] < self.basis[current])
            });
            if better {
                best = Some((row, ratio));
            }
        }

        best.map(|(row, _)| row)
    }

    /// Makes `column` basic in `row`.
    fn pivot(&mut self, row: usize, column: usize) {
        let columns = self.columns;
        let scale = 1.0 / self.cells[row * columns + column];
        for cell in &mut self.cells[row * columns..(row + 1) * columns] {
            *cell *= scale;
        }

        for other in 0..=self.rows {
            if other == row {
                continue;
            }

            let factor = self.cells[other * columns + column];
            if factor == 0.0 {
                continue;
            }

            for index in 0..columns {
                let delta = factor * self.cells[row * columns + index];
                self.cells[other * columns + index] -= delta;
            }
        }

        self.basis[row] = column;
    }

    /// Reads the weight off the solved tableau, restoring `w_N = 1 - sum(w_i)`.
    fn weight<const N: usize>(&self) -> [f64; N] {
        let mut weight = [0.0; N];
        for (row, &variable) in self.basis.iter().enumerate() {
            if variable < self.structural - 1 {
                weight[variable] = self.cells[row * self.columns + self.columns - 1].max(0.0);
            }
        }

        let rest: f64 = weight[..N - 1].iter().sum();
        weight[N - 1] = (1.0 - rest).max(0.0);

        weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector in the middle of a hull edge has margin zero; a vertex has a positive one.
    #[test]
    fn margin_of_vertices_and_edge_points() {
        let mut buffer = Vec::new();
        let others = [[2.0, 0.0], [0.0, 2.0]];
        let (_, edge) = best_weight(&[1.0, 1.0], &others, &mut buffer);
        assert!(edge.abs() < 1e-12, "edge point margin {edge}");

        let (weight, vertex) = best_weight(&[0.5, 0.5], &others, &mut buffer);
        assert!((vertex - 0.5).abs() < 1e-12, "vertex margin {vertex}");
        assert!((weight[0] - 0.5).abs() < 1e-12);

        let (_, dominated) = best_weight(&[3.0, 3.0], &others, &mut buffer);
        assert!(dominated < 0.0);
    }

    /// The covering test agrees with the margin of the primal program.
    #[test]
    fn covering_agrees_with_the_margin() {
        let mut buffer = Vec::new();
        let mut state = 11_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            f64::from(u8::try_from(state % 16).unwrap()) / 4.0
        };
        for _ in 0..500 {
            let v = [next(), next(), next()];
            let count = if next() < 2.0 { 1 } else { 3 };
            let others: Vec<[f64; 3]> = (0..count).map(|_| [next(), next(), next()]).collect();
            let (_, gap) = best_weight(&v, &others, &mut buffer);
            for margin in [0.0, 0.1, 0.5] {
                if (gap - margin).abs() > 1e-9 {
                    assert_eq!(
                        is_covered(&v, &others, margin, &mut buffer),
                        gap <= margin,
                        "{v:?} {others:?} gap {gap}"
                    );
                }
            }
        }
    }

    /// One objective: the margin is the gap to the smallest other value.
    #[test]
    fn single_objective_margin_is_the_gap() {
        let mut buffer = Vec::new();
        let (weight, gap) = best_weight(&[1.0], &[[4.0], [2.5]], &mut buffer);
        assert_eq!(weight, [1.0]);
        assert!((gap - 1.5).abs() < 1e-12);
    }

    /// Three objectives: the corner vector wins under the weight on its own objective.
    #[test]
    fn three_objectives() {
        let mut buffer = Vec::new();
        let others = [[0.0, 5.0, 5.0], [5.0, 5.0, 0.0]];
        let (weight, gap) = best_weight(&[5.0, 0.0, 5.0], &others, &mut buffer);
        assert!(gap > 1.0, "margin {gap}");
        assert!((margin(&weight, &[5.0, 0.0, 5.0], &others) - gap).abs() < 1e-12);
    }
}
