//! The seeded pseudo-random generator behind LRTDP's successor sampling.
//!
//! An own generator rather than a dependency, so that a seed reproduces a run across releases
//! of this crate regardless of other crates' versions (see `docs/design.md`).

/// The `SplitMix64` generator (Steele, Lea and Flood 2014).
///
/// Fast, with a full `2^64` period and good equidistribution for sampling. Not
/// cryptographically secure, which sampling does not need.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    /// The generator state, advanced by a fixed odd increment per draw.
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from a seed. Every seed, including `0`, is valid.
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// Returns a uniform sample from `[0, 1)` with 53 random bits.
    pub(crate) fn next_unit(&mut self) -> f64 {
        /// `2^-53`, the spacing of the 53-bit grid on `[0, 1)`.
        const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

        #[expect(
            clippy::cast_precision_loss,
            reason = "the value has at most 53 bits, which f64 represents exactly"
        )]
        let bits = (self.next_u64() >> 11) as f64;

        bits * SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    /// The first outputs for seed 0 match the reference implementation.
    #[test]
    fn matches_reference_sequence() {
        let mut rng = SplitMix64::new(0);

        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    /// Unit samples stay in `[0, 1)`.
    #[test]
    fn unit_samples_are_in_range() {
        let mut rng = SplitMix64::new(42);

        for _ in 0..1000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
