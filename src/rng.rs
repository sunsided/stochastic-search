//! The seeded pseudo-random generator behind LRTDP's successor sampling and the POMDP
//! planners' simulations.
//!
//! An own generator rather than a dependency, so that a seed reproduces a run across releases
//! of this crate regardless of other crates' versions (see `docs/design.md`). It implements
//! `rand_core`'s generator trait, which only fixes the interface, not the sequence.

use core::convert::Infallible;

use rand_core::Rng;

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
        unit_interval(self)
    }
}

impl rand_core::TryRng for SplitMix64 {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.next_u64() >> 32) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.next_u64())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
    }
}

/// Returns a uniform sample from `[0, 1)` with 53 random bits drawn from `rng`.
///
/// `rand_core` has no floating-point helpers, and a generative
/// [`Pomdp`](crate::Pomdp) usually needs one; this is the conversion the crate uses itself.
///
/// # Example
///
/// ```
/// use stochastic_search::unit_interval;
///
/// # struct Counter(u64);
/// # impl rand_core::TryRng for Counter {
/// #     type Error = core::convert::Infallible;
/// #     fn try_next_u32(&mut self) -> Result<u32, Self::Error> { Ok(0) }
/// #     fn try_next_u64(&mut self) -> Result<u64, Self::Error> { self.0 += 1 << 62; Ok(self.0) }
/// #     fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Self::Error> { Ok(()) }
/// # }
/// let mut rng = Counter(0);
/// assert_eq!(unit_interval(&mut rng), 0.25);
/// ```
pub fn unit_interval<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    /// `2^-53`, the spacing of the 53-bit grid on `[0, 1)`.
    const SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

    #[expect(
        clippy::cast_precision_loss,
        reason = "the value has at most 53 bits, which f64 represents exactly"
    )]
    let bits = (rng.next_u64() >> 11) as f64;

    bits * SCALE
}

/// Derives the seed of an independent stream from `seed` and two stream coordinates, by
/// running each coordinate through `SplitMix64`'s output mix in turn.
pub(crate) fn derive_seed(seed: u64, a: u64, b: u64) -> u64 {
    let first = SplitMix64::new(seed ^ a.wrapping_mul(0xD1B5_4A32_D192_ED03)).next_u64();

    SplitMix64::new(first ^ b.wrapping_mul(0xA24B_AED4_963E_E407)).next_u64()
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

    /// Derived streams differ per coordinate and repeat for equal ones.
    #[test]
    fn derived_seeds_are_distinct_and_stable() {
        use super::derive_seed;

        let seeds = [
            derive_seed(1, 0, 0),
            derive_seed(1, 1, 0),
            derive_seed(1, 0, 1),
            derive_seed(2, 0, 0),
        ];
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert_ne!(a, b);
            }
        }

        assert_eq!(derive_seed(1, 1, 0), seeds[1]);
    }
}
