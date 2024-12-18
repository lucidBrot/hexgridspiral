/*!
Hexagonal Cube-Coordinate System with Spiralling Integer Tile Identifiers

Read the [README](https://github.com/lucidBrot/hexgridspiral) for
* an Abstract
* Construction Examples
* A list of Features

*/
// We have code that uses the nightly toolchain. It is gated behind the "nightly" crate feature flag.
// This means the crate can still be used with stable rust - it will just not have all functions.
#![cfg_attr(feature = "nightly", feature(step_trait))]
#![cfg_attr(feature = "nightly", feature(coroutines))]
#![cfg_attr(feature = "nightly", feature(iter_from_coroutine))]
#![cfg_attr(feature = "nightly", feature(impl_trait_in_assoc_type))]
#![cfg_attr(feature = "nightly", feature(doc_cfg))]

// The objects Tile, Ring, TileIndex, RingIndex are not supposed to be mutated.
// Instead, (they) make new objects.
use derive_more::{Add, Display, From, Into, Mul, Neg, Sub};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::ops;

#[derive(
    Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Add, Sub, Mul, Display, From, Into,
)]
pub struct TileIndex(pub u64);

cfg_if::cfg_if! {if #[cfg(feature = "nightly")] {
#[cfg(any(feature = "nightly", doc))]
#[doc(cfg(feature = "nightly"))]
impl std::iter::Step for TileIndex {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        return if end < start {
            // End < Start is illegal : https://doc.rust-lang.org/nightly/std/iter/trait.Step.html#tymethod.steps_between
            (0, None)
        } else {
            match usize::try_from((*end - *start).value() as usize) {
                // The happy path:
                Result::Ok(value) => (value, Some(value)),
                // Returns (usize::MAX, None) if the number of steps would overflow usize, or is infinite.
                Result::Err(_) => (usize::MAX, None),
            }
        };

        // This is currently a (lower-bound, upper-bound) tuple in nightly, "like size_hint".
        // It is exactly defined when the `n` must be `0` or `n` or `usize::max`. The rest seems not too relevant.
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Some(
            u64::try_from((start + u64::try_from(count).ok()?).value())
                .ok()?
                .into(),
        )
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        (start - u64::try_from(count).ok()?).try_into().ok()
    }
}
}}

/// Which Ring around the origin we're at, counting from 1.
/// Implementation Detail: The RingIndex wraps an integer that equals the number of tiles in one edge of the Ring (including both corners).
#[derive(Debug, Copy, Clone, PartialEq, Add, Sub, Mul, Display, From)]
pub struct RingIndex(u64);

impl TileIndex {
    pub fn value(&self) -> u64 {
        self.0
    }

    pub fn is_origin(&self) -> bool {
        self.0 == 0
    }
}

impl RingIndex {
    pub const ORIGIN_RING: RingIndex = RingIndex(1);

    pub fn value(&self) -> u64 {
        let val = self.0;
        assert!(val > 0, "{val} is not a valid RingIndex value.");
        val
    }
}

// Define Addition for an u64 (offset) to a Tile/Ring-Index.
// Tbh this adds too much magic and the boilerplate is also needed for subtraction, multiplication, etc. and for all integer types...
impl ops::Add<u64> for TileIndex {
    type Output = TileIndex;

    fn add(self, rhs: u64) -> TileIndex {
        TileIndex(self.value() + rhs)
    }
}

impl ops::Sub<u64> for TileIndex {
    type Output = TileIndex;

    fn sub(self, rhs: u64) -> TileIndex {
        assert!(rhs <= self.value());
        TileIndex(self.value() - rhs)
    }
}

impl ops::Rem<u64> for TileIndex {
    type Output = TileIndex;

    fn rem(self, rhs: u64) -> TileIndex {
        TileIndex(self.value() % rhs)
    }
}

impl ops::Add<TileIndex> for u64 {
    type Output = TileIndex;

    fn add(self, rhs: TileIndex) -> TileIndex {
        TileIndex(self + rhs.value())
    }
}

impl ops::Add<u64> for RingIndex {
    type Output = RingIndex;

    fn add(self, rhs: u64) -> RingIndex {
        RingIndex(self.value() + rhs)
    }
}

impl ops::Add<RingIndex> for u64 {
    type Output = RingIndex;

    fn add(self, rhs: RingIndex) -> RingIndex {
        RingIndex(self + rhs.value())
    }
}

impl ops::Sub<u64> for RingIndex {
    type Output = RingIndex;

    fn sub(self, rhs: u64) -> RingIndex {
        assert!(rhs <= self.value());
        RingIndex(self.value() - rhs)
    }
}

/// A Ring is hexagonal and consists of tiles.
/// All Ring-Corner Tiles have the same number of steps to the origin.
/// The other tiles in the Ring are on straight edges between corners.
/// The [RingIndex] counts from 1 and thus is always equal to the [Ring::edge_size].
#[derive(Debug, Copy, Clone, PartialEq, Add, Display, From, Into)]
pub struct Ring {
    /// ring-index
    n: RingIndex,
}

/// Hexgrid Spiral Tile
///
/// Identified by a single integer index ([TileIndex]) that spirals around the origin.
#[derive(Debug, Copy, Clone, PartialEq, Display, From, Into)]
#[display("HGSTile: {h}")]
pub struct HGSTile {
    // tile-index
    h: TileIndex,
    ring: Ring,
}

/// CubeCoordinates Tile
/// For reference: <https://www.redblobgames.com/grids/hexagons/>
///
/// `r` is constant 0 along the x-Axis (towards right) while `q` increments and `s` decrements.
///
/// Towards the bottom (negative y-Axis), `r` increments while the other decrement equally to ensure the invariant `q+r+s=0` always holds.
///
/// Towards the bottom-right, `q` stays constant and is negative on the bottom side.
/// Towards the right, `r` stays constant and is *positive* on the bottom side.
/// Towards the top-right, `s` stays constant and is negative on the bottom side.
// TODO: Division is not implemented for CCTile. If needed, it should yield a CCTileFloat type.
#[derive(Debug, Copy, Clone, PartialEq, Display, From, Into, Eq, Neg, Add, Mul, Sub)]
#[display("CCTile: ({q}, {r},{s})")]
pub struct CCTile {
    q: i64,
    r: i64,
    s: i64,
}

/// An identifier for each side of the ring's edge.
/// Numbered counterclockwise from the top-right edge.
#[derive(Debug, Copy, Clone, PartialEq, Add, Display, From, Into)]
pub struct RingEdge(u8);

/// An identifier for each corner of the ring's edge.
/// Numbered counterclockwise, from the right corner.
/// The n-th corner is the start of the n-th edge.
#[derive(Debug, Copy, Clone, PartialEq, Add, Display, TryFromPrimitive, IntoPrimitive, Eq)]
#[repr(u8)]
pub enum RingCornerIndex {
    RIGHT = 0,
    TOPRIGHT = 1,
    TOPLEFT = 2,
    LEFT = 3,
    BOTTOMLEFT = 4,
    BOTTOMRIGHT = 5,
}

impl RingEdge {
    pub fn start(&self) -> RingCornerIndex {
        TryInto::<RingCornerIndex>::try_into(self.0).unwrap()
    }

    pub fn end(&self) -> RingCornerIndex {
        TryInto::<RingCornerIndex>::try_into((self.0 + 1) % 6).unwrap()
    }

    /// Returns a RingCornerIndex that signifies the edge direction. E.g. the edge going towards the top-left would return the top-left corner.
    pub fn direction(&self) -> RingCornerIndex {
        self.end().next()
    }

    /// Transforms two distinct corners into an edge.
    /// Not defined for just one corner.
    pub fn from_corners<'a>(a: &'a RingCornerIndex, b: &'a RingCornerIndex) -> Self {
        assert_ne!(a, b);
        let a_val: u8 = (*a).into();
        let b_val: u8 = (*b).into();
        let edge_start: RingCornerIndex =
            RingCornerIndex::try_from_primitive(u8::min(a_val, b_val)).unwrap();
        // if there is a zero-crossing, need to special-case:
        if ((*b == RingCornerIndex::BOTTOMRIGHT) && (*a == RingCornerIndex::RIGHT))
            || ((*a == RingCornerIndex::BOTTOMRIGHT) && (*b == RingCornerIndex::RIGHT))
        {
            // bottom-right edge
            return Self(5);
        } else {
            let res: RingEdge = u8::from(edge_start).into();
            debug_assert_eq!(edge_start, res.start());
            res
        }
    }

    pub fn from_primitive(p: u8) -> Self {
        Self(p % 6)
    }
}

impl RingCornerIndex {
    pub fn next(&self) -> Self {
        let v = *self as u8;
        let v2 = match v {
            0..=4 => v + 1,
            5 => 0,
            // TODO: Replace all panics and unwraps with actual error handling where possible.
            6_u8..=u8::MAX => panic!("Invalid corner index used as start."),
        };
        TryInto::<RingCornerIndex>::try_into(v2).unwrap()
    }

    pub fn all() -> impl std::iter::Iterator<Item = RingCornerIndex> {
        Self::all_from(Self::RIGHT)
    }

    // See https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html
    // the "Precise captures use" section.
    pub fn all_from(start: RingCornerIndex) -> impl std::iter::Iterator<Item = RingCornerIndex> {
        let mut ctr = start;
        let mut done = false;
        std::iter::from_fn(move || {
            ctr = ctr.next();
            if done {
                return None;
            }
            if ctr == start {
                done = true;
            };
            return Some(ctr);
        })
    }
}

impl HGSTile {
    pub fn new(tile_index: TileIndex) -> Self {
        return Self {
            h: tile_index,
            ring: Ring::new(ring_index_for_tile_index(tile_index)),
        };
    }

    /// Shorthand for creating a tile in HexGridSpiral format with TileIndex(n).
    pub fn make(tile_index: u64) -> Self {
        HGSTile::new(TileIndex(tile_index))
    }

    pub fn is_origin_tile(&self) -> bool {
        self.h == TileIndex(0)
    }

    pub fn increment_spiral(&self) -> Self {
        Self::new(self.h + TileIndex(1))
    }

    /// `steps` steps along the spiral. May be negative, but may not lead to a negative tile-index.
    pub fn spiral_steps(&self, steps: i64) -> Self {
        assert!(steps <= self.h.value() as i64);
        Self::new(TileIndex((self.h.value() as i64 + steps) as u64))
    }

    pub fn decrement_spiral(&self) -> Self {
        assert!(
            self.h.value() > 0,
            "Can not decrement from the origin-tile."
        );
        Self::new(self.h - TileIndex(1))
    }

    // The maximum tile-index in the current ring.
    pub fn ring_max(&self) -> Self {
        HGSTile::new(self.ring.max())
    }

    pub fn ring_min(&self) -> Self {
        HGSTile::new(self.ring.min())
    }

    /// Edge-index on the ring, where the tile resides.
    /// Not well-defined for the origin-tile.
    pub fn ring_edge(&self) -> RingEdge {
        self.ring.edge(self.h)
    }

    pub fn cc(&self) -> CCTile {
        self.into()
    }
}

impl Sub<i32> for TileIndex {
    type Output = TileIndex;

    fn sub(self, rhs: i32) -> Self::Output {
        TileIndex((self.value() as i64 - rhs as i64) as u64)
    }
}

impl Ring {
    pub fn new(ring_index: RingIndex) -> Self {
        Ring { n: ring_index }
    }

    pub fn from_tile_index(tile_index: &TileIndex) -> Self {
        Ring::new(ring_index_for_tile_index(*tile_index))
    }

    pub fn next_ring(&self) -> Ring {
        Ring { n: self.n + 1 }
    }

    pub fn prev_ring(&self) -> Ring {
        assert!(self.n.value() > 0, "Cannot decrement the innermost circle.");
        Ring { n: self.n - 1 }
    }

    pub fn neighbors_in_ring(&self, tile: &HGSTile) -> Vec<HGSTile> {
        // in most cases, they are simply one up and down the tile index.
        // The exception is the case where it is the maximum or minimum entry.
        let base = ring_min(self.n);
        let h_offset = tile.h - base;
        let size = ring_size(self.n);
        let lesser_neighbor: HGSTile = HGSTile::new(base + ((h_offset - 1) % size));
        let greater_neighbor: HGSTile = HGSTile::new(base + ((h_offset + 1) % size));
        return vec![lesser_neighbor, greater_neighbor];
    }

    /// Edge size of the [Ring], counting both ends (Ring corners).
    pub fn full_edge_size(&self) -> u64 {
        return self.n.value();
    }

    pub fn size(&self) -> u64 {
        return ring_size(self.n);
    }

    pub fn min(&self) -> TileIndex {
        return ring_min(self.n);
    }

    pub fn max(&self) -> TileIndex {
        return ring_max(self.n);
    }

    /// Edge index on the ring where the Tile h resides in.
    pub fn edge(&self, h: TileIndex) -> RingEdge {
        assert!(!h.is_origin());
        // The ring-minimum has offset 1
        // The ring-maximum has offset 0.
        let b = self.min();
        let ring = Ring::from_tile_index(&h);
        let side_size = ring.full_edge_size() - 1;
        assert!(h >= b);
        let offset: u64 = (h - b).value();
        let ring_size = ring.size();
        assert!(
            offset < ring_size,
            "offset={offset:?}, ringsize={ring_size:?}, h={h:?}, ring={ring:?}"
        );
        return RingEdge::from_primitive((offset / side_size).try_into().unwrap());
    }

    /// Length of the edge of the ring, counting only one corner.
    pub fn edge_size(&self) -> u64 {
        self.full_edge_size() - 1
    }

    /// Get the c-th corner of the ring as tile index.
    pub fn corner(&self, c: RingCornerIndex) -> TileIndex {
        let val: u8 = c.into();
        // The start of the edge i is the corner at index i.
        if val == 0 {
            return self.max();
        }

        // need to make  edge_len * num_edges  steps from the starting corner.
        // The ring.min() is already at step 1.
        return self.min() + self.edge_size() * (val as u64) - 1;
    }

    pub fn random_tile_in_ring<RNG: rand::Rng>(&self, rng: &mut RNG) -> TileIndex {
        TileIndex(rng.gen_range(self.min().value()..=self.max().value()))
    }

    /// Like `ring.n` but counts from zero. I.e. the Origin-Tile is in the ring 1, which has radius 0.
    pub fn radius(&self) -> u64 {
        (self.n - RingIndex::ORIGIN_RING).value()
    }
}

fn ring_size(n: RingIndex) -> u64 {
    // n is the edge-length with both corners. A hexagon has six sides.
    // The origin ring is the exception
    if n == RingIndex::ORIGIN_RING {
        return 1;
    }
    return (n.value() - 1) * 6;
}

// The ring-max lies always on a ring-corner tile.
fn ring_max(n: RingIndex) -> TileIndex {
    /*
     max(n) := max(n-1) + size(n)
     That leads to a sum:
     max(n) := sum_{i=1}^{n}{size(i)}
             = sum_{i=1}^{n}{6*(i-1)}
             = 6 sum_{i=0}^{n-1}{i}
     Which is known as a Triangular Number, computable in closed-form.
     max(n) := (n-1)*n/2
    */
    if n.value() == 0 {
        return TileIndex(0);
    }
    return TileIndex(3 * (n.value() - 1) * n.value());
}

/// Compute the ring-index n for the ring with the maximum element h.
/// You can also pass any other tile-index.
/// This is the inversion of ring_max.
pub fn ring_index_for_tile_index(h: TileIndex) -> RingIndex {
    // Since the equation in ring_max is quadratic, we get two potential solutions for n.
    // But one of them is, for positive h, always negative and thus invalid.
    // For a maximum it's an integer result.
    // For a nonmaximum value in the ring, it'll be lower than that integer but higher than the next-lower integer result.
    // And since the results (n) can be any integer value, it is thus always close enough to be roundable.
    return RingIndex((1. / 6. * (3. + f64::sqrt((12 * h.value() + 9) as f64))).ceil() as u64);
}

fn ring_min(n: RingIndex) -> TileIndex {
    if n == RingIndex::ORIGIN_RING {
        return TileIndex(0);
    }
    return ring_max(n - 1) + 1;
}

impl Sub<i64> for TileIndex {
    type Output = TileIndex;

    fn sub(self, rhs: i64) -> Self::Output {
        TileIndex(((self.value() as i64) - rhs) as u64)
    }
}

impl CCTile {
    pub fn new(tile_index: TileIndex) -> Self {
        HGSTile::new(tile_index).into()
    }

    pub fn make(tile_index: u64) -> Self {
        HGSTile::make(tile_index).into()
    }

    pub fn origin() -> CCTile {
        CCTile::from_qrs(0, 0, 0)
    }
    pub fn from_qrs(q: i64, r: i64, s: i64) -> CCTile {
        assert_eq!(
            q + r + s,
            0,
            "q + r + s does not sum up to zero for the tile ({q},{r},{s})!"
        );
        CCTile { q, r, s }
    }
    pub fn from_qr(q: i64, r: i64) -> CCTile {
        CCTile::from_qrs(q, r, 0 - q - r)
    }
    pub fn from_qrs_tuple(t: (i64, i64, i64)) -> CCTile {
        CCTile::from_qrs(t.0, t.1, t.2)
    }

    pub fn into_qrs_tuple(&self) -> (i64, i64, i64) {
        (self.q, self.r, self.s)
    }

    pub fn unit(direction: &RingCornerIndex) -> CCTile {
        CCTile::from_qrs_tuple(match direction {
            RingCornerIndex::RIGHT => (1, 0, -1),
            RingCornerIndex::TOPRIGHT => (1, -1, 0),
            RingCornerIndex::TOPLEFT => (0, -1, 1),
            RingCornerIndex::LEFT => (-1, 0, 1),
            RingCornerIndex::BOTTOMLEFT => (-1, 1, 0),
            RingCornerIndex::BOTTOMRIGHT => (0, 1, -1),
        })
    }

    pub fn units() -> impl Iterator<Item = CCTile> {
        RingCornerIndex::all().map(|rci| CCTile::unit(&rci))
    }

    /// `self` must be a *corner* of a ring with ring-index >= 2.
    /// Returns the corresponding direction.
    pub fn corner_to_direction(corner: &Self) -> RingCornerIndex {
        assert!(corner.is_corner());
        assert!(!corner.is_origin_tile());
        if corner.r == 0 {
            // We are on the r-axis, so it must be LEFT or RIGHT
            if corner.q > 0 {
                return RingCornerIndex::RIGHT;
            } else {
                return RingCornerIndex::LEFT;
            }
        }
        if corner.q == 0 {
            if corner.r < 0 {
                return RingCornerIndex::TOPLEFT;
            } else {
                return RingCornerIndex::BOTTOMRIGHT;
            }
        }
        if corner.s == 0 {
            if corner.q > 0 {
                return RingCornerIndex::TOPRIGHT;
            } else {
                return RingCornerIndex::BOTTOMLEFT;
            }
        }
        panic!("This can't happen. A non-origin corner tile has no axis set to 0.")
    }

    /// Returns `true`  iff the tile lies on the corner of a ring
    pub fn is_corner(&self) -> bool {
        self.q == 0 || self.r == 0 || self.s == 0
    }

    /// Adapted Euclidean Distance by
    /// Xiangguo Li
    ///
    /// <https://www.researchgate.net/publication/235779843_Storage_and_addressing_scheme_for_practical_hexagonal_image_processing>
    /// DOI <https://doi.org/10.1117/1.JEI.22.1.010502>
    ///
    /// I did not read much of the paper, but it's probably what you get when you splat the 3D-embedding cartesian coordinates to the 2D-plane.
    pub fn norm_euclidean(&self) -> f64 {
        let li = Self::origin().euclidean_distance_to(self);
        let redblob = f64::sqrt(self.norm_euclidean_sq() as f64);
        // I don't think this makes sense... why should this be equal?
        debug_assert_eq!(li, redblob as f64);
        redblob
    }

    pub fn norm_euclidean_sq(&self) -> i64 {
        self.q * self.q + self.r * self.r + self.q * self.r
    }

    /// The distance between this tile and the origin, in discrete steps.
    ///
    /// The blogpost <https://www.redblobgames.com/grids/hexagons/#distances-cube>
    /// explains this theoretically.
    /// Much easier is to observe:
    ///
    /// Along the corner axes, the distance to the origin is simply the max coordinate (absolute value).
    /// For the other entries in the ring, the maximum coordinate is the same value.
    ///
    ///
    /// Taking the maximum here is equal to computing
    /// ` (abs(q) + abs(r) + abs(s)) / 2`
    /// because we know that `q+r+s == 0` and we also know that two of the three coordinates must be of the same sign (pigeonhole principle). That means the third must be in absolute value as big as the two of the same sign. So the maximal abs is equivalent to the above.. qed.
    pub fn norm_steps(&self) -> u64 {
        self.max_coord()
    }

    fn max_coord(&self) -> u64 {
        let result = [self.q, self.r, self.s]
            .iter()
            .map(|coord| coord.abs() as u64)
            .max()
            .unwrap();
        assert!(result > 0 || *self == Self::from_qrs(0, 0, 0));
        result
    }

    /// Distance in discrete steps on the hexagonal grid.
    pub fn grid_distance_to(&self, other: &CCTile) -> u64 {
        (*self - *other).norm_steps()
    }

    /// Squared Euclidean Distance, see [Self::euclidean_distance_to]
    pub fn euclidean_distance_sq(&self, other: &CCTile) -> i64 {
        let dq = self.q - other.q;
        let dr = self.r - other.r;
        // let _li = dq * dq + dr * dr - dq * dr;
        let redblob = dq * dq + dr * dr + dq * dr;
        redblob
    }

    /// Euclidean Distance by Xiangguo Li
    /// adapted by redblobgames
    ///
    /// $$
    /// distance(a, b) := (a.q - b.q) ** 2 + (a.r - b.r) ** 2 + (a.q - b.q)(a.r - b.r)
    /// $$
    ///
    /// <https://www.researchgate.net/publication/235779843_Storage_and_addressing_scheme_for_practical_hexagonal_image_processing>
    /// DOI <https://doi.org/10.1117/1.JEI.22.1.010502>
    /// Equation 9 in section 3.3.
    ///
    /// Adaptation: <https://www.redblobgames.com/grids/hexagons/#distances-cube>
    ///
    /// Li uses "- dq * dr" in Eq. 9 while redblob uses + dq*dr.
    /// This is because the neighbors around (0,0) in Li 2013 are enumerated as
    /// `(0,1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)`
    /// We (and redblob) use instead a numbering that sums to zero or one (in the first ring), never to two.
    /// `(0,-1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)`
    /// It is evident that flipping our `r` axis' sign makes them equivalent.
    /// The sign flip does not matter in the squares, but propagates to the sign of `dr`.
    /// Hence we compute
    /// ```
    /// # let (dq, dr) = (1, 1);
    /// let redblob = dq * dq + dr * dr + dq * dr;
    /// ```
    /// instead of
    /// ```
    /// # let (dq, dr) = (1, 1);
    /// let _li = dq * dq + dr * dr - dq * dr;
    /// ```
    pub fn euclidean_distance_to(&self, other: &CCTile) -> f64 {
        f64::sqrt(self.euclidean_distance_sq(other) as f64)
    }

    pub fn is_origin_tile(&self) -> bool {
        self.q == 0 && self.r == 0 && self.s == 0
    }

    /// Which wedge of the grid the current tile is in.
    /// The wedges are defined as in <https://www.redblobgames.com/grids/hexagons/directions.html> in the first diagram.
    /// That is that the wedge borders go through the corners of the origin-tile, not through the corners of the HGS rings.
    /// Returns a second wedge if there are two wedges because the tile lies on their border.
    /// If so, the two wedges are sorted that the first is right before the second in counter-clockwise rotation.
    /// Returns no wedges if the tile is the origin.
    // TODO: Test this
    pub fn wedge_around_ringcorner(&self) -> Vec<RingCornerIndex> {
        // > Hex grids have six primary directions.
        // > Look at the max of |s-q|, |r-s|, |q-r|, and it will tell you which wedge you're in.
        // For the maximum, in `|a-b|` the `a` and `b` have opposite signs (except at 0,0,0).
        // If they are of equal abs value, we are on the border of a wedge. Inside the wedge, one of them is larger.
        // It's a bit more intuitive when using |q|, |r|, |s| instead but then the wedges would cross the corners.
        // TODO: understand this reasoning better.

        if self.is_origin_tile() {
            return vec![];
        }

        // positive: top-right
        let q_r = (self.q - self.r).abs();
        // positive: bottom-right
        let r_s = (self.r - self.s).abs();
        // positive: left
        let s_q = (self.s - self.q).abs();

        let rci_qr = if self.q - self.r > 0 {
            RingCornerIndex::TOPRIGHT
        } else {
            RingCornerIndex::BOTTOMLEFT
        };
        let rci_rs = if self.r - self.s > 0 {
            RingCornerIndex::BOTTOMRIGHT
        } else {
            RingCornerIndex::TOPLEFT
        };
        let rci_sq = if self.s - self.q > 0 {
            RingCornerIndex::LEFT
        } else {
            RingCornerIndex::RIGHT
        };
        let corner_indices = [rci_sq, rci_rs, rci_qr];

        let axes = [s_q, r_s, q_r];
        let mut sorted_indices = [0, 1, 2];
        sorted_indices.sort_by_key(|el| axes[*el]);
        let [_min_index, middle_index, max_index] = sorted_indices;
        let max: i64 = axes[sorted_indices[2]];
        let middle: i64 = axes[sorted_indices[1]];
        let min: i64 = axes[sorted_indices[0]];

        assert!(min <= max);
        if max > middle {
            return vec![corner_indices[max_index]];
        }

        assert_eq!(middle, max);
        // We have two wedges. Need to return them sorted consistently: the second should come ccw after the first.
        let edge =
            RingEdge::from_corners(&corner_indices[middle_index], &corner_indices[max_index]);
        return vec![edge.start(), edge.end()];
    }

    // The closest ring-corner. If tied, the previous corner in ccw direction (or the next corner in clockwise direction)
    // Might be this tile itself.
    // TODO: Test
    pub fn closest_corner_hgs(&self) -> HGSTile {
        assert!(!self.is_origin_tile());
        // get corner index
        let w = self.wedge_around_ringcorner()[0];
        // get ring radius
        let r = Ring::from(*self);
        HGSTile::new(r.corner(w))
    }

    // The previous ring-corner in ccw direction (or the next corner in clockwise direction)
    // Might be this tile itself.
    // TODO: Test
    pub fn previous_corner_cc(&self) -> CCTile {
        // TODO: This is used in converting from CCTile to HGSTile, so it may not rely on conversion to hgs.
        assert!(!self.is_origin_tile());
        let closest_corner = self.closest_corner_cc();
        if &closest_corner == self {
            return closest_corner;
        }
        let pre_closest_corner = closest_corner.rot60cw();
        // One of these must be the previous corner of self.
        // If closest_corner already is the previous corner, then the
        // pre_closest_corner will not be on the same ring-edge as self anymore.
        //
        // One the same ring-edge (and beyond, on the same line), one coordinate
        // remains constant. When we now take a turn on the ring (around a corner),
        // this coord will change.
        if self.are_colinear_with(&pre_closest_corner, &closest_corner) {
            return pre_closest_corner;
        }
        // All coordinates were different. The pre_corner is too far away.
        return closest_corner;
        // It can never happen that two coordinates match. Either one, zero, or three.

        // I believe this function could also be implemented by simply checking whether the pre_corner is colinear with the tile.
        // Which, again a belief, I think we can do simply by checking whether at least one coordinate is the same value.
        // This would need some testing though.
    }

    pub fn is_colinear_with(&self, other: &CCTile) -> bool {
        self.q == other.q || self.r == other.r || self.s == other.s
    }

    pub fn are_colinear_with(&self, other: &CCTile, third: &CCTile) -> bool {
        if other.q == third.q {
            if self.q == other.q {
                // We're on the same line as both corners, so
                // we should be in the middle of them.
                return true;
            }
        }
        if other.r == third.r {
            if self.r == other.r {
                // We're on the same line as both corners, so
                // we should be in the middle of them.
                return true;
            }
        }
        if other.s == third.s {
            if self.s == other.s {
                // We're on the same line as both corners, so
                // we should be in the middle of them.
                return true;
            }
        }
        return false;
    }

    // The closest ring-corner. If there are two, the previous in ccw direction (or the next corner in clockwise direction)
    // Might be this tile itself.
    pub fn closest_corner_cc(&self) -> CCTile {
        assert!(!self.is_origin_tile());
        // get corner index
        let w = self.wedge_around_ringcorner()[0];
        // get ring radius
        let r = Ring::from(*self);
        let direction = CCTile::unit(&w);
        direction * ((r.n.value() as i64) - 1)
    }

    /// Rotate 60 degrees counter-clockwise
    /// To rotate by other amounts, consider using spiral steps in HGSTile notation.
    /// <https://www.redblobgames.com/grids/hexagons/#rotation>
    pub fn rot60ccw(&self) -> CCTile {
        CCTile::from_qrs(-self.s, -self.q, -self.r)
    }
    /// Rotate 60 degrees clockwise
    /// <https://www.redblobgames.com/grids/hexagons/#rotation>
    pub fn rot60cw(&self) -> CCTile {
        CCTile::from_qrs(-self.r, -self.s, -self.q)
    }

    // TODO: Impement CC Reflection on all axes
    /// See <https://www.redblobgames.com/grids/hexagons/#reflection> for a
    /// visualization.
    ///
    /// This function reflects a tile across an axis.
    /// It can be ambiguous what an "axis" means, so to be clear:
    /// When reflecting along the q-Axis, we mean here that the q-coordinate stays constant.
    /// ```
    /// use hexgridspiral::CCTile;
    /// let tile = CCTile::from_qrs(4, -3, -1);
    /// let tile_r = tile.reflect_along_constant_axis(false, true, false);
    /// assert_eq!(tile_r, (-1, -3, 4).into());
    /// let tile_q = tile.reflect_along_constant_axis(true, false, false);
    /// assert_eq!(tile_q, (4, -1, -3).into());
    /// ```
    /// Applies multiple reflections if multiple orders are specified. In this case the order is Q,R,S such that
    /// ```
    /// use hexgridspiral::CCTile;
    /// let tile = CCTile::from_qrs(4, -3, -1);
    /// let tile_q = tile.reflect_along_constant_axis(true, false, false);
    /// let tile_qr = tile_q.reflect_along_constant_axis(false, true, false);
    /// let tile_qrs = tile_qr.reflect_along_constant_axis(false, false, true);
    /// let tile_direct_qrs = tile.reflect_along_constant_axis(true, true, true);
    /// assert_eq!(tile_qrs, tile_direct_qrs);
    /// ```
    /// So for readability it's probably better to just specify each seperately in sequence.
    pub fn reflect_along_constant_axis(&self, q_: bool, r_: bool, s_: bool) -> CCTile {
        // swap the coordinates that are not on the constant axis.
        let (mut q, mut r, mut s) = (self.q, self.r, self.s);
        if q_ {
            (r, s) = (s, r);
        }
        if r_ {
            (q, s) = (s, q);
        }
        if s_ {
            (q, r) = (r, q);
        }
        CCTile::from_qrs(q, r, s)
    }

    /// Reflect the tile to the other side of the line where the specified axis remains constant.
    /// Applies multiple times if multiple orders are specified, in order Q,R,S.
    /// > "To reflect over a line that's not at 0, pick a reference point on that line. Subtract the reference point, perform the reflection, then add the reference point back."
    /// > -- [](https://www.redblobgames.com/grids/hexagons/#reflection)
    // TODO: Read all comments on the redblobgames post.
    pub fn reflect_orthogonally_across_constant_axis(
        &self,
        q_: bool,
        r_: bool,
        s_: bool,
    ) -> CCTile {
        self.reflect_along_constant_axis(q_, r_, s_)
            .reflect_diagonally()
    }

    /// Reflects diagonally: along the line through this tile and the origin.
    /// The resulting tile lies on the opposite side of the origin, with the same distance.
    ///
    /// If it is unclear what this function does, check out <https://www.redblobgames.com/grids/hexagons/#reflection>.
    pub fn reflect_diagonally(&self) -> CCTile {
        CCTile::from_qrs(-self.q, -self.r, -self.s)
    }

    /// Reflects the `self` tile across the left-to-right axis that runs through the origin.
    /// ```rust
    /// let t = hexgridspiral::CCTile::from_qrs(2, -1, -1);
    /// let reflected = t.reflect_vertically();
    /// assert_eq!(reflected, (1, 1, -2).into());
    /// ```
    pub fn reflect_vertically(&self) -> CCTile {
        self.reflect_orthogonally_across_constant_axis(false, true, false)
    }

    /// Reflects the `self` tile across the top-to-bottom axis that runs through the origin.
    /// ```rust
    /// let t = hexgridspiral::CCTile::from_qrs(2, -1, -1);
    /// let reflected = t.reflect_horizontally();
    /// assert_eq!(reflected, (-1, -1, 2).into());
    /// ```
    pub fn reflect_horizontally(&self) -> CCTile {
        self.reflect_along_constant_axis(false, true, false)
    }

    /// Reflections like done with [Self::reflect_orthogonally_across_constant_axis],
    /// [Self::reflect_along_constant_axis], and [Self::reflect_diagonally], but with a reference point that is not necessarily the origin.
    ///
    /// This is done by subtracting a tile on the line (the reference point), reflecting as usual, and then adding the tile again.
    ///
    /// Usage will look like this:
    /// ```rust
    /// use hexgridspiral::CCTile;
    /// let tile1 = CCTile::from_qrs(1, -1, 0);
    /// // We want to reflect across the line where r == 1
    /// // so we choose an arbitrary tile on that line.
    /// let r = 1;
    /// let arbitrary_q = 10;
    /// let reference_point = CCTile::from_qr(arbitrary_q, r);
    ///
    /// let reflected_tile1: CCTile = tile1.reflect_with_reference_point(&reference_point,
    ///     |t: &CCTile | {t.reflect_orthogonally_across_constant_axis(false, true, false)}
    /// );
    /// assert_eq!(reflected_tile1, CCTile::from_qrs(-1, 3, -2));
    /// ```
    pub fn reflect_with_reference_point(
        &self,
        reference_point: &CCTile,
        reflection_fn: fn(&CCTile) -> CCTile,
    ) -> CCTile {
        let shifted = *self - *reference_point;
        let reflected = reflection_fn(&shifted);
        let unshifted = reflected + *reference_point;
        unshifted
    }

    pub fn spiral_steps(&self, steps: i64) -> Self {
        let ht: HGSTile = (*self).into();
        let ht2 = ht.spiral_steps(steps);
        ht2.into()
    }

    /// computes the coordinates in the plane as floats.
    /// * `unit_step` : The size of one step from a hex tile's center to its neighboring tile's center.
    /// The `unit_step` is twice the incircle radius of the hex. Or `sqrt(3) * outcircle_radius`.
    /// * `origin` : The pixel position of the origin-tile's center.
    ///
    /// The resulting pixel coordinates are in a system where positive x corresponds to [RingCornerIndex::RIGHT] and
    /// positive y corresponds to the UP-direction between ([RingCornerIndex::TOPLEFT] and [RingCornerIndex::TOPRIGHT])..
    pub fn to_pixel(&self, origin: (f64, f64), unit_step: f64) -> (f64, f64) {
        // We have point-top hexes. Call the hex inner-radius iR and the hex outer-radius oR.
        // The width (aka iR) of a hex equals sqrt(3)/2 * height (aka oR).
        // On the redblobgames website, they call the "size" what I'd call oR.
        // On the x-axis, two centers are exactly 2*iR away.
        // On the y-axis, they are not on the same x-coord, so it's different.
        // There they are 3/4 * oR away from each other.
        // oR is the outer radius of the hex circumcircle; or the edge length.
        // The unit_step is two times the height of a equilateral triangle, so
        // unit_step = sqrt(3.)/2*oR*2
        let outer_radius = unit_step / f64::sqrt(3.);
        let _redblob_size = outer_radius;
        // Walk both unit vectors.
        // Math according to https://www.redblobgames.com/grids/hexagons/#hex-to-pixel-axial
        // the unit vectors are, relative to the hex-edge length,
        // q_unit := (x = sqrt(3), y = 0) and r_unit := ( x= sqrt(3)/2, y=-3/2)
        // corresponding to q and r vectors from the origin tile to the next tile in pixels.
        // Note that the y in the r_unit basis vector should be negative because I make y go upwards, not downwards.

        // We can observe the following on an example or by looking at our unit vectors:
        // One step of incrementing q is one step along the x-axis.
        // But moving along the x-axis does not change r ... why does it contribute here?
        // Because it matters: One step of incrementing r, given a fixed q. That is also half a step along the x axis.
        let x = unit_step * ((self.q as f64) + (self.r as f64) / 2.);

        // Expressing the basis vectors in terms of unit_step:
        // One step of incrementing q given a fixed r is half a step along the y axis.
        // One step of incrementing r given a fixed q is some fraction of a step along the negative y axis.
        // To compute the fraction, look at the triangle (hex1 center, hex2 center, y-axis).
        // We know the hypotenuse (`c`) is unit_step long.
        // And the x-axis is given above.
        // Then dy = sqrt(dx^2 + c^2)
        // Or, because this is in a equilateral triangle,
        // the dy is the height thereof, so sqrt(3.)/2 * unit_step.
        // mine2:
        let y = f64::sqrt(3.) / 2. * unit_step * (-self.r as f64);

        // redblob: Equivalent to my "mine2" formula, because `redblob_size * sqrt(3) = unit_step`:
        // Unit_step is twice the height of the equilateral triangle spanned by two oR and an edge.
        //let y = redblob_size * 3. / 2. * (-self.r as f64);

        return (origin.0 + x, origin.1 + y);
    }

    pub fn hgs(&self) -> HGSTile {
        self.into()
    }
    pub fn spiral_index(&self) -> TileIndex {
        self.hgs().h
    }

    /// Wrapper around [CCTile::from_pixel_] with unit_step of size 1.
    pub fn from_pixel(pixel: (f64, f64)) -> Self {
        Self::from_pixel_(pixel, (0., 0.), 1.)
    }

    /// Inversion of the formula in [CCTile::to_pixel] and rounding to the nearest hex tile.
    /// `unit_step` determines the scaling - it is the distance between two adjacent hex tile centers, in pixel units.
    pub fn from_pixel_(pixel: (f64, f64), origin: (f64, f64), unit_step: f64) -> Self {
        let x = pixel.0 - origin.0;
        let y = pixel.1 - origin.1;
        // Based on to_pixel formulas:
        // let x = unit_step * ((self.q as f64) + (self.r as f64) / 2.);
        // let y = f64::sqrt(3.)/2.*unit_step*(-self.r as f64);
        let r = -2. / f64::sqrt(3.) * y / unit_step;
        let q = x / unit_step - (r / 2.);
        Self::round_to_nearest_tile(q, r)
    }

    // https://www.redblobgames.com/grids/hexagons/#rounding
    pub fn round_to_nearest_tile(frac_q: f64, frac_r: f64) -> Self {
        // infer s from the sum-to-zero constraint
        let frac_s: f64 = 0. - frac_q - frac_r;
        // round to nearest integer
        let mut q = f64::round(frac_q);
        let mut r = f64::round(frac_r);
        let mut s = f64::round(frac_s);

        // find error
        let q_diff = f64::abs(q - frac_q);
        let r_diff = f64::abs(r - frac_r);
        let s_diff = f64::abs(s - frac_s);

        // fix the largest error coordinate
        if q_diff > r_diff && q_diff > s_diff {
            q = -r - s
        } else if r_diff > s_diff {
            r = -q - s
        } else {
            s = -q - r
        };

        Self::from_qrs(q as i64, r as i64, s as i64)
    }

    /// Compute all reachable tiles in `steps`.
    /// Explained in <https://www.redblobgames.com/grids/hexagons/#range-coordinate> .
    pub fn movement_range(&self, steps: u64) -> MovementRange {
        let n: i64 = steps as i64;
        MovementRange {
            q_min: self.q - n,
            q_max: self.q + n,
            r_min: self.r - n,
            r_max: self.r + n,
            s_min: self.s - n,
            s_max: self.s + n,
        }
    }

    /// How big a single row step in the direction [RingCornerIndex::TOPRIGHT] is in pixels.
    /// - `unit_step`: The step size between two adjacent tile centers, in pixels (floats, not manhattan distance).
    pub fn pixel_step_vertical(unit_step: f64) -> (f64, f64) {
        // Notably, the step down is (oR + 0.5*edge), equals (1.5 * edge),
        // equals (1.5 * 2 iR/sqrt(3.)), equals (1.5 * unit_step/sqrt(3.)),
        // equals (unit_step* 3/2 /sqrt(3.)),
        // equals (unit_step * sqrt(3.) / 2).
        (unit_step, unit_step * f64::sqrt(3.) / 2.)
    }
}

/// <https://www.redblobgames.com/grids/hexagons/#range-coordinate>
#[derive(Copy, Clone, Debug)]
pub struct MovementRange {
    q_min: i64,
    q_max: i64,
    r_min: i64,
    r_max: i64,
    s_min: i64,
    s_max: i64,
}
impl MovementRange {
    pub fn around(tile: &CCTile, steps: i64) -> Self {
        let n: i64 = i64::abs(steps);
        MovementRange {
            q_min: tile.q - n,
            q_max: tile.q + n,
            r_min: tile.r - n,
            r_max: tile.r + n,
            s_min: tile.s - n,
            s_max: tile.s + n,
        }
    }

    pub fn intersect(&self, other: &MovementRange) -> MovementRange {
        // The intersection in one direction is [max_lower, min_upper].
        MovementRange {
            q_min: i64::max(self.q_min, other.q_min),
            q_max: i64::min(self.q_max, other.q_max),
            r_min: i64::max(self.r_min, other.r_min),
            r_max: i64::min(self.r_max, other.r_max),
            s_min: i64::max(self.s_min, other.s_min),
            s_max: i64::min(self.s_max, other.s_max),
        }
    }

    pub fn contains(&self, tile: &CCTile) -> bool {
        tile.q <= self.q_max
            && tile.r <= self.r_max
            && tile.s <= self.s_max
            && self.q_min <= tile.q
            && self.r_min <= tile.r
            && self.s_min <= tile.s
    }

    cfg_if::cfg_if! {if #[cfg(feature = "nightly")] {
        #[cfg(any(feature = "nightly", doc))]
        #[doc(cfg(feature = "nightly"))]
        pub fn count_tiles(&self) -> usize {
            let mut count = 0;
            for _ in *self {
                count += 1;
            }
            return count;
        }
    }
    }
}

cfg_if::cfg_if! {if #[cfg(feature = "nightly")] {
#[cfg(any(feature = "nightly", doc))]
#[doc(cfg(feature = "nightly"))]
impl IntoIterator for MovementRange {
    type Item = CCTile;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let it = std::iter::from_coroutine(
            #[coroutine]
            move || {
                // We would loop over all values of q, r, and s, but filter out the ones that do not satisfy
                // our usual constraint ` q + r + s == 0`. This loop computes s to ensure this is fulfilled.
                //
                // We loop from min to max around our center... that would be from -N to +N. But then
                // there are some tiles we accidentally cover.
                // Along the axes, the value can deviate at most N from the center.
                // But we know that the extreme value of r on the axis where q is constant will be larger than
                // toward the top or bottom of our (hexagonal) range.
                // So if q is smaller, r may be larger, otherwise not.
                for q in (self.q_min)..=self.q_max {
                    // We loop from -N to N, so we only need to cover one side per iteration.

                    // choose the less extreme minimum
                    let r_min = i64::max(self.r_min, -q - self.s_max);
                    // choose the less extreme maximum
                    let r_max = i64::min(self.r_max, -q - self.s_min);
                    for r in r_min..=r_max {
                        #[cfg(feature = "nightly")]
                        yield CCTile::from_qr(q, r);
                    }
                }
            },
        );
        it
    }
}
}}

// Conversion from HexGridSpiral to Cube Coordinates:
// We can easily get the previous corner.
// Then add the rest.
impl From<HGSTile> for CCTile {
    fn from(item: HGSTile) -> Self {
        (&item).into()
    }
}

impl From<&HGSTile> for CCTile {
    fn from(item: &HGSTile) -> Self {
        if item.is_origin_tile() {
            return CCTile::origin();
        }
        let edge_section = item.ring_edge();
        let corner_index = edge_section.start();
        // Find the previous corner in Cube Coordinates
        // This part is straightforward because it's simply moving along an axis
        let cc_axis_unit = CCTile::unit(&corner_index);
        let cc_edge_start = cc_axis_unit * (item.ring.radius() as i64);
        // Then make steps in the edge direction
        let cc_edge_unit = CCTile::unit(&edge_section.direction());
        let corner_h = item.ring.corner(corner_index);
        if corner_h > item.h {
            // corner_h is the maximum in the ring.
            let fake_corner_h = corner_h - item.ring.size();
            let steps_from_corner = item.h - fake_corner_h;
            return cc_edge_start + cc_edge_unit * (steps_from_corner.value() as i64);
        }
        let steps_from_corner = item.h - corner_h;
        return cc_edge_start + cc_edge_unit * (steps_from_corner.value() as i64);
    }
}

impl From<&CCTile> for Ring {
    fn from(item: &CCTile) -> Self {
        Ring::new(RingIndex(item.norm_steps() + 1))
    }
}

impl From<CCTile> for Ring {
    fn from(item: CCTile) -> Self {
        (&item).into()
    }
}

impl From<&CCTile> for HGSTile {
    fn from(item: &CCTile) -> Self {
        let ring = Ring::from(item);
        let ring_index = ring.n;
        if ring.n == RingIndex::ORIGIN_RING {
            return HGSTile::new(TileIndex(0));
        }
        assert!(
            !item.is_origin_tile(),
            "This can not happen, the ring index of {item:?} is not 1. It is {ring_index:?}"
        );
        let wedges = item.wedge_around_ringcorner();
        // If the item tile is a ring-corner, there will only be one wedge. Otherwise, if it is the border of a wedge, there might be two.
        // TODO: does rust do runtime checks at release build runtime for these vecs? Just out of curiosity.
        let closest_corner_hgs = HGSTile::new(ring.corner(wedges[0]));
        // If there are two wedges, the tile lies on the diagonal axes that lie between the usual CC grid axes.
        if wedges.len() == 2 {
            // This can only happen on rings with odd full edgelengths, otherwise there is no tile on the wedge border.
            assert_eq!(ring.full_edge_size() % 2, 1);
            let corner0_hgs = closest_corner_hgs;
            let corner1_hgs = HGSTile::new(ring.corner(wedges[1]));
            // corner0_hgs is in ccw order right before corner1_hgs.
            // If corner0_hgs is the ring-maximum, we have to special-case the addition to stay in the ring.
            if corner1_hgs.h < corner0_hgs.h {
                return HGSTile::new(corner0_hgs.ring_min().h + ring.edge_size() / 2 - 1);
            } else {
                return HGSTile::new(corner0_hgs.h + ring.edge_size() / 2);
            }
        }
        assert_eq!(wedges.len(), 1);
        // We have the corner in the middle of this wedge (corner0). How do we get the correct tile's hsg index?
        let previous_corner = &item.previous_corner_cc();
        let offset_along_edge_hgs = item.grid_distance_to(previous_corner);

        // We need to find the previous_corner's hgs index.
        // We know it is a corner, so we can look up it's orientation.
        let rci = CCTile::corner_to_direction(previous_corner);
        let previous_corner_hgs = HGSTile::new(ring.corner(rci));
        return HGSTile::new(previous_corner_hgs.h + offset_along_edge_hgs);
    }
}

// Conversion from Cube Coordinates to HexGridSpiral:
// TODO: Thoroughly test this.
impl From<CCTile> for HGSTile {
    fn from(item: CCTile) -> Self {
        (&item).into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_multiplication_with_unit() {
        let u = CCTile::unit(&RingCornerIndex::RIGHT);
        let two_u_mul = u * 2;
        let two_u_add = u + u;
        assert_eq!(two_u_mul, two_u_add);
        println!("Testing Tests works.");
    }

    #[test]
    fn test_hexgridspiral_cc_conversion() {
        let origin = CCTile::from_qr(0, 0);
        let tile0: HGSTile = origin.into();
        let h0 = tile0.h;
        assert_eq!(h0, TileIndex(0));

        let o_cc: CCTile = CCTile::from(HGSTile::make(9));
        let nine = CCTile::from_qr(1, -2);
        assert_eq!(nine, o_cc);
    }

    #[test]
    fn test_cc_hexgridspiral_conversion() {
        let origin = HGSTile::make(0);
        let tile0: CCTile = origin.into();
        assert_eq!(tile0, CCTile::origin());

        // after corner
        let nine = CCTile::from_qr(1, -2);
        let nine_to_hgs = nine.hgs();
        assert_eq!(nine_to_hgs, HGSTile::make(9));

        // before corner
        let seven2 = CCTile::from_qr(2, -1);
        assert_eq!(seven2.hgs(), HGSTile::make(7));

        let seven = CCTile::from_qrs(2, -1, -1);
        assert_eq!(seven.hgs(), HGSTile::make(7));

        // corner
        let eight = CCTile::from_qrs(2, -2, 0);
        assert_eq!(eight.hgs(), HGSTile::make(8));

        // before a corner, and not in the wedge of the previous corner
        let tile_a = CCTile::from_qr(1, -3);
        assert_eq!(tile_a.hgs(), HGSTile::make(23));
    }

    #[test]
    fn test_hexcount_steps_from_zero() {
        let start = HGSTile::new(TileIndex(0));
        let mut x1 = start.increment_spiral();
        for _i in 1..6 {
            x1 = x1.increment_spiral();
        }
        // we did six steps, so should be same as skipping one ring.
        let x2 = start.ring.next_ring().max();
        assert!(x1.h == x2, "Six spiral-steps should equal one ring-step in the first non-origin ring, but we got {x1} and {x2}.");
        assert!(x2.value() == 6);
    }

    #[test]
    fn test_hexcount_steps_from_one() {
        let start = HGSTile::new(TileIndex(1));
        let mut x1 = start.increment_spiral();
        for _i in 1..6 {
            x1 = x1.increment_spiral();
        }
        // we did six steps from ring_min, so should be same as skipping one ring.
        let x2 = start.ring.next_ring().min();
        assert_eq!(x1.h, x2);
    }

    #[test]
    fn test_hexcount_steps_from_seven() {
        let start = HGSTile::new(TileIndex(7));
        let mut x1 = start;
        for _i in 0..start.ring.size() {
            x1 = x1.increment_spiral();
        }
        // we did six steps from ring_min, so should be same as skipping one ring.
        let x2 = start.ring.next_ring().min();
        assert_eq!(x1.h, x2);
    }

    #[test]
    fn test_ring_max_min() {
        let min1 = ring_min(RingIndex::ORIGIN_RING.into());
        let max1 = ring_max(RingIndex::ORIGIN_RING.into());
        assert!(min1.value() == 0);
        assert!(max1.value() == 0);

        let min2 = ring_min(2.into());
        let max2 = ring_max(2.into());
        assert!(min2.value() == 1);
        assert!(max2.value() == 6, "Should be 6 but was {max1}");

        let min3 = ring_min(3.into());
        let max3 = ring_max(3.into());
        assert_eq!(min3.value(), 7);
        assert_eq!(max3.value(), 18);
    }

    #[test]
    fn test_ring_size() {
        for n in 1..10 {
            let ring = Ring::new(n.into());
            assert_eq!(ring.size(), ring.max().value() - ring.min().value() + 1);
        }
    }

    #[test]
    fn test_ring_index_for_min() {
        let a = ring_index_for_tile_index(TileIndex(0));
        let b = ring_index_for_tile_index(TileIndex(1));
        let c = ring_index_for_tile_index(TileIndex(7));
        assert_eq!(a, RingIndex(1));
        assert_eq!(b, RingIndex(2));
        assert_eq!(c, RingIndex(3));
    }

    #[test]
    fn test_ring_index_for_max() {
        let a = ring_index_for_tile_index(TileIndex(0));
        assert_eq!(a, RingIndex(1));

        let b = ring_index_for_tile_index(TileIndex(6));
        assert_eq!(b, RingIndex(2));

        let d = ring_index_for_tile_index(TileIndex(18));
        assert_eq!(d, RingIndex(3));

        let c = ring_index_for_tile_index(TileIndex(36));
        assert_eq!(c, RingIndex(4));
    }

    #[test]
    fn test_ring_index_for_any() {
        for h in 1..=6 {
            let b = ring_index_for_tile_index(TileIndex(h));
            assert_eq!(b, RingIndex(2));
        }

        for h in 19..=35 {
            let d = ring_index_for_tile_index(TileIndex(h));
            assert_eq!(d, RingIndex(4));
        }
    }

    #[test]
    fn test_ring_max() {
        let a = ring_max(RingIndex(1));
        let b = ring_max(RingIndex(2));
        let c = ring_max(RingIndex(3));
        assert_eq!(a, TileIndex(0));
        assert_eq!(b, TileIndex(6));
        assert_eq!(c, TileIndex(18));
    }

    #[test]
    fn test_wedge_around_ringcorner1() {
        // innermost ring
        let topright = CCTile::from_qr(1, -1);
        let rci_vec = topright.wedge_around_ringcorner();
        assert_eq!(rci_vec.len(), 1);
        let mut rci = rci_vec[0];
        assert_eq!(rci, RingCornerIndex::TOPRIGHT);

        rci = rci.next();
        let topleft = topright + CCTile::unit(&RingCornerIndex::LEFT);
        let rci2_vec = topleft.wedge_around_ringcorner();
        assert_eq!(rci2_vec.len(), 1);
        assert_eq!(rci2_vec[0], rci);
    }

    #[test]
    fn test_wedge_around_ringcorner2() {
        // bottom right, not on the diagonal of the wedge border
        let tile = CCTile::from_qrs(1, 3, -4);
        let rci_vec = tile.wedge_around_ringcorner();
        assert_eq!(rci_vec.len(), 1);
        let rci = rci_vec[0];
        assert_eq!(rci, RingCornerIndex::BOTTOMRIGHT);
    }

    #[test]
    fn test_wedge_around_ringcorner3() {
        // bottom left, on the diagonal where two wedges meet
        let tile = CCTile::from_qrs(-4, 2, 2);
        let rci_vec = tile.wedge_around_ringcorner();
        assert_eq!(rci_vec.len(), 2);
        // the first result should be ccw before the second
        assert_eq!(rci_vec[0], RingCornerIndex::LEFT);
        assert_eq!(rci_vec[1], RingCornerIndex::BOTTOMLEFT);
    }

    cfg_if::cfg_if! {if #[cfg(feature = "nightly")] {
    #[test]
    fn test_ring_index_for_tile_index_small() {
        for ring_index in 1..3 {
            let ring = Ring::new(ring_index.into());
            assert!(ring.n.value() > 0);
            for tile_index in ring.min()..=ring.max() {
                let ring_index_returned = ring_index_for_tile_index(tile_index);
                assert_eq!(ring_index, ring_index_returned.value());
            }
        }
    }
    }}

    #[test]
    fn test_ring_index_construction_speed() {
        for ring_index in 100000..120000 {
            // test a few random tiles
            let ring = Ring::new(ring_index.into());
            let ring_index_returned = ring_index_for_tile_index(ring.min());
            assert_eq!(ring_index, ring_index_returned.value());
        }

        for ring_index in 100000..120000 {
            // test a few random tiles
            let ring = Ring::new(ring_index.into());
            let ring_index_returned = ring_index_for_tile_index(ring.max());
            assert_eq!(ring_index, ring_index_returned.value());
        }

        for ring_index in 100..120 {
            // test a few random tiles
            let ring = Ring::new(ring_index.into());
            let mid = ring.min() + (ring.size() / 2);
            let ring_index_returned = ring_index_for_tile_index(mid);
            assert_eq!(ring_index, ring_index_returned.value());
        }
    }
    #[test]
    fn test_ring_index_for_tile_index_big() {
        let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(40);
        for ring_index in 100..120 {
            // test a few random tiles
            let ring = Ring::new(ring_index.into());
            let tile_index = ring.random_tile_in_ring(&mut rng);
            let ring_index_returned = ring_index_for_tile_index(tile_index);
            assert_eq!(ring_index, ring_index_returned.value());
        }
    }

    #[test]
    fn test_norm_steps() {
        let u1 = CCTile::unit(&RingCornerIndex::RIGHT);
        assert_eq!(u1.norm_steps(), 1);

        let u2 = u1 + u1;
        assert_eq!(u2.norm_steps(), 2);

        let t1 = CCTile::from_qrs(1, -2, 1);
        let t2 = CCTile::from_qrs(2, -2, 0);
        let t3 = CCTile::from_qrs(0, 2, -2);
        let n1 = t1.norm_steps();
        assert_eq!(n1, 2);
        assert_eq!(t2.norm_steps(), n1);
        assert_eq!(t3.norm_steps(), n1);

        assert_eq!(CCTile::from_qr(0, 0).norm_steps(), 0);

        // test several rings
        for ring_step in 1..4 {
            // tiles located opposite each other
            for direction in CCTile::units() {
                let tile_a = direction * ring_step;
                let tile_b = direction * -ring_step;
                assert_eq!(tile_a.norm_steps(), tile_b.norm_steps());
                assert_eq!(tile_a.norm_steps(), ring_step.try_into().unwrap());
            }
        }

        let tile35 = HGSTile::make(35).cc();
        assert_eq!(tile35.norm_steps(), 3);
    }

    #[test]
    fn test_ring_edge_is_ordered_ccw() {
        let ring_edge = RingEdge::from_corners(&RingCornerIndex::RIGHT, &RingCornerIndex::TOPRIGHT);
        assert_eq!(
            vec![ring_edge.start(), ring_edge.end()],
            vec![RingCornerIndex::RIGHT, RingCornerIndex::TOPRIGHT]
        );

        let ring_edge =
            RingEdge::from_corners(&RingCornerIndex::LEFT, &RingCornerIndex::BOTTOMLEFT);
        assert_eq!(
            vec![ring_edge.start(), ring_edge.end()],
            vec![RingCornerIndex::LEFT, RingCornerIndex::BOTTOMLEFT]
        );

        let ring_edge2 =
            RingEdge::from_corners(&RingCornerIndex::BOTTOMRIGHT, &RingCornerIndex::BOTTOMLEFT);
        assert_eq!(
            vec![ring_edge2.start(), ring_edge2.end()],
            vec![RingCornerIndex::BOTTOMLEFT, RingCornerIndex::BOTTOMRIGHT]
        );

        // More generally:
        for rci_a in RingCornerIndex::all() {
            let rci_b = rci_a.next();
            let ring_edge = RingEdge::from_corners(&rci_a, &rci_b);
            assert_eq!(ring_edge.start(), rci_a);
            assert_eq!(ring_edge.end(), rci_b);

            let ring_edge2 = RingEdge::from_corners(&rci_b, &rci_a);
            assert_eq!(ring_edge2.start(), rci_a);
            assert_eq!(ring_edge2.end(), rci_b);
        }
    }

    #[test]
    fn test_ring_edge() {
        let r0 = RingEdge(0);
        assert_eq!(r0.start(), RingCornerIndex::RIGHT);
        assert_eq!(r0.end(), RingCornerIndex::TOPRIGHT);

        let r1 = RingEdge::from_primitive(1);
        assert_eq!(
            vec![r1.start(), r1.end()],
            vec![RingCornerIndex::TOPRIGHT, RingCornerIndex::TOPLEFT]
        );

        let r2 = RingEdge::from_primitive(2);
        assert_eq!(
            vec![r2.start(), r2.end()],
            vec![RingCornerIndex::TOPLEFT, RingCornerIndex::LEFT]
        );

        let r3 = RingEdge::from_primitive(3);
        assert_eq!(
            vec![r3.start(), r3.end()],
            vec![RingCornerIndex::LEFT, RingCornerIndex::BOTTOMLEFT]
        );

        let r4 = RingEdge::from_primitive(4);
        assert_eq!(
            vec![r4.start(), r4.end()],
            vec![RingCornerIndex::BOTTOMLEFT, RingCornerIndex::BOTTOMRIGHT]
        );

        let r5 = RingEdge::from_primitive(5);
        assert_eq!(
            vec![r5.start(), r5.end()],
            vec![RingCornerIndex::BOTTOMRIGHT, RingCornerIndex::RIGHT]
        );

        let r6 = RingEdge::from_primitive(6);
        assert_eq!(
            vec![r6.start(), r6.end()],
            vec![RingCornerIndex::RIGHT, RingCornerIndex::TOPRIGHT]
        );
    }

    #[test]
    fn test_ring_edge_from_corners() {
        let r0 = RingEdge::from_primitive(0);
        assert_eq!(
            vec![r0.start(), r0.end()],
            vec![RingCornerIndex::RIGHT, RingCornerIndex::TOPRIGHT]
        );

        let ring_corner0: RingCornerIndex = RingCornerIndex::try_from_primitive(0).unwrap();
        let ring_corner1: RingCornerIndex = RingCornerIndex::try_from_primitive(1).unwrap();
        let r0c = RingEdge::from_corners(&ring_corner0, &ring_corner1);
        assert_eq!(
            vec![r0c.start(), r0c.end()],
            vec![RingCornerIndex::RIGHT, RingCornerIndex::TOPRIGHT]
        );

        let r0c2 = RingEdge::from_corners(&ring_corner1, &ring_corner0);
        assert_eq!(
            vec![r0c2.start(), r0c2.end()],
            vec![RingCornerIndex::RIGHT, RingCornerIndex::TOPRIGHT]
        );
    }

    #[test]
    fn test_ring_corner_index_all() {
        let a = RingCornerIndex::all().count();
        assert_eq!(a, 6);
    }

    #[test]
    fn test_rot60_cc() {
        // test all corners
        let mut i = 0;
        for r in RingCornerIndex::all_from(RingCornerIndex::BOTTOMLEFT) {
            i += 1;
            let t = CCTile::unit(&r) * i;
            let q = CCTile::unit(&r.next()) * i;
            assert_eq!(t.rot60ccw(), q);
            assert_eq!(q.rot60cw(), t);
        }

        // Also test some not-corner tiles
        for h in (1..100).step_by(7) {
            let not_corner: CCTile = HGSTile::new(TileIndex(h)).into();
            let a = not_corner.rot60cw().rot60cw().rot60cw();
            let b = not_corner.rot60ccw().rot60ccw().rot60ccw();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_euclidean_distance() {
        let tile0 = HGSTile::make(0);
        let tile1 = CCTile::unit(&RingCornerIndex::RIGHT);
        assert_eq!(
            tile1.euclidean_distance_to(&Into::<CCTile>::into(tile0)),
            1.
        );

        let tile2 = CCTile::unit(&RingCornerIndex::TOPRIGHT);
        assert_eq!(tile1.euclidean_distance_to(&tile1), 0.);
        assert_eq!(tile1.euclidean_distance_to(&tile2), 1.);
        assert_eq!(tile2.euclidean_distance_to(&tile1), 1.);

        let tile7: CCTile = HGSTile::make(7).into();
        let tileo: CCTile = CCTile::origin();
        assert_eq!(tile7.euclidean_distance_to(&tile1), 1.);
        assert_eq!(tile7.euclidean_distance_sq(&tileo), 3);

        let tile8 = CCTile::make(8);
        assert_eq!(tile8.euclidean_distance_to(&tileo), 2.);
        assert_eq!(tile8.euclidean_distance_to(&tile7), 1.);

        let tile_minus_8 = tile8 * (-1);
        // this tile should have the same norm to the origin
        let norm8 = tile8.norm_euclidean();
        let norm8_minus = tile_minus_8.norm_euclidean();
        assert_eq!(norm8, norm8_minus);
        let norm_distance = tile8.euclidean_distance_to(&tile_minus_8);
        assert_eq!(norm_distance, 2. * norm8);
    }

    #[test]
    fn test_conversion_to_pixel() {
        {
            let tile1_cc = CCTile::make(1);
            assert_eq!(tile1_cc, CCTile::from_qr(1, -1));
            let tile1_px = tile1_cc.to_pixel((0., 0.), 1.);
            assert_eq!(tile1_px, (0.5, f64::sqrt(3.) / 2.));
        }

        let tile1_cc = CCTile::unit(&RingCornerIndex::RIGHT);
        let tile1_px = tile1_cc.to_pixel((0., 0.), 1.);
        assert_eq!(tile1_px, (1., 0.));

        let tile_right = tile1_cc * 5;
        assert_eq!(tile_right.to_pixel((1., 2.), 1.), (6., 2.));

        // Same at larger scale:
        assert_eq!(tile1_cc.to_pixel((0., 0.), 10000.), (10000., 0.));

        // Same with a tile that is not at r == 0:
        let tile2 = CCTile::unit(&RingCornerIndex::TOPLEFT);
        assert_eq!(tile2, CCTile::from_qr(0, -1));
        let tile2_px = tile2.to_pixel((0., 0.), 1.);
        let inner_radius = 0.5;
        let outer_radius = 2. / f64::sqrt(3.) * inner_radius;
        let y_should = 1.5 * outer_radius;
        assert_approx_eq!(tile2_px.0, -0.5);
        assert_approx_eq!(tile2_px.1, y_should);
        // Sanity Check that the library is not always saying it's approximately equal
        assert!(f64::abs(tile2_px.1 - y_should) < f64::EPSILON * 5.);

        // Test with a tile where both q and r are relevant, and origin and unit step are non-default.
        let tile35 = HGSTile::make(35).cc();
        let pixel35 = tile35.to_pixel((2., 3.), 4.);
        // Tile 35, aka (3, 1, -4), is 2.5 steps to the right and one step down.
        let pxstep = CCTile::pixel_step_vertical(4.);
        let should_be_pixel_35 = (2. + 2.5 * pxstep.0, 3. - pxstep.1);
        assert_eq!(pixel35, should_be_pixel_35);
    }

    #[test]
    fn test_conversion_from_pixel_and_back2() {
        let origin = (0., 0.);
        // TODO: What about other unit step sizes?
        let unit_step = 1.;

        {
            let h = TileIndex(7);
            let tile = CCTile::new(h);
            let expected_pixel_x = unit_step * 1.5;
            let expected_pixel_y = unit_step * f64::sqrt(3.) / 2.;
            let pixel = tile.to_pixel(origin, unit_step);
            assert_approx_eq!(pixel.0, expected_pixel_x);
            assert_approx_eq!(pixel.1, expected_pixel_y);
            let tile2 = CCTile::from_pixel(pixel);
            assert_eq!(tile, tile2);
        }
    }

    #[test]
    fn test_conversion_from_pixel() {
        let origin = (0., 0.);
        let unit_step = 1.;

        // Origin should be at (0,0)
        {
            let tile = CCTile::make(0);
            assert_eq!(tile.to_pixel(origin, unit_step), (0., 0.));
            assert_eq!(CCTile::from_pixel((0., 0.)), tile);
        }

        {
            let tile1 = CCTile::make(1);
            assert_eq!(CCTile::from_pixel((0.5, f64::sqrt(3.) / 2.)), tile1);
        }

        // First-ring tile
        //
        // Imagine same-sided triangles in the hex.
        // triangle_height = sqrt(3)/2. * triangle_edge
        // unit_step = 2*triangle_height
        //
        // Now imagine a same-sided triangle formed by three hexes.
        // y = big_triangle_height = sqrt(3)/2. * big_triangle_edge
        //   = sqrt(3)/2. * unit_step
        {
            let tile = CCTile::make(1);
            assert_eq!(tile, CCTile::from_qr(1, -1));
            let pixel = tile.to_pixel(origin, unit_step);
            assert_approx_eq!(pixel.0, 0.5);
            assert_approx_eq!(pixel.1, f64::sqrt(3.) / 2. * unit_step);
        }
    }

    #[test]
    fn test_conversion_from_pixel_and_back() {
        let origin = (0., 0.);
        // TODO: What about other unit step sizes?
        let unit_step = 1.;
        // TODO: Test to create from_pixel with other unit step and origin?
        // Test that to and from pixel are consistent.
        for h in [0, 1, 2, 4, 5, 6, 8, 10, 27, 100] {
            let tile = CCTile::make(h);
            let pixel = tile.to_pixel(origin, unit_step);
            let tile2 = CCTile::from_pixel(pixel);
            assert_eq!(tile, tile2);
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_movement_range() {
        for h in [0, 1, 2, 4, 5, 6, 7, 10, 27, 100] {
            let o = CCTile::make(h);
            for max_steps in 0..10 {
                let m = o.movement_range(max_steps);
                // Everything in the range must be close enough.
                for tile in m {
                    assert!(tile.grid_distance_to(&o) <= max_steps);
                }
                // Everything close enough must be in the range.
                // We check this by ensuring that the number of entries is correct.
                let collected: Vec<CCTile> = m.into_iter().collect();
                let num_tiles = collected.len() as u64;
                // num_tiles should equal the number of differently-sized rings' tiles.
                // `1 +  0 + 6 * 1 + 6 * 2 + ... + 6 * num_steps`
                // which is equal to `6 * sum_from_0_to_num_steps`.
                let s = 1 + 6 * ((0..=max_steps).sum::<u64>());
                assert_eq!(s, num_tiles);
            }
        }
    }

    #[test]
    #[cfg(feature = "nightly")]
    fn test_movement_range_intersection() {
        let tile_a = CCTile::origin() + CCTile::unit(&RingCornerIndex::LEFT);
        let tile_b = CCTile::origin() + CCTile::unit(&RingCornerIndex::RIGHT) * 2;
        //_ _ A O _ B _ _
        let range_a = tile_a.movement_range(2);
        assert!(range_a.contains(&tile_a));
        assert_eq!(range_a.contains(&CCTile::origin()), true);
        assert_eq!(range_a.count_tiles(), 19);

        let range_b = tile_b.movement_range(1);
        assert!(range_b.contains(&tile_b));
        assert_eq!(range_b.count_tiles(), 7);
        assert_eq!(range_b.contains(&CCTile::origin()), false);

        let range_ab = range_a.intersect(&range_b);
        assert_eq!(range_ab.contains(&tile_a), false);
        assert_eq!(range_ab.contains(&tile_b), false);
        assert_eq!(range_ab.contains(&CCTile::origin()), false);

        let mut num_elems = 0;
        for _elem in range_ab {
            num_elems += 1;
        }
        assert_eq!(num_elems, 1);
    }

    #[test]
    fn test_spiral_steps() {
        let ht: HGSTile = CCTile::unit(&RingCornerIndex::RIGHT).into();
        assert_eq!(ht.h.value(), 6);
        let ht_minus1 = ht.decrement_spiral();
        assert_eq!(ht_minus1.h, ht.h - 1);
        let ht2 = ht.spiral_steps(3);
        assert_eq!(ht2.h.value(), 9);
    }

    #[test]
    // used in readme, so should be correct pls
    fn readme_hgs1() {
        let tile = CCTile::from_qrs(2, -1, -1);
        assert_eq!(tile.spiral_index(), TileIndex(7));
        let tile2 = tile.spiral_steps(2);
        assert_eq!(tile2.spiral_index(), TileIndex(9));
        assert_eq!(tile2, CCTile::from_qrs(1, -2, 1));
    }

    #[test]
    fn test_reflect() {
        let tile = CCTile::from_qrs(4, -3, -1);
        let tile_r = tile.reflect_along_constant_axis(false, true, false);
        assert_eq!(tile_r, (-1, -3, 4).into());
        let tile_q = tile.reflect_along_constant_axis(true, false, false);
        assert_eq!(tile_q, (4, -1, -3).into());
        let tile_s = tile.reflect_along_constant_axis(false, false, true);
        assert_eq!(tile_s, (-3, 4, -1).into());

        let tile_reflected = tile.reflect_diagonally();
        assert_eq!(tile_reflected, (-4, 3, 1).into());

        let tile_ortho_r = tile.reflect_orthogonally_across_constant_axis(false, true, false);
        assert_eq!(tile_ortho_r, (1, 3, -4).into());
        let tile_ortho_s = tile.reflect_orthogonally_across_constant_axis(false, false, true);
        assert_eq!(tile_ortho_s, (3, -4, 1).into());
        let tile_ortho_q = tile.reflect_orthogonally_across_constant_axis(true, false, false);
        assert_eq!(tile_ortho_q, (-4, 1, 3).into());
    }

    #[test]
    fn test_grid_distance_to() {
        let tile_left = CCTile::unit(&RingCornerIndex::LEFT);
        let tile_right = CCTile::unit(&RingCornerIndex::RIGHT);
        assert_eq!(tile_left.grid_distance_to(&tile_right), 2);
        let tile1 = CCTile::from_qrs(1, 2, -3);
        assert_eq!(tile1.grid_distance_to(&tile_right), 2);
        assert_eq!(tile_right.grid_distance_to(&tile1), 2);
        assert_eq!(tile1.grid_distance_to(&tile1), 0);
        assert_eq!(tile1.grid_distance_to(&tile_left), 4);
        let tile2 = CCTile::from_qrs(-3, 2, 1);
        assert_eq!(tile2.grid_distance_to(&tile1), 4);
    }
}
