// In this file, I propose a grid system I came up with independently.
// The hexagonal grid assigns each tile a unique integer index (h) that increments, spiralling outwards.
// This means that incrementing equals walking along the spiral.
//
// The spiral can be split into conceptual rings. Each ring consists of tiles that are consecutively numbered.
// All tiles that lie on the horizontal ray going from the origin tile to the right are assigned an index that is the largest in that ring. To the top-left of that tile is the smallest index in that ring. This is assuming tiling with peaky-top regular hexagons.
//
// Each ring has a unique integer ring-index (n) that is equal to that ring's side-length in hexes (including both corners).
// Given the ring-index n, we can compute the max and min tile-index inside the ring.
#![feature(iter_chain)]
#![feature(step_trait)]

// TODO: public re-exports?

// The objects Tile, Ring, TileIndex, RingIndex are not supposed to be mutated.
// Instead, (they) make new objects.
use derive_more::{Add, Display, From, Into, Mul, Neg, Sub};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::Rng;
use std::ops;

#[derive(
    Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Add, Sub, Mul, Display, From, Into,
)]
pub struct TileIndex(pub u64);

impl std::iter::Step for TileIndex {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        return if end < start {
            None
        } else {
            usize::try_from((*end - *start).value() as usize).ok()
        };
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

#[derive(Debug, Copy, Clone, PartialEq, Add, Display, From, Into)]
pub struct Ring {
    /// ring-index
    n: RingIndex,
}

/// Hexgrid Spiral Tile
#[derive(Debug, Copy, Clone, PartialEq, Display, From, Into)]
#[display("HGSTile: {h}")]
pub struct HGSTile {
    // tile-index
    h: TileIndex,
    ring: Ring,
}

/// CubeCoordinates Tile
/// For reference: https://www.redblobgames.com/grids/hexagons/
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
    pub fn from_corners<'a>(mut a: &'a RingCornerIndex, mut b: &'a RingCornerIndex) -> Self {
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
        Self((p % 6))
    }
}

impl RingCornerIndex {
    pub fn next(&self) -> Self {
        let v = *self as u8;
        let v2 = match v {
            0..=4 => v + 1,
            5 => 0,
            6_u8..=u8::MAX => panic!("Invalid corner index used as start."),
        };
        TryInto::<RingCornerIndex>::try_into(v2).unwrap()
    }

    pub fn all() -> impl std::iter::Iterator<Item = RingCornerIndex> {
        Self::all_from(&Self::RIGHT)
    }

    // See https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html
    // the "Precise captures use" section.
    pub fn all_from(
        start: &RingCornerIndex,
    ) -> impl std::iter::Iterator<Item = RingCornerIndex> + use<'_> {
        let mut ctr = start.clone();
        let mut done = false;
        std::iter::from_fn(move || {
            ctr = ctr.next();
            if done {
                return None;
            }
            if ctr == *start {
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

    /// Not well-defined for the origin-tile
    pub fn ring_edge(&self) -> RingEdge {
        self.ring.edge(self.h)
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
        let offset = (h - b).value();
        let ring_size = ring.size();
        assert!(offset >= 0);
        assert!(
            offset < ring_size,
            "offset={offset:?}, ringsize={ring_size:?}, h={h:?}, ring={ring:?}"
        );
        return RingEdge::from_primitive((offset / side_size).try_into().unwrap());
    }

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

impl CCTile {
    pub fn origin() -> CCTile {
        CCTile::from_qrs(0, 0, 0)
    }
    pub fn from_qrs(q: i64, r: i64, s: i64) -> CCTile {
        assert_eq!(q + r + s, 0);
        CCTile { q, r, s }
    }
    pub fn from_qr(q: i64, r: i64) -> CCTile {
        CCTile::from_qrs(q, r, 0 - q - r)
    }
    pub fn from_qrs_tuple(t: (i64, i64, i64)) -> CCTile {
        CCTile::from_qrs(t.0, t.1, t.2)
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

    /// Adapted Euclidean Distance by
    /// Xiangguo Li
    ///
    /// https://www.researchgate.net/publication/235779843_Storage_and_addressing_scheme_for_practical_hexagonal_image_processing
    /// DOII https://doi.org/10.1117/1.JEI.22.1.010502
    /// TODO: Test that norm_li, norm_redblob, and norm_redblob_max are all equivalent?
    pub fn norm_li(&self) -> i64 {
        self.q * self.q + self.r * self.r + self.q * self.r
    }

    /// The distance between this tile and the origin, in discrete steps.
    ///
    /// The blogpost https://www.redblobgames.com/grids/hexagons/#distances-cube
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
    // TODO: Test this.
    pub fn norm_steps(&self) -> u64 {
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

    pub fn is_origin_tile(&self) -> bool {
        self.q == 0 && self.r == 0 && self.s == 0
    }

    /// Which wedge of the grid the current tile is in.
    /// The wedges are defined as in https://www.redblobgames.com/grids/hexagons/directions.html in the first diagram.
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

        let rci_qr = if (self.q - self.r > 0) {
            RingCornerIndex::TOPRIGHT
        } else {
            RingCornerIndex::BOTTOMLEFT
        };
        let rci_rs = if (self.r - self.s > 0) {
            RingCornerIndex::BOTTOMRIGHT
        } else {
            RingCornerIndex::TOPLEFT
        };
        let rci_sq = if (self.s - self.q > 0) {
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

    // The previous ring-corner in ccw direction (or the next corner in clockwise direction)
    // Might be this tile itself.
    // TODO: Test
    pub fn previous_corner_hgs(&self) -> HGSTile {
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
        assert!(!self.is_origin_tile());
        // get corner index
        let w = self.wedge_around_ringcorner()[0];
        // get ring radius
        let r = Ring::from(*self);
        let direction = CCTile::unit(&w);
        direction * (r.n.value() as i64)
    }

    /// Rotate 60 degrees counter-clockwise
    /// To rotate by other amounts, consider using spiral steps in HGSTile notation.
    /// https://www.redblobgames.com/grids/hexagons/#rotation
    pub fn rot60ccw(&self) -> CCTile {
        CCTile::from_qrs(-self.s, -self.q, -self.r)
    }
    /// Rotate 60 degrees clockwise
    /// https://www.redblobgames.com/grids/hexagons/#rotation
    pub fn rot60cw(&self) -> CCTile {
        CCTile::from_qrs(-self.r, -self.s, -self.q)
    }

    // TODO: Impement CC Reflection on all axes
    // https://www.redblobgames.com/grids/hexagons/#reflection

    // TODO: Implement CC to Cartesian2D
    // https://www.redblobgames.com/grids/hexagons/#hex-to-pixel

    pub fn spiral_steps(&self, steps: i64) -> Self {
        let ht: HGSTile = (*self).into();
        let ht2 = ht.spiral_steps(steps);
        ht2.into()
    }
}

// Conversion from HexGridSpiral to Cube Coordinates:
// We can easily get the ring_max. From there we jump to the next corner at most five times.
// Then add the rest.
// TODO: Test this.
impl From<HGSTile> for CCTile {
    fn from(item: HGSTile) -> Self {
        if item.is_origin_tile() {
            return CCTile::origin();
        }
        let max_hgs = item.ring_max();
        let edge_section = max_hgs.ring_edge();
        let corner_index = edge_section.start();
        // Find the previous corner in Cube Coordinates
        // This part is straightforward because it's simply moving along an axis
        let cc_axis_unit = CCTile::unit(&corner_index);
        let cc_edge_start = cc_axis_unit * (item.ring.n.value() as i64);
        // Then make steps in the edge direction
        let cc_edge_unit = CCTile::unit(&edge_section.direction());
        let corner_h = item.ring.corner(corner_index);
        assert!(corner_h <= item.h);
        let steps_from_corner = item.h - corner_h;
        return cc_edge_start + cc_edge_unit * (steps_from_corner.value() as i64);
    }
}

impl From<CCTile> for Ring {
    fn from(item: CCTile) -> Self {
        Ring::new(RingIndex(item.norm_steps() + 1))
    }
}

// Conversion from Cube Coordinates to HexGridSpiral:
// TODO: Thoroughly test this.
impl From<CCTile> for HGSTile {
    fn from(item: CCTile) -> Self {
        let ring = Ring::from(item);
        let ring_index = ring.n;
        if ring.n == RingIndex(1) {
            return HGSTile::new(TileIndex(0));
        }
        assert!(
            !item.is_origin_tile(),
            "This can not happen, the ring index of {item:?} is not 1. It is {ring_index:?}"
        );
        let wedges = item.wedge_around_ringcorner();
        // If the item tile is a ring-corner, there will only be one wedge. Otherwise, if it is the border of a wedge, there might be two.
        // TODO: does rust do runtime checks at release build runtime for these vecs? Just out of curiosity.
        let corner0_hgs = HGSTile::new(ring.corner(wedges[0]));
        // If there are two wedges, the tile lies on the diagonal axes that lie between the usual CC grid axes.
        // This can only happen on rings with odd full edgelengths, otherwise there is no tile on the wedge border.
        if wedges.len() == 2 {
            let corner1_hgs = HGSTile::new(ring.corner(wedges[1]));
            let smaller_corner_index = corner0_hgs.h.min(corner1_hgs.h);
            assert!(ring.full_edge_size() % 2 == 1);
            return HGSTile::new(smaller_corner_index + ring.edge_size() / 2);
        }
        assert!(wedges.len() == 1);
        // We have the corner in the middle of this wedge (corner0). How do we get the correct tile's hsg index?
        let offset_along_edge_hgs = item.grid_distance_to(&item.previous_corner_cc());
        return HGSTile::new(corner0_hgs.h + offset_along_edge_hgs);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::RingCornerIndex::BOTTOMLEFT;
    use rand::thread_rng;

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
        let H0: HGSTile = origin.into();
        let h0 = H0.h;
        assert_eq!(h0, TileIndex(0));

        let o_cc: CCTile = CCTile::from(HGSTile::new(h0));
        // TODO: implement this test for other tiles than origin.
    }

    // TODO: Debug these tests. add a test for the ring min and max.

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
        let mut rci = rci_vec[0];
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
        for r in RingCornerIndex::all_from(&RingCornerIndex::BOTTOMLEFT) {
            i += 1;
            let t = CCTile::unit(&r) * i;
            let q = CCTile::unit(&r.next());
            assert_eq!(t.rot60cw(), q);
            assert_eq!(q.rot60ccw(), t);
        }

        // Also test some not-corner tiles
        for h in (1..100).step_by(7) {
            let not_corner: CCTile = HGSTile::new(TileIndex(h)).into();
            let a = not_corner.rot60cw().rot60cw().rot60cw();
            let b = not_corner.rot60ccw().rot60ccw().rot60ccw();
            assert_eq!(a, b);
        }
    }
}