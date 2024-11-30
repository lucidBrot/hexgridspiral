# HexGridSpiral

* A Coordinate System for hexagonal maps
* Each tile is identified by a  **single unique integer**
* Tile indices start at 0 and **spiral outwards**
* Ideal for e.g. a Level Selection Screen because the level number maps directly to a hex tile without scattering them all over the place.

* My main contribution is **efficiently finding the ring-index** in HexGridSpiral-Space (HGS).

However, most other use-cases benefit greatly from the Cube Coordinates outlined in [this redblobgames article](https://www.redblobgames.com/grids/hexagons/#distances-cube), this repo also implements

* Conversion to **Cube Coordinates** for efficient **Neighbour** finding
* A **Distance** between tiles in grid steps
* **Vector Calculus**  
  You can subtract two tiles to get a vector and apply it to a different tile.  You can scale vectors by multiplication.
* A **Euclidean Norm** ([Xiangguo Li 2013](https://www.researchgate.net/publication/235779843_Storage_and_addressing_scheme_for_practical_hexagonal_image_processing))
* Conversion to **Cartesian Pixel Coordinates**
* Rotation by 60 degrees around the origin
* **Addition**, Subtraction of tiles
* Which **Wedge** of the world (pizza slice) a tile is in

## Usage

Ideally have a look at the code - many features are only implemented on either `HGSTile` or `CCTile`, between which you can convert with `.into()`. `HGSTile` is related to spirals and rings, `CCTile`  has a solid coordinate system to perform arithmetic in.

Here are some ways to **construct** tiles:  

```rust
let tile1 = CCTile::unit(&RingCornerIndex::RIGHT);
let tile2 = CCTile::unit(&RingCornerIndex::TOPRIGHT);
assert_eq!(tile1.euclidean_distance_to(&tile1), 0.);

let tile7: CCTile = HGSTile::make(7).into();
let tileo : CCTile = CCTile::origin();
assert_eq!(tile7.euclidean_distance_sq(&tileo), 3);

let corner0_hgs = HGSTile::new(ring.corner(&RingCornerIndex::BOTTOMLEFT));
```