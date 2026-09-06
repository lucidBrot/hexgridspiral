# Changelog
## v0.3.3

* Work with current nightly `std 1.100.0-nightly`
* Upgrade `serde` from `1.0.219`  to `1.0.229`
* Upgrade `rand` from `0.10.1`  to `0.10.2`
* Move `rand_chacha` to dev dependencies

## v0.3.2

* Major-Version bump of `rand` and `rand_chacha` to `0.10.0` 
* Increase rust edition of this crate to `2024`

## v0.3.1

* Only `cargo update` 

## v0.3.0
* Fix [issue-1](https://github.com/lucidBrot/hexgridspiral/issues/2): Conversions from `CCTile` to `HGSTile` were incorrect for non-corner tiles in the top-right region.

## v0.2.8
* Add `serde` dependency and implement `Serialize` and `Deserialize` on most types.

## v0.2.7

Just documentation changes.

* Readme mentions the [Example Website](https://lucidbrot.github.io/hexgridspiral-example/) and the [Corresponding Repository](https://github.com/lucidBrot/hexgridspiral-example)
* Clarify that the redblobgames blogspot is a living document that was expanded to mention spiral systems _after_ I implemented it.

## v0.2.6

* Expose `HGSTile::spiral_index(&self)` to get the tile index.
* Expose `HGSTile::ring(&self)` to get the ring.
* Expose `Ring::ring_index(&self)` to get the ring's index.

## v0.2.5
* Upgrade dependencies, including breaking changes:
    ```
       Upgrading derive_more ^1 -> ^2
       Upgrading rand ^0.8.5 -> ^0.9.0
       Upgrading rand_chacha ^0.3.1 -> ^0.9.0
    ```

## v0.2.4
* Derive `Hash` for `TileIndex`

## v0.2.1
* Replace relative links to images with absolute urls, so they should now render in crates.io and docs.rs.

## v0.2.0
* `RingCornerIndex::all_from()`  now takes ownership of its argument. This is more flexible because the caller can choose whether they want to clone or not.
