//! The open mask vocabulary: combinations, naming, validation, and the brute-
//! force cross-check of the mixed-radix group arithmetic.

use rustrain_parallel::{GroupMask, Mesh, MeshFingerprint, ParallelConfig, ParallelError};

fn cfg(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> ParallelConfig {
    ParallelConfig {
        tensor: tp,
        context: cp,
        expert: ep,
        data: dp,
        pipeline: pp,
    }
}

fn mesh(c: ParallelConfig) -> Mesh {
    Mesh::from_config(&c)
}

/// Coordinates of `rank` on every mesh axis.
fn coords(mesh: &Mesh, rank: usize) -> Vec<usize> {
    (0..mesh.axis_count())
        .map(|axis| {
            let stride = mesh.stride(axis).unwrap();
            let degree = mesh.degree(axis).unwrap();
            (rank / stride) % degree
        })
        .collect()
}

/// Brute-force reference for the group of `rank` under `mask`: members are the
/// ranks sharing every unmasked coordinate; the index is the mixed-radix
/// number over the masked coordinates; the id is the mixed-radix number over
/// the unmasked ones. Computed without any of the mesh's group methods.
fn brute_group(mesh: &Mesh, mask: GroupMask, rank: usize) -> (Vec<usize>, usize, usize) {
    let c = coords(mesh, rank);
    let mut members = Vec::new();
    for r in 0..mesh.world_size() {
        let rc = coords(mesh, r);
        if (0..mesh.axis_count()).all(|axis| mask.contains(axis) || rc[axis] == c[axis]) {
            members.push(r);
        }
    }
    let mut index = 0usize;
    let mut weight = 1usize;
    for (axis, &coord) in c.iter().enumerate() {
        if mask.contains(axis) {
            index += coord * weight;
            weight *= mesh.degree(axis).unwrap();
        }
    }
    let mut id = 0usize;
    let mut weight = 1usize;
    for (axis, &coord) in c.iter().enumerate() {
        if !mask.contains(axis) {
            id += coord * weight;
            weight *= mesh.degree(axis).unwrap();
        }
    }
    (members, index, id)
}

/// Test 1 of D3 step 1: `GroupMask` expresses `tp|ep`, `tp|dp`, `ep|dp` and
/// all five axes at once; `Mesh::group_name` renders them; a bit outside the
/// mesh errors; 33 axes error; duplicate/empty axis names error.
#[test]
fn group_combinations_and_names() {
    let m = mesh(cfg(4, 3, 2, 5, 2));
    assert_eq!(m.world_size(), 240);

    let tp = GroupMask::single(0).unwrap();
    let cp = GroupMask::single(1).unwrap();
    let ep = GroupMask::single(2).unwrap();
    let dp = GroupMask::single(3).unwrap();
    let pp = GroupMask::single(4).unwrap();

    // Combination masks are first-class groups.
    let tp_ep = tp.union(ep);
    let tp_dp = tp.union(dp);
    let ep_dp = ep.union(dp);
    let all = tp.union(cp).union(ep).union(dp).union(pp);
    assert_eq!(tp_ep, GroupMask::from_bits(0b00101));
    assert_eq!(tp_dp, GroupMask::from_bits(0b01001));
    assert_eq!(ep_dp, GroupMask::from_bits(0b01100));
    assert_eq!(all, GroupMask::from_bits(0b11111));

    assert_eq!(tp_ep.degree(&m).unwrap(), 4 * 2);
    assert_eq!(tp_dp.degree(&m).unwrap(), 4 * 5);
    assert_eq!(ep_dp.degree(&m).unwrap(), 2 * 5);
    assert_eq!(all.degree(&m).unwrap(), 240);
    assert_eq!(GroupMask::NONE.degree(&m).unwrap(), 1);

    // Names come from the mesh, joined in axis order.
    assert_eq!(m.group_name(tp).unwrap(), "tp");
    assert_eq!(m.group_name(tp_ep).unwrap(), "tp|ep");
    assert_eq!(m.group_name(tp_dp).unwrap(), "tp|dp");
    assert_eq!(m.group_name(ep_dp).unwrap(), "ep|dp");
    assert_eq!(m.group_name(all).unwrap(), "global");
    assert_eq!(m.group_name(GroupMask::NONE).unwrap(), "none");

    // The bit form is what `Display` prints; names need a mesh.
    assert_eq!(tp_ep.to_string(), "mask(0b101)");
    assert_eq!(GroupMask::NONE.to_string(), "mask(0b0)");
    assert_eq!(GroupMask::from_bits(21).to_string(), "mask(0b10101)");

    // A bit outside the mesh is the error case.
    let stray = GroupMask::from_bits(1 << 5);
    assert_eq!(
        m.group_name(stray),
        Err(ParallelError::GroupOutOfRange { bit: 5, axes: 5 })
    );
    assert_eq!(
        stray.validate(&m),
        Err(ParallelError::GroupOutOfRange { bit: 5, axes: 5 })
    );
    // A combined mask with one stray bit errors too, naming the bit.
    let mostly_ok = tp_ep.union(GroupMask::from_bits(1 << 31));
    assert_eq!(
        mostly_ok.validate(&m),
        Err(ParallelError::GroupOutOfRange { bit: 31, axes: 5 })
    );

    // Set algebra.
    assert_eq!(tp_ep.intersect(ep_dp), ep);
    assert_eq!(tp_ep.without(ep), tp);
    assert_eq!(tp_ep.union(ep_dp), GroupMask::from_bits(0b01101));
    assert_eq!(all.axis_count(), 5);
    assert_eq!(GroupMask::NONE.axis_count(), 0);
    assert!(tp.contains(0));
    assert!(!tp.contains(1));
    assert!(!tp.contains(31)); // out of the mask's bit range: not contained
    assert!(GroupMask::NONE.is_empty());
    assert!(!tp.is_empty());
    assert_eq!(tp.bits(), 0b1);
}

/// The combined-mask group arithmetic agrees with a brute-force computation
/// over coordinates: members, index and id, for a spread of combinations.
#[test]
fn combined_mask_groups_match_brute_force() {
    let m = mesh(cfg(4, 3, 2, 5, 2));
    let masks = [
        GroupMask::NONE,
        GroupMask::from_bits(0b00001),
        GroupMask::from_bits(0b00101), // tp|ep
        GroupMask::from_bits(0b01001), // tp|dp
        GroupMask::from_bits(0b01100), // ep|dp
        GroupMask::from_bits(0b10101), // tp|ep|pp
        GroupMask::from_bits(0b11111), // all
    ];
    for mask in masks {
        let degree = mask.degree(&m).unwrap();
        let mut seen: Vec<usize> = Vec::new();
        for rank in 0..m.world_size() {
            let (members, index, id) = brute_group(&m, mask, rank);
            assert_eq!(members.len(), degree);
            assert!(members.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(m.group_ranks(mask, rank).unwrap(), members);
            assert_eq!(m.group_index(mask, rank).unwrap(), index);
            assert_eq!(m.group_id(mask, rank).unwrap(), id);
            assert_eq!(members[index], rank);
            if !seen.contains(&id) {
                seen.push(id);
            }
        }
        // Walking ranks ascending visits group ids ascending: groups are
        // ordered by first member.
        assert_eq!(
            seen,
            (0..m.world_size() / degree).collect::<Vec<usize>>(),
            "mask {mask}"
        );
        // The group ids really partition the world.
        assert_eq!(seen.len(), m.world_size() / degree);
    }
}

#[test]
fn mesh_rejects_bad_axis_lists() {
    let axes =
        |n: usize| -> Vec<(String, usize)> { (0..n).map(|i| (format!("a{i}"), 2)).collect() };
    // 33 axes: one past MAX_AXES.
    assert_eq!(
        Mesh::new(axes(33)),
        Err(ParallelError::TooManyAxes { count: 33, max: 32 })
    );
    // 32 axes are fine.
    assert!(Mesh::new(axes(32)).is_ok());
    // No axes at all.
    assert_eq!(
        Mesh::new(Vec::new()),
        Err(ParallelError::NoAxes { max: 32 })
    );
    // Empty name.
    assert_eq!(
        Mesh::new(vec![("a".to_string(), 2), (String::new(), 3)]),
        Err(ParallelError::EmptyAxisName { index: 1 })
    );
    // Duplicate name.
    assert_eq!(
        Mesh::new(vec![
            ("a".to_string(), 2),
            ("b".to_string(), 3),
            ("a".to_string(), 4),
        ]),
        Err(ParallelError::DuplicateAxis {
            name: "a".to_string()
        })
    );
    // Zero degree.
    assert_eq!(
        Mesh::new(vec![("a".to_string(), 2), ("b".to_string(), 0)]),
        Err(ParallelError::ZeroDegree {
            name: "b".to_string()
        })
    );
}

#[test]
fn single_rejects_axes_past_the_mask_bits() {
    assert!(GroupMask::single(31).is_ok());
    assert_eq!(
        GroupMask::single(32),
        Err(ParallelError::AxisOutOfRange { axis: 32, max: 32 })
    );
    assert_eq!(
        GroupMask::single(usize::MAX),
        Err(ParallelError::AxisOutOfRange {
            axis: usize::MAX,
            max: 32
        })
    );
}

/// A mesh is open, not five closed axes: arbitrary names and orders work, and
/// the stride convention is `axes[0]` fastest.
#[test]
fn an_open_mesh_with_custom_axes() {
    let m = Mesh::new(vec![
        ("seq".to_string(), 2),
        ("model".to_string(), 3),
        ("stage".to_string(), 5),
    ])
    .unwrap();
    assert_eq!(m.world_size(), 30);
    assert_eq!(m.stride(0), Some(1));
    assert_eq!(m.stride(1), Some(2));
    assert_eq!(m.stride(2), Some(6));
    assert_eq!(m.index_of("stage"), Some(2));
    // rank 17 = 1*1 + 2*2 + 2*6 -> seq=1, model=2, stage=2.
    let seq = GroupMask::single(0).unwrap();
    let model = GroupMask::single(1).unwrap();
    let stage = GroupMask::single(2).unwrap();
    assert_eq!(m.group_ranks(seq, 17).unwrap(), vec![16, 17]);
    assert_eq!(m.group_ranks(model, 17).unwrap(), vec![13, 15, 17]);
    assert_eq!(m.group_ranks(stage, 17).unwrap(), vec![5, 11, 17, 23, 29]);
    assert_eq!(m.group_name(seq.union(stage)).unwrap(), "seq|stage");
}

/// The fingerprint is the plan's copy of the mesh: the ordered axes and
/// nothing else (invariant I-6), and it serializes deterministically.
#[test]
fn fingerprint_carries_axes_only() {
    let m = mesh(cfg(4, 3, 2, 5, 2));
    let fp: MeshFingerprint = m.fingerprint();
    assert_eq!(fp.axes(), m.axes());
    assert_eq!(fp.world_size(), m.world_size());
    assert_eq!(
        fp.group_name(GroupMask::from_bits(0b00101)).unwrap(),
        "tp|ep"
    );
    assert_eq!(
        fp.group_name(GroupMask::from_bits(1 << 5)),
        Err(ParallelError::GroupOutOfRange { bit: 5, axes: 5 })
    );
    // Deterministic serialization: an ordered list, never a map.
    let json = serde_json::to_string(&fp).unwrap();
    assert_eq!(
        json,
        r#"{"axes":[["tp",4],["cp",3],["ep",2],["dp",5],["pp",2]]}"#
    );
    let back: MeshFingerprint = serde_json::from_str(&json).unwrap();
    assert_eq!(back, fp);
}

/// `GroupMask` serializes as its `u32` bits — a plain number in JSON.
#[test]
fn group_mask_serializes_as_bits() {
    let mask = GroupMask::from_bits(0b10101);
    assert_eq!(serde_json::to_string(&mask).unwrap(), "21");
    let back: GroupMask = serde_json::from_str("21").unwrap();
    assert_eq!(back, mask);
}

/// A mesh whose degrees do not multiply into a `usize` is rejected rather than silently indexed
/// with a saturated world size: every rank number, stride and group id comes from that product.
#[test]
fn a_mesh_whose_world_size_overflows_is_rejected() {
    let err = Mesh::new(vec![
        ("tp".to_string(), 1 << 40),
        ("cp".to_string(), 1 << 40),
    ])
    .unwrap_err();
    assert_eq!(
        err,
        ParallelError::MeshWorldSizeOverflow {
            degrees: vec![1 << 40, 1 << 40],
        }
    );
    // The largest product that fits is still accepted.
    let ok = Mesh::new(vec![("tp".to_string(), 1 << 31), ("cp".to_string(), 2)]).unwrap();
    assert_eq!(ok.world_size(), 1usize << 32);
}

/// A fingerprint travels as data, so it is re-validated on the way back to a mesh: that is where a
/// plan compiled for a different (or corrupted) topology is caught.
#[test]
fn a_fingerprint_rebuilds_its_mesh_and_is_revalidated() {
    let m = mesh(cfg(2, 2, 4, 2, 2));
    let back = m.fingerprint().to_mesh().unwrap();
    assert_eq!(back, m);
    assert_eq!(
        back.group_name(GroupMask::from_bits(0b11)).unwrap(),
        "tp|cp"
    );

    // Invalid axes cannot sneak in through the serde form either.
    let bad: MeshFingerprint = serde_json::from_str(r#"{"axes":[["tp",0]]}"#).unwrap();
    assert_eq!(
        bad.to_mesh(),
        Err(ParallelError::ZeroDegree {
            name: "tp".to_string()
        })
    );
    let empty: MeshFingerprint = serde_json::from_str(r#"{"axes":[]}"#).unwrap();
    assert_eq!(empty.to_mesh(), Err(ParallelError::NoAxes { max: 32 }));
}
