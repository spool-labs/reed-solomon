// Shape-coverage gate for the snapshot outer swap: every (k, n-k) tape-internal
// can ask for must construct and round-trip. k = ceil(n/3) per
// snapshot_outer_k, n = active spool group count. n=1 gives m=0, which
// tape-internal short-circuits before reaching a coder at all.
use tape_reed_solomon::ReedSolomon16;

fn main() {
    // 64 bytes is tape-internal's minimum chunk (it rounds up to a 64-byte
    // multiple), so it is the tightest real shard size.
    let sz = 64usize;
    let mut fails = Vec::new();
    let mut checked = 0usize;

    for n in 2..=300usize {
        let k = n.div_ceil(3);
        let m = n - k;
        if m == 0 {
            continue;
        }
        let rs = match ReedSolomon16::new(k, m) {
            Ok(rs) => rs,
            Err(e) => {
                fails.push(format!("n={n} (k={k},m={m}): construct failed: {e:?}"));
                continue;
            }
        };
        // Distinct bytes per shard so a mixed-up row cannot pass by accident.
        let mut shards: Vec<Vec<u8>> = (0..k)
            .map(|i| (0..sz).map(|j| (i * 7 + j * 3) as u8).collect())
            .collect();
        shards.extend((0..m).map(|_| vec![0u8; sz]));
        let original = shards[..k].to_vec();
        if let Err(e) = rs.encode(&mut shards) {
            fails.push(format!("n={n} (k={k},m={m}): encode failed: {e:?}"));
            continue;
        }
        // Erase the first m data shards (worst case for a data-heavy recovery)
        // capped at k, and rebuild.
        let erase = m.min(k);
        let mut opt: Vec<Option<Vec<u8>>> = shards.iter().cloned().map(Some).collect();
        for o in opt.iter_mut().take(erase) {
            *o = None;
        }
        if let Err(e) = rs.reconstruct(&mut opt) {
            fails.push(format!("n={n} (k={k},m={m}): reconstruct failed: {e:?}"));
            continue;
        }
        for i in 0..erase {
            if opt[i].as_ref() != Some(&original[i]) {
                fails.push(format!("n={n} (k={k},m={m}): shard {i} wrong after rebuild"));
                break;
            }
        }
        checked += 1;
    }

    println!("checked {checked} snapshot shapes (n=2..300, k=ceil(n/3)) at {sz}B shards");
    if fails.is_empty() {
        println!("ALL PASS");
    } else {
        println!("FAILURES ({}):", fails.len());
        for f in fails.iter().take(20) {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}
