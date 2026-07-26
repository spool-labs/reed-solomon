fn main() {
    for &(k, m) in &[(7usize,13usize),(10,10),(14,14),(16,16),(17,33),(18,6),(32,32)] {
        let rs = tape_reed_solomon::ReedSolomon::new(k, m).unwrap();
        println!("({k},{m}) 16KiB: {}   4MiB: {}", rs.encode_route(16*1024), rs.encode_route(4*1024*1024));
    }
}
