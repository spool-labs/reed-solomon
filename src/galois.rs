//! GF(2^8) finite field arithmetic.

const FIELD_SIZE: usize = 256;
/// Reduction constant for the primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d).
const GENERATING_POLYNOMIAL: usize = 0x1d;
/// EXP table is doubled (minus 2) so `log_a + log_b` (max 254+254) never overflows.
const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;

const fn gen_log_table() -> [u8; FIELD_SIZE] {
    let mut result = [0u8; FIELD_SIZE];
    let mut b: usize = 1;
    let mut log: usize = 0;
    while log < FIELD_SIZE - 1 {
        result[b] = log as u8;
        b <<= 1;
        if b >= FIELD_SIZE {
            b = (b - FIELD_SIZE) ^ GENERATING_POLYNOMIAL;
        }
        log += 1;
    }
    result
}

const fn gen_exp_table() -> [u8; EXP_TABLE_SIZE] {
    let log_table = gen_log_table();
    let mut result = [0u8; EXP_TABLE_SIZE];
    let mut i = 1;
    while i < FIELD_SIZE {
        let log = log_table[i] as usize;
        result[log] = i as u8;
        result[log + FIELD_SIZE - 1] = i as u8;
        i += 1;
    }
    result
}

pub(crate) const fn gen_mul_table() -> [[u8; FIELD_SIZE]; FIELD_SIZE] {
    let log_table = gen_log_table();
    let exp_table = gen_exp_table();
    let mut result = [[0u8; FIELD_SIZE]; FIELD_SIZE];
    let mut a = 0usize;
    while a < FIELD_SIZE {
        let mut b = 0usize;
        while b < FIELD_SIZE {
            result[a][b] = if a == 0 || b == 0 {
                0
            } else {
                let log_a = log_table[a] as usize;
                let log_b = log_table[b] as usize;
                exp_table[log_a + log_b]
            };
            b += 1;
        }
        a += 1;
    }
    result
}

/// log_2 table (base = generator 2). `LOG_TABLE[0]` is unused (set to 0).
pub static LOG_TABLE: [u8; FIELD_SIZE] = gen_log_table();
/// Antilog table, doubled to avoid a modulo in `mul`.
pub static EXP_TABLE: [u8; EXP_TABLE_SIZE] = gen_exp_table();
/// Full 256x256 multiplication table.
pub static MUL_TABLE: [[u8; FIELD_SIZE]; FIELD_SIZE] = gen_mul_table();

/// Add (== subtract) two field elements.
#[inline]
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Multiply two field elements.
#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    MUL_TABLE[a as usize][b as usize]
}

/// Divide `a` by `b`. Panics if `b == 0`.
#[inline]
pub fn div(a: u8, b: u8) -> u8 {
    if a == 0 {
        0
    } else if b == 0 {
        panic!("Divisor is 0")
    } else {
        let log_a = LOG_TABLE[a as usize];
        let log_b = LOG_TABLE[b as usize];
        let mut log_result = log_a as isize - log_b as isize;
        if log_result < 0 {
            log_result += 255;
        }
        EXP_TABLE[log_result as usize]
    }
}

/// Compute `a^n`.
#[inline]
pub fn exp(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        let log_a = LOG_TABLE[a as usize];
        let mut log_result = log_a as usize * n;
        while 255 <= log_result {
            log_result -= 255;
        }
        EXP_TABLE[log_result]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference log table for the canonical GF(2^8) field; the generated
    // LOG_TABLE must match it byte-for-byte.
    static BACKBLAZE_LOG_TABLE: [u8; 256] = [
        0, 0, 1, 25, 2, 50, 26, 198, 3, 223, 51, 238, 27, 104, 199, 75, 4, 100, 224, 14, 52, 141,
        239, 129, 28, 193, 105, 248, 200, 8, 76, 113, 5, 138, 101, 47, 225, 36, 15, 33, 53, 147,
        142, 218, 240, 18, 130, 69, 29, 181, 194, 125, 106, 39, 249, 185, 201, 154, 9, 120, 77,
        228, 114, 166, 6, 191, 139, 98, 102, 221, 48, 253, 226, 152, 37, 179, 16, 145, 34, 136, 54,
        208, 148, 206, 143, 150, 219, 189, 241, 210, 19, 92, 131, 56, 70, 64, 30, 66, 182, 163,
        195, 72, 126, 110, 107, 58, 40, 84, 250, 133, 186, 61, 202, 94, 155, 159, 10, 21, 121, 43,
        78, 212, 229, 172, 115, 243, 167, 87, 7, 112, 192, 247, 140, 128, 99, 13, 103, 74, 222,
        237, 49, 197, 254, 24, 227, 165, 153, 119, 38, 184, 180, 124, 17, 68, 146, 217, 35, 32,
        137, 46, 55, 63, 209, 91, 149, 188, 207, 205, 144, 135, 151, 178, 220, 252, 190, 97, 242,
        86, 211, 171, 20, 42, 93, 158, 132, 60, 57, 83, 71, 109, 65, 162, 31, 45, 67, 216, 183,
        123, 164, 118, 196, 23, 73, 236, 127, 12, 111, 246, 108, 161, 59, 82, 41, 157, 85, 170,
        251, 96, 134, 177, 187, 204, 62, 90, 203, 89, 95, 176, 156, 169, 160, 81, 11, 245, 22, 235,
        122, 117, 44, 215, 79, 174, 213, 233, 230, 231, 173, 232, 116, 214, 244, 234, 168, 80, 88,
        175,
    ];

    #[test]
    fn log_table_same_as_backblaze() {
        for i in 0..256 {
            assert_eq!(LOG_TABLE[i], BACKBLAZE_LOG_TABLE[i]);
        }
    }

    #[test]
    fn known_products() {
        assert_eq!(mul(3, 4), 12);
        assert_eq!(mul(7, 7), 21);
        assert_eq!(mul(23, 45), 41);
        assert_eq!(exp(2, 2), 4);
        assert_eq!(exp(5, 20), 235);
        assert_eq!(exp(13, 7), 43);
    }

    #[test]
    fn field_axioms() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(mul(a, b), mul(b, a));
            }
            if a != 0 {
                assert_eq!(mul(a, div(1, a)), 1);
            }
        }
    }
}
