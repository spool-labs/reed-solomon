//! GF(2^8) matrices: Vandermonde construction, multiplication, and inversion.

use crate::errors::Error;
use crate::galois;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Matrix {
    row_count: usize,
    col_count: usize,
    // flattened row-major storage
    data: Vec<u8>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            row_count: rows,
            col_count: cols,
            data: vec![0u8; rows * cols],
        }
    }

    pub fn identity(size: usize) -> Matrix {
        let mut result = Self::new(size, size);
        for i in 0..size {
            result.set(i, i, 1);
        }
        result
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> u8 {
        self.data[r * self.col_count + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: u8) {
        self.data[r * self.col_count + c] = val;
    }

    pub fn get_row(&self, row: usize) -> &[u8] {
        let start = row * self.col_count;
        &self.data[start..start + self.col_count]
    }

    pub fn multiply(&self, rhs: &Matrix) -> Matrix {
        if self.col_count != rhs.row_count {
            panic!(
                "Column count on left is different from row count on right, lhs: {}, rhs: {}",
                self.col_count, rhs.row_count
            )
        }
        let mut result = Self::new(self.row_count, rhs.col_count);
        for r in 0..self.row_count {
            for c in 0..rhs.col_count {
                let mut val = 0u8;
                for i in 0..self.col_count {
                    val = galois::add(val, galois::mul(self.get(r, i), rhs.get(i, c)));
                }
                result.set(r, c, val);
            }
        }
        result
    }

    pub fn augment(&self, rhs: &Matrix) -> Matrix {
        if self.row_count != rhs.row_count {
            panic!(
                "Matrices do not have the same row count, lhs: {}, rhs: {}",
                self.row_count, rhs.row_count
            )
        }
        let mut result = Self::new(self.row_count, self.col_count + rhs.col_count);
        for r in 0..self.row_count {
            for c in 0..self.col_count {
                result.set(r, c, self.get(r, c));
            }
            for c in 0..rhs.col_count {
                result.set(r, self.col_count + c, rhs.get(r, c));
            }
        }
        result
    }

    pub fn sub_matrix(&self, rmin: usize, cmin: usize, rmax: usize, cmax: usize) -> Matrix {
        let mut result = Self::new(rmax - rmin, cmax - cmin);
        for r in rmin..rmax {
            for c in cmin..cmax {
                result.set(r - rmin, c - cmin, self.get(r, c));
            }
        }
        result
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        let r1_s = r1 * self.col_count;
        let r2_s = r2 * self.col_count;
        for i in 0..self.col_count {
            self.data.swap(r1_s + i, r2_s + i);
        }
    }

    pub fn is_square(&self) -> bool {
        self.row_count == self.col_count
    }

    pub fn gaussian_elim(&mut self) -> Result<(), Error> {
        for r in 0..self.row_count {
            if self.get(r, r) == 0 {
                for r_below in r + 1..self.row_count {
                    if self.get(r_below, r) != 0 {
                        self.swap_rows(r, r_below);
                        break;
                    }
                }
            }
            // No pivot found: the matrix is singular.
            if self.get(r, r) == 0 {
                return Err(Error::IncorrectShardSize);
            }
            if self.get(r, r) != 1 {
                let scale = galois::div(1, self.get(r, r));
                for c in 0..self.col_count {
                    self.set(r, c, galois::mul(scale, self.get(r, c)));
                }
            }
            // Add and subtract are both xor in GF(2^8).
            for r_below in r + 1..self.row_count {
                if self.get(r_below, r) != 0 {
                    let scale = self.get(r_below, r);
                    for c in 0..self.col_count {
                        let v = galois::add(self.get(r_below, c), galois::mul(scale, self.get(r, c)));
                        self.set(r_below, c, v);
                    }
                }
            }
        }

        // Clear the entries above the diagonal.
        for d in 0..self.row_count {
            for r_above in 0..d {
                if self.get(r_above, d) != 0 {
                    let scale = self.get(r_above, d);
                    for c in 0..self.col_count {
                        let v = galois::add(self.get(r_above, c), galois::mul(scale, self.get(d, c)));
                        self.set(r_above, c, v);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn invert(&self) -> Result<Matrix, Error> {
        if !self.is_square() {
            panic!("Trying to invert a non-square matrix")
        }
        let row_count = self.row_count;
        let col_count = self.col_count;

        let mut work = self.augment(&Self::identity(row_count));
        work.gaussian_elim()?;

        Ok(work.sub_matrix(0, row_count, col_count, col_count * 2))
    }

    pub fn vandermonde(rows: usize, cols: usize) -> Matrix {
        let mut result = Self::new(rows, cols);
        for r in 0..rows {
            // Distinct field element per row keeps the matrix invertible.
            let r_a = r as u8;
            for c in 0..cols {
                result.set(r, c, galois::exp(r_a, c));
            }
        }
        result
    }
}
