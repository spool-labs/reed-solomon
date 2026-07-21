//! Dense matrices over a binary extension field: Vandermonde construction,
//! multiplication, and inversion.
//!
//! Generic over [`FieldElement`], so one implementation serves both the GF(2^8)
//! coder and the GF(2^16) tower coder. The arithmetic is identical; only the
//! element type and its multiply/inverse change.

use crate::errors::Error;
use crate::field::FieldElement;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Matrix<Element: FieldElement> {
    row_count: usize,
    col_count: usize,
    // flattened row-major storage
    data: Vec<Element>,
}

impl<Element: FieldElement> Matrix<Element> {
    pub fn new(rows: usize, cols: usize) -> Matrix<Element> {
        Matrix {
            row_count: rows,
            col_count: cols,
            data: vec![Element::ZERO; rows * cols],
        }
    }

    pub fn identity(size: usize) -> Matrix<Element> {
        let mut result = Self::new(size, size);
        for i in 0..size {
            result.set(i, i, Element::ONE);
        }
        result
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Element {
        self.data[r * self.col_count + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: Element) {
        self.data[r * self.col_count + c] = val;
    }

    pub fn get_row(&self, row: usize) -> &[Element] {
        let start = row * self.col_count;
        &self.data[start..start + self.col_count]
    }

    pub fn multiply(&self, rhs: &Matrix<Element>) -> Matrix<Element> {
        if self.col_count != rhs.row_count {
            panic!(
                "Column count on left is different from row count on right, lhs: {}, rhs: {}",
                self.col_count, rhs.row_count
            )
        }
        // Accumulate row-at-a-time rather than cell-at-a-time: the left element
        // is then loop-invariant across the inner loop, and both the right row
        // and the output row are walked sequentially instead of by column
        // stride. A zero left element contributes nothing, so it skips the row.
        let mut result = Self::new(self.row_count, rhs.col_count);
        for r in 0..self.row_count {
            for i in 0..self.col_count {
                let scale = self.get(r, i);
                if scale == Element::ZERO {
                    continue;
                }
                let start = r * result.col_count;
                for c in 0..rhs.col_count {
                    let product = scale.gf_mul(rhs.data[i * rhs.col_count + c]);
                    result.data[start + c] = result.data[start + c].gf_add(product);
                }
            }
        }
        result
    }

    pub fn augment(&self, rhs: &Matrix<Element>) -> Matrix<Element> {
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

    pub fn sub_matrix(&self, rmin: usize, cmin: usize, rmax: usize, cmax: usize) -> Matrix<Element> {
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
            if self.get(r, r) == Element::ZERO {
                for r_below in r + 1..self.row_count {
                    if self.get(r_below, r) != Element::ZERO {
                        self.swap_rows(r, r_below);
                        break;
                    }
                }
            }
            // No pivot found: the matrix is singular.
            if self.get(r, r) == Element::ZERO {
                return Err(Error::SingularMatrix);
            }
            if self.get(r, r) != Element::ONE {
                let scale = self.get(r, r).gf_inv();
                for c in 0..self.col_count {
                    self.set(r, c, scale.gf_mul(self.get(r, c)));
                }
            }
            // Add and subtract are both xor in a characteristic-2 field.
            for r_below in r + 1..self.row_count {
                if self.get(r_below, r) != Element::ZERO {
                    let scale = self.get(r_below, r);
                    for c in 0..self.col_count {
                        let v = self.get(r_below, c).gf_add(scale.gf_mul(self.get(r, c)));
                        self.set(r_below, c, v);
                    }
                }
            }
        }

        // Clear the entries above the diagonal.
        for d in 0..self.row_count {
            for r_above in 0..d {
                if self.get(r_above, d) != Element::ZERO {
                    let scale = self.get(r_above, d);
                    for c in 0..self.col_count {
                        let v = self.get(r_above, c).gf_add(scale.gf_mul(self.get(d, c)));
                        self.set(r_above, c, v);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn invert(&self) -> Result<Matrix<Element>, Error> {
        if !self.is_square() {
            panic!("Trying to invert a non-square matrix")
        }
        let row_count = self.row_count;
        let col_count = self.col_count;

        let mut work = self.augment(&Self::identity(row_count));
        work.gaussian_elim()?;

        Ok(work.sub_matrix(0, row_count, col_count, col_count * 2))
    }

    pub fn vandermonde(rows: usize, cols: usize) -> Matrix<Element> {
        let mut result = Self::new(rows, cols);
        for r in 0..rows {
            // Distinct field element per row keeps the matrix invertible; each
            // row is the running powers 1, point, point^2, ..., of that element.
            let point = Element::evaluation_point(r);
            let mut power = Element::ONE;
            for c in 0..cols {
                result.set(r, c, power);
                power = power.gf_mul(point);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One generic inversion serves both fields: a Vandermonde block times its
    // inverse is the identity, over GF(2^8) and over the GF(2^16) tower.
    fn invert_round_trips<Element: FieldElement>(size: usize) {
        let m = Matrix::<Element>::vandermonde(size, size);
        let inverse = m.invert().expect("Vandermonde block is invertible");
        let product = m.multiply(&inverse);
        for r in 0..size {
            for c in 0..size {
                let want = if r == c { Element::ONE } else { Element::ZERO };
                assert_eq!(product.get(r, c), want, "r={r} c={c}");
            }
        }
    }

    #[test]
    fn vandermonde_inverse_is_identity_gf8() {
        invert_round_trips::<u8>(20);
    }

    #[test]
    fn vandermonde_inverse_is_identity_gf16() {
        invert_round_trips::<u16>(86);
    }

    // A singular matrix is rejected rather than silently producing garbage.
    #[test]
    fn singular_matrix_errors() {
        let mut m = Matrix::<u16>::new(2, 2);
        m.set(0, 0, 1);
        m.set(0, 1, 1);
        m.set(1, 0, 1);
        m.set(1, 1, 1);
        assert_eq!(m.invert(), Err(Error::SingularMatrix));
    }
}
