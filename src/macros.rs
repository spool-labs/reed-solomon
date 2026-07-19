//! Crate macros: the FFT program expander
//!
//! fft_run expands a dumped encode program (see fft_programs.rs) into
//! straight-line engine calls. Every register index is a literal, so the
//! register array decomposes into SSA values and lives entirely in machine
//! registers.
//!
//! Token forms, one per op:
//!   ld reg shard;      register = load(input[shard] + offset)
//!   mx dst src c;      register[dst] ^= c * register[src]
//!   xo dst src;        register[dst] ^= register[src]
//!   mc dst src c;      register[dst] = c * register[src]
//!   cp dst src;        register[dst] = register[src]
//!   st parity reg;     store(output[parity] + offset, register[reg])

macro_rules! fft_run {
    ($engine:ty, $r:ident, $in:ident, $out:ident, $off:ident; $($rest:tt)*) => {
        $crate::macros::fft_run!(@op $engine, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident;) => {};
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; ld $reg:literal $shard:literal; $($rest:tt)*) => {
        $r[$reg] = <$e>::load($in[$shard], $off);
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; mx $dst:literal $src:literal $c:literal; $($rest:tt)*) => {
        $r[$dst] = <$e>::mul_xor($r[$dst], $r[$src], $c);
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; xo $dst:literal $src:literal; $($rest:tt)*) => {
        $r[$dst] = <$e>::xor($r[$dst], $r[$src]);
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; mc $dst:literal $src:literal $c:literal; $($rest:tt)*) => {
        $r[$dst] = <$e>::mul_of($r[$src], $c);
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; cp $dst:literal $src:literal; $($rest:tt)*) => {
        $r[$dst] = $r[$src];
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
    (@op $e:ty, $r:ident, $in:ident, $out:ident, $off:ident; st $parity:literal $reg:literal; $($rest:tt)*) => {
        <$e>::store($out[$parity], $off, $r[$reg]);
        $crate::macros::fft_run!(@op $e, $r, $in, $out, $off; $($rest)*);
    };
}
pub(crate) use fft_run;
