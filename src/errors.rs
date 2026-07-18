//! Error types returned by the codec.

use core::fmt;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Error {
    TooFewShards,
    TooManyShards,
    TooFewDataShards,
    TooManyDataShards,
    TooFewParityShards,
    TooManyParityShards,
    TooFewBufferShards,
    TooManyBufferShards,
    IncorrectShardSize,
    TooFewShardsPresent,
    EmptyShard,
    InvalidShardFlags,
    InvalidIndex,
}

impl Error {
    fn as_str(&self) -> &'static str {
        match *self {
            Error::TooFewShards => "The number of provided shards is smaller than the one in codec",
            Error::TooManyShards => "The number of provided shards is greater than the one in codec",
            Error::TooFewDataShards => "The number of provided data shards is smaller than the one in codec",
            Error::TooManyDataShards => "The number of provided data shards is greater than the one in codec",
            Error::TooFewParityShards => "The number of provided parity shards is smaller than the one in codec",
            Error::TooManyParityShards => "The number of provided parity shards is greater than the one in codec",
            Error::TooFewBufferShards => "The number of provided buffer shards is smaller than the number of parity shards in codec",
            Error::TooManyBufferShards => "The number of provided buffer shards is greater than the number of parity shards in codec",
            Error::IncorrectShardSize => "At least one of the provided shards is not of the correct size",
            Error::TooFewShardsPresent => "The number of shards present is smaller than number of parity shards, cannot reconstruct missing shards",
            Error::EmptyShard => "The first shard provided is of zero length",
            Error::InvalidShardFlags => "The number of flags does not match the total number of shards",
            Error::InvalidIndex => "The data shard index provided is greater or equal to the number of data shards in codec",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::error::Error for Error {}
