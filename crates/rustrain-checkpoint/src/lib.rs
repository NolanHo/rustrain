// rustrain-checkpoint: safetensors I/O, manifest schema, delta/adapter/shard

pub mod io;
pub mod manifest;

#[cfg(feature = "tch")]
pub mod safetensors;
