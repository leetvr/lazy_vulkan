use std::fmt;

use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    PoolExhausted,
    NoCompatibleMemoryType,
    DedicatedAllocationRequired,
    InvalidPoolConfiguration,
    InvalidStagingConfiguration,
    StagingExhausted,
    Vulkan(vk::Result),
}

impl From<vk::Result> for AllocationError {
    fn from(error: vk::Result) -> Self {
        Self::Vulkan(error)
    }
}

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolExhausted => {
                formatter.write_str("no aligned range fits in the reserved memory pool")
            }
            Self::NoCompatibleMemoryType => {
                formatter.write_str("resource is incompatible with the reserved memory type")
            }
            Self::DedicatedAllocationRequired => {
                formatter.write_str("resource requires a dedicated memory allocation")
            }
            Self::InvalidPoolConfiguration => {
                formatter.write_str("invalid memory pool reservation")
            }
            Self::InvalidStagingConfiguration => {
                formatter.write_str("invalid staging buffer capacity")
            }
            Self::StagingExhausted => {
                formatter.write_str("upload does not fit in the staging buffer")
            }
            Self::Vulkan(error) => write!(formatter, "Vulkan allocation failed: {error}"),
        }
    }
}

impl std::error::Error for AllocationError {}
