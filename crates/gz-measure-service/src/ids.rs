use crate::{ServiceError, ServiceResult};
use std::fmt;
use std::str::FromStr;

macro_rules! define_id {
    ($name:ident, $len:literal) => {
        #[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name([u8; $len]);

        impl $name {
            pub const BYTE_LEN: usize = $len;

            #[must_use]
            pub const fn from_bytes(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }

            pub fn from_slice(bytes: &[u8]) -> ServiceResult<Self> {
                let bytes: [u8; $len] = bytes.try_into().map_err(|_| {
                    ServiceError::protocol(format!(
                        "{} must be {} bytes, got {}",
                        stringify!($name),
                        $len,
                        bytes.len()
                    ))
                })?;
                Ok(Self(bytes))
            }

            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; $len] {
                &self.0
            }

            #[must_use]
            pub fn to_vec(self) -> Vec<u8> {
                self.0.to_vec()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                for byte in self.0 {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({self})", stringify!($name))
            }
        }

        impl FromStr for $name {
            type Err = ServiceError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                if value.len() != $len * 2 {
                    return Err(ServiceError::configuration(format!(
                        "{} hex must be {} characters, got {}",
                        stringify!($name),
                        $len * 2,
                        value.len()
                    )));
                }

                let mut bytes = [0; $len];
                for (index, byte) in bytes.iter_mut().enumerate() {
                    let high = hex_digit(value.as_bytes()[index * 2]).ok_or_else(|| {
                        ServiceError::configuration(format!(
                            "{} contains invalid hex",
                            stringify!($name)
                        ))
                    })?;
                    let low = hex_digit(value.as_bytes()[index * 2 + 1]).ok_or_else(|| {
                        ServiceError::configuration(format!(
                            "{} contains invalid hex",
                            stringify!($name)
                        ))
                    })?;
                    *byte = (high << 4) | low;
                }
                Ok(Self(bytes))
            }
        }
    };
}

define_id!(DeviceId, 16);
define_id!(SessionId, 16);
define_id!(LeaseId, 16);
define_id!(RequestNonce, 16);
define_id!(ArtifactDigest, 32);
define_id!(CertificateFingerprint, 32);
define_id!(DeviceProfileHash, 32);
define_id!(MeasurementKey, 32);
define_id!(JobId, 32);

impl DeviceId {
    #[must_use]
    pub fn random() -> Self {
        let mut bytes = [0; Self::BYTE_LEN];
        rand::fill(&mut bytes);
        Self(bytes)
    }
}

impl SessionId {
    #[must_use]
    pub fn random() -> Self {
        let mut bytes = [0; Self::BYTE_LEN];
        rand::fill(&mut bytes);
        Self(bytes)
    }
}

impl LeaseId {
    #[must_use]
    pub fn random() -> Self {
        let mut bytes = [0; Self::BYTE_LEN];
        rand::fill(&mut bytes);
        Self(bytes)
    }
}

impl RequestNonce {
    #[must_use]
    pub fn random() -> Self {
        let mut bytes = [0; Self::BYTE_LEN];
        rand::fill(&mut bytes);
        Self(bytes)
    }
}

#[must_use]
pub fn certificate_fingerprint(certificate_der: &[u8]) -> CertificateFingerprint {
    CertificateFingerprint::from_bytes(*blake3::hash(certificate_der).as_bytes())
}

const fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
