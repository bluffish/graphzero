use crate::{JobId, ServiceError, ServiceResult, wire};
use prost::Message;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const LOG_NAME: &str = "receipts.log";
const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub enum ReceiptLedgerConfig {
    Memory { capacity: usize },
    Directory { path: PathBuf },
}

impl Default for ReceiptLedgerConfig {
    fn default() -> Self {
        Self::Memory { capacity: 4096 }
    }
}

pub(crate) enum ReceiptLedger {
    Memory {
        capacity: usize,
        receipts: HashMap<JobId, [u8; 32]>,
    },
    File {
        file: File,
        receipts: HashMap<JobId, [u8; 32]>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReceiptState {
    Missing,
    Identical,
    Conflict,
}

impl ReceiptLedgerConfig {
    pub(crate) fn validate(&self) -> ServiceResult<()> {
        match self {
            Self::Memory { capacity: 0 } => Err(ServiceError::configuration(
                "receipt ledger capacity must be greater than zero",
            )),
            Self::Directory { path } if path.as_os_str().is_empty() => Err(
                ServiceError::configuration("receipt ledger directory is empty"),
            ),
            _ => Ok(()),
        }
    }
}

impl ReceiptLedger {
    pub(crate) fn open(config: &ReceiptLedgerConfig) -> ServiceResult<Self> {
        match config {
            ReceiptLedgerConfig::Memory { capacity } => Ok(Self::Memory {
                capacity: *capacity,
                receipts: HashMap::new(),
            }),
            ReceiptLedgerConfig::Directory { path } => open_file_ledger(path),
        }
    }

    pub(crate) fn can_admit(&self, active_jobs: usize) -> bool {
        match self {
            Self::Memory { capacity, receipts } => active_jobs
                .checked_add(receipts.len())
                .is_some_and(|total| total < *capacity),
            Self::File { .. } => true,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Memory { receipts, .. } | Self::File { receipts, .. } => receipts.len(),
        }
    }

    pub(crate) fn state(&self, job_id: JobId, report: &wire::MeasureReport) -> ReceiptState {
        let receipts = match self {
            Self::Memory { receipts, .. } | Self::File { receipts, .. } => receipts,
        };
        match receipts.get(&job_id) {
            None => ReceiptState::Missing,
            Some(expected) if *expected == report_digest(report) => ReceiptState::Identical,
            Some(_) => ReceiptState::Conflict,
        }
    }

    pub(crate) fn contains(&self, job_id: JobId) -> bool {
        match self {
            Self::Memory { receipts, .. } | Self::File { receipts, .. } => {
                receipts.contains_key(&job_id)
            }
        }
    }

    pub(crate) fn commit(
        &mut self,
        job_id: JobId,
        report: &wire::MeasureReport,
    ) -> ServiceResult<ReceiptState> {
        let digest = report_digest(report);
        match self {
            Self::Memory { capacity, receipts } => {
                if let Some(existing) = receipts.get(&job_id) {
                    return Ok(if *existing == digest {
                        ReceiptState::Identical
                    } else {
                        ReceiptState::Conflict
                    });
                }
                if receipts.len() >= *capacity {
                    return Err(ServiceError::capacity("coordinator receipt ledger is full"));
                }
                receipts.insert(job_id, digest);
            }
            Self::File { file, receipts } => {
                if let Some(existing) = receipts.get(&job_id) {
                    return Ok(if *existing == digest {
                        ReceiptState::Identical
                    } else {
                        ReceiptState::Conflict
                    });
                }
                let encoded = report.encode_to_vec();
                if encoded.len() > MAX_RECORD_BYTES {
                    return Err(ServiceError::capacity("receipt record is too large"));
                }
                let length = u32::try_from(encoded.len())
                    .map_err(|_| ServiceError::capacity("receipt record is too large"))?;
                file.write_all(&length.to_le_bytes())
                    .and_then(|()| file.write_all(&encoded))
                    .and_then(|()| file.sync_data())
                    .map_err(|error| ServiceError::io(error.to_string()))?;
                receipts.insert(job_id, digest);
            }
        }
        Ok(ReceiptState::Missing)
    }
}

fn open_file_ledger(directory: &Path) -> ServiceResult<ReceiptLedger> {
    std::fs::create_dir_all(directory).map_err(|error| ServiceError::io(error.to_string()))?;
    let path = directory.join(LOG_NAME);
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .map_err(|error| ServiceError::io(error.to_string()))?;
    let (receipts, valid_bytes) = read_receipts(&mut file)?;
    let actual_bytes = file
        .metadata()
        .map_err(|error| ServiceError::io(error.to_string()))?
        .len();
    if valid_bytes != actual_bytes {
        file.set_len(valid_bytes)
            .map_err(|error| ServiceError::io(error.to_string()))?;
        file.sync_data()
            .map_err(|error| ServiceError::io(error.to_string()))?;
    }
    file.seek(SeekFrom::End(0))
        .map_err(|error| ServiceError::io(error.to_string()))?;
    Ok(ReceiptLedger::File { file, receipts })
}

fn read_receipts(file: &mut File) -> ServiceResult<(HashMap<JobId, [u8; 32]>, u64)> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| ServiceError::io(error.to_string()))?;
    let mut receipts = HashMap::new();
    let mut valid_bytes = 0_u64;
    loop {
        let mut length = [0_u8; 4];
        match file.read_exact(&mut length) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(ServiceError::io(error.to_string())),
        }
        let length = u32::from_le_bytes(length) as usize;
        if length == 0 || length > MAX_RECORD_BYTES {
            return Err(ServiceError::protocol(
                "receipt ledger contains an invalid record length",
            ));
        }
        let mut encoded = vec![0_u8; length];
        match file.read_exact(&mut encoded) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(ServiceError::io(error.to_string())),
        }
        let report = wire::MeasureReport::decode(encoded.as_slice())
            .map_err(|error| ServiceError::protocol(error.to_string()))?;
        let job_id = JobId::from_slice(&report.job_id)?;
        let digest = *blake3::hash(&encoded).as_bytes();
        if let Some(existing) = receipts.insert(job_id, digest)
            && existing != digest
        {
            return Err(ServiceError::protocol(
                "receipt ledger contains conflicting reports for one job",
            ));
        }
        valid_bytes = valid_bytes
            .checked_add(4 + length as u64)
            .ok_or_else(|| ServiceError::capacity("receipt ledger offset overflow"))?;
    }
    Ok((receipts, valid_bytes))
}

fn report_digest(report: &wire::MeasureReport) -> [u8; 32] {
    *blake3::hash(&report.encode_to_vec()).as_bytes()
}
