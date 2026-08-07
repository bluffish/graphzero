use crate::sample_protocol::{
    ERROR_BAD_REQUEST, ERROR_EMPTY_STORE, ERROR_ENCODING, ERROR_MISSING_FEATURES, ERROR_PROTOCOL,
    FRAME_HELLO, FRAME_SAMPLE, FRAME_SAMPLE_RESULT, HelloAck, SampleRequest, read_frame,
    send_error, validate_hello, write_frame,
};
pub use crate::sample_protocol::{HELLO_ACK_FIXED_LEN, SAMPLE_PROTOCOL_VERSION};
use gz_engine::EngineIdentity;
use gz_features::{
    FeatureCollator, FeatureRow, FeatureSchema, RowTargets, decode_feature_row,
    encode_feature_schema_config, encode_training_targets, validate_feature_row_header,
};
use gz_replay::{ReplayError, ReplayStore, SampleConfig};
use std::num::NonZeroUsize;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ReplayServeConfig {
    pub replay_dir: PathBuf,
    pub socket: PathBuf,
    pub max_batch: usize,
}

impl ReplayServeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.as_os_str().is_empty() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.socket.as_os_str().is_empty() {
            return Err("missing required --socket".to_owned());
        }
        if self.max_batch == 0 {
            return Err("--max-batch must be greater than zero".to_owned());
        }
        Ok(())
    }
}

pub fn run(config: ReplayServeConfig) -> Result<(), String> {
    let server = ReplaySampleServer::bind(config)?;
    loop {
        if let Err(error) = server.accept_concurrent() {
            eprintln!("replay sample accept failed: {error}");
        }
    }
}

pub fn run_one(config: ReplayServeConfig) -> Result<(), String> {
    ReplaySampleServer::bind(config)?.accept_one()
}

/// Serve loop over a store shared with a live producer (the in-process
/// sample service of `graphzero selfplay --serve-socket`). Appends are
/// serialized inside the store while committed rows remain concurrently
/// sampleable, so one process still owns the RocksDB writer.
pub fn run_shared(
    store: std::sync::Arc<ReplayStore>,
    socket: PathBuf,
    max_batch: usize,
) -> Result<(), String> {
    let server = ReplaySampleServer::bind_shared(store, socket, max_batch)?;
    loop {
        if let Err(error) = server.accept_concurrent() {
            eprintln!("replay sample accept failed: {error}");
        }
    }
}

struct ReplaySampleServer {
    listener: UnixListener,
    store: std::sync::Arc<ReplayStore>,
    schema: FeatureSchema,
    engine_identity: EngineIdentity,
    max_batch: NonZeroUsize,
}

struct ReplaySampleSession {
    store: std::sync::Arc<ReplayStore>,
    collator: FeatureCollator,
    engine_identity: EngineIdentity,
    max_batch: NonZeroUsize,
}

impl ReplaySampleServer {
    fn bind(config: ReplayServeConfig) -> Result<Self, String> {
        config.validate()?;
        let store = ReplayStore::open(&config.replay_dir).map_err(|error| error.to_string())?;
        Self::bind_shared(std::sync::Arc::new(store), config.socket, config.max_batch)
    }

    fn bind_shared(
        store: std::sync::Arc<ReplayStore>,
        socket: PathBuf,
        max_batch: usize,
    ) -> Result<Self, String> {
        let schema_config = store
            .feature_schema()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "store was not produced by featurized selfplay".to_owned())?;
        let schema = FeatureSchema::new(schema_config).map_err(|error| error.to_string())?;
        let engine_identity = store
            .engine_identity()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "replay store has no engine identity".to_owned())?;
        let max_batch = NonZeroUsize::new(max_batch)
            .ok_or_else(|| "--max-batch must be greater than zero".to_owned())?;

        if socket.exists() {
            std::fs::remove_file(&socket).map_err(|error| error.to_string())?;
        }
        let listener = UnixListener::bind(&socket).map_err(|error| error.to_string())?;

        Ok(Self {
            listener,
            store,
            schema,
            engine_identity,
            max_batch,
        })
    }

    fn accept_stream(&self) -> Result<UnixStream, String> {
        let (stream, _) = self.listener.accept().map_err(|error| error.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        Ok(stream)
    }

    fn session(&self) -> ReplaySampleSession {
        ReplaySampleSession {
            store: std::sync::Arc::clone(&self.store),
            collator: FeatureCollator::new(self.schema.clone(), self.max_batch),
            engine_identity: self.engine_identity,
            max_batch: self.max_batch,
        }
    }

    fn accept_one(&self) -> Result<(), String> {
        let mut stream = self.accept_stream()?;
        self.session().handle_client(&mut stream)
    }

    fn accept_concurrent(&self) -> Result<(), String> {
        let mut stream = self.accept_stream()?;
        let mut session = self.session();
        std::thread::Builder::new()
            .name("replay-sample-client".to_owned())
            .spawn(move || {
                if let Err(error) = session.handle_client(&mut stream) {
                    // A client disappearing or timing out ends only its own
                    // session; the listener keeps accepting other trainers.
                    eprintln!("replay sample connection ended: {error}");
                }
            })
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

impl ReplaySampleSession {
    fn handle_client(&mut self, stream: &mut UnixStream) -> Result<(), String> {
        let mut read_buf = Vec::new();
        let mut write_buf = Vec::new();
        let mut batch_buf = Vec::new();
        let mut target_buf = Vec::new();

        let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        else {
            return Ok(());
        };
        if frame_type != FRAME_HELLO {
            send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected HELLO")?;
            return Ok(());
        }
        if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
            send_error(stream, &mut write_buf, error.0, error.1)?;
            return Ok(());
        }

        while let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        {
            // A repeated HELLO re-acks with fresh produced_rows so a
            // long-lived trainer connection can watch production advance.
            if frame_type == FRAME_HELLO {
                if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
                continue;
            }
            if frame_type != FRAME_SAMPLE {
                send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected SAMPLE")?;
                return Ok(());
            }
            match self.handle_sample(payload, &mut batch_buf, &mut target_buf) {
                Ok(()) => {
                    let gzfb_len = (batch_buf.len() as u32).to_le_bytes();
                    write_frame(
                        stream,
                        &mut write_buf,
                        FRAME_SAMPLE_RESULT,
                        &[&gzfb_len, &batch_buf, &target_buf],
                    )
                    .map_err(|error| error.to_string())?;
                }
                Err(error) => {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn handle_hello(
        &self,
        payload: &[u8],
        stream: &mut UnixStream,
        write_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        validate_hello(payload)?;

        let mut schema_config = Vec::new();
        encode_feature_schema_config(self.collator.schema().config(), &mut schema_config)
            .map_err(|_| (ERROR_ENCODING, "failed to encode schema config"))?;
        let (episodes, episodes_stopped) = self.store.episode_counters();
        // Unseeded EMAs surface as zeros; consumers gate on the episode count.
        let (cost_ema, len_ema, stop_ema) = self.store.outcome_emas().unwrap_or((0.0, 0.0, 0.0));
        // Unseeded surfaces as -1.0: 0.0 is a legitimate all-loss rate.
        let win_ema = self.store.win_rate_ema().unwrap_or(-1.0);
        let latency_ema = self.store.episode_latency_ema().unwrap_or(-1.0);
        let (value_sign_early_ema, value_sign_late_ema) = self.store.value_sign_accuracy_emas();
        let value_sign_early_ema = value_sign_early_ema.unwrap_or(-1.0);
        let value_sign_late_ema = value_sign_late_ema.unwrap_or(-1.0);
        let best_cost = self.store.best_cost().unwrap_or(0.0);
        let symmetric = self.store.symmetric_selfplay_metrics();
        let counters = self.store.counters();
        let symmetric_metrics = symmetric.map(|metrics| {
            [
                metrics.p1_win_rate_ema as f32,
                metrics.p2_win_rate_ema as f32,
                metrics.draw_rate_ema as f32,
                metrics.p1_terminal_cost_ema as f32,
                metrics.p2_terminal_cost_ema as f32,
                metrics.terminal_cost_margin_ema as f32,
                metrics.terminal_cost_best as f32,
                metrics.p1_episode_len_ema as f32,
                metrics.p2_episode_len_ema as f32,
                metrics.episode_len_margin_ema as f32,
            ]
        });
        let ack = HelloAck {
            feature_schema_hash: self.collator.schema().hash(),
            max_batch: self.max_batch,
            produced_rows: counters.produced_rows,
            episodes,
            episodes_stopped,
            outcome_emas: [cost_ema as f32, len_ema as f32, stop_ema as f32],
            learner_win_rate_ema: win_ema as f32,
            episode_latency_ema: latency_ema as f32,
            best_cost: best_cost as f32,
            value_sign_accuracy_emas: [value_sign_early_ema as f32, value_sign_late_ema as f32],
            symmetric_metrics,
            engine_identity: self.engine_identity,
            feature_schema: &schema_config,
        };
        let mut ack_payload = Vec::new();
        ack.encode_into(&mut ack_payload);
        write_frame(
            stream,
            write_buf,
            crate::sample_protocol::FRAME_HELLO_ACK,
            &[&ack_payload],
        )
        .map_err(|_| (ERROR_PROTOCOL, "failed to write HELLO_ACK"))
    }

    fn handle_sample(
        &mut self,
        payload: &[u8],
        batch_buf: &mut Vec<u8>,
        target_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        let request = SampleRequest::decode(payload, self.max_batch)?;

        let sample = SampleConfig {
            batch: request.batch,
            window_rows: request.window_rows,
            seed: request.seed,
        };
        let rows = if request.terminal_only {
            self.store.sample_terminal_rows(sample)
        } else {
            self.store.sample_rows(sample)
        }
        .map_err(sample_error)?;
        let mut feature_rows = Vec::<FeatureRow>::with_capacity(rows.len());
        let mut targets = Vec::<RowTargets>::with_capacity(rows.len());
        let schema_hash = self.collator.schema().hash();

        for (_, row) in rows {
            let bytes = row
                .feature_row
                .ok_or((ERROR_MISSING_FEATURES, "row is missing feature payload"))?;
            validate_feature_row_header(&bytes, &schema_hash)
                .map_err(|_| (ERROR_ENCODING, "feature row schema mismatch"))?;
            let feature_row =
                decode_feature_row(&bytes).map_err(|_| (ERROR_ENCODING, "bad feature row"))?;
            let reward = row
                .reward_target
                .ok_or((ERROR_ENCODING, "missing reward target"))?;
            targets.push(RowTargets {
                policy: row.policy_target,
                value: row.value_target,
                horizon_value: row.horizon_value_targets,
                reward,
            });
            feature_rows.push(feature_row);
        }

        self.collator
            .collate_into(&feature_rows, batch_buf)
            .map_err(|_| (ERROR_ENCODING, "feature collation failed"))?;
        encode_training_targets(
            &targets,
            self.max_batch.get(),
            self.collator.schema().config().max_actions as usize,
            target_buf,
        )
        .map_err(|_| (ERROR_ENCODING, "target encoding failed"))?;

        Ok(())
    }
}

fn sample_error(error: ReplayError) -> (u32, &'static str) {
    match error {
        ReplayError::Empty => (ERROR_EMPTY_STORE, "replay store is empty"),
        ReplayError::DataModeMismatch => (ERROR_BAD_REQUEST, "replay data mode mismatch"),
        _ => (ERROR_BAD_REQUEST, "sampling failed"),
    }
}
