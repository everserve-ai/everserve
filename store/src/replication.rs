// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Replication between stores (decision 0016): what one commit wrote for a rank goes to the
//! stores on the nodes holding that rank's eligible spares, so a restore there is local.
//!
//! The source acknowledges a commit as soon as its own tables have it; replication runs behind,
//! one task per replicated rank, over the wire crate's transport (a Unix socket on the same node,
//! TCP under mutual TLS across nodes). The replica applies rows with the same per-slot write steps
//! as the source, so an incremental restore from it continues from the same watermark. A target
//! that is unreachable is retried with backoff and the rank's lag, in steps, is a metric.
//!
//! No step is ever dropped. Each target's queue is bounded at its bytes plus one step (the
//! store's table budget unless configured: a queue never needs more than the data it mirrors;
//! a step takes whatever room is left rather than waiting forever for room it may never get),
//! and a commit whose rows do not fit waits for room before it is acknowledged, so a replica
//! that cannot keep up
//! slows the engine to what its link carries rather than being left with a hole: a dropped step's
//! rows never arrive while the next step's counts do, and a restore from that replica serves
//! stale slots as written. The wait is a metric. A target the coordinator removes releases every
//! reservation on it.
//!
//! A replica is never silently behind. On every connection the carrier asks where the replica
//! stands (`ReplicaStateQuery`) and, unless that is where it last left it, sends the rank as the
//! source holds it now before the queue: a target named after the engine ran, a replica or a
//! source restarted, acknowledgements lost with a link. Steps the snapshot already covers are
//! skipped. This is the follower contract of a replicated log (Raft's snapshot install, Kafka's
//! fetch from the follower's own end): the leader learns the follower's position and fills from
//! there; the follower never guesses.
//!
//! A step goes over the wire in parts that each fit one frame: a prefill burst for one rank runs
//! to 236 MB (measured over a VPC), and sent as one frame over a 16 MiB cap it is refused whole,
//! so the replica takes no byte while the source reconnects every 250 ms. The parts of a step are
//! sent back to back and their replies read after, in order, so a step costs one round trip
//! however many parts it has. The replica applies each part's rows as they come, with the same
//! epoch and step; the request counts ride in the last part, so a restore sees a step's counts
//! only once its rows are all there.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use everserve_wire::frame as wire;
use everserve_wire::proto::everserve::mirror::v1::{
    CaughtUp, MirrorReply, MirrorRequest, ReplicaStateQuery, Replicated, ReplicatedTable,
    mirror_reply, mirror_request,
};
use everserve_wire::transport::{Address, Tls, connect, jittered};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::metrics::StoreMetrics;
use crate::storage::{SnapshotPlan, Storage};

/// The unit a queue's room is counted in.
const KIB: usize = 1024;
/// The first wait after a failed connection; it doubles up to [`RECONNECT_MAX`].
pub const RECONNECT_MIN: Duration = Duration::from_millis(200);
/// The longest wait between connection attempts.
pub const RECONNECT_MAX: Duration = Duration::from_secs(5);
/// The default for how long a carrier waits for a replica's reply before taking the link as
/// failed (`--replication-reply-timeout-s`): a peer whose host is gone without a reset (a
/// half-open link) answers nothing, and the coordinator's node loss removes the target only
/// when the node's daemon is lost too. Longer than any apply of a frame-sized part, shorter
/// than an operator's patience.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(30);
/// What a part of `len` encoded bytes costs as a frame: inside its `MirrorRequest` envelope
/// (one tag byte and a length), which is what the wire's cap is measured against.
fn framed(len: usize) -> usize {
    1 + prost::length_delimiter_len(len) + len
}

/// What `table` adds to a part: its tag byte, its length, and itself.
fn table_cost(table: &ReplicatedTable) -> usize {
    use prost::Message;

    framed(table.encoded_len())
}

/// A step as parts that each fit a frame of `frame` bytes, in table order.
///
/// Tables are packed whole while they fit; a table that does not fit a frame by itself is split
/// by runs of slots, each run with its rows. Every part carries the step's engine, rank, epoch
/// and step; the request counts ride in the last part only. A step that fits is one part,
/// itself. The accounting is exact: a part's framed size is never over `frame`.
#[must_use]
pub fn parts(msg: Replicated, frame: usize) -> Vec<Replicated> {
    use prost::Message;

    let head = |tables: Vec<ReplicatedTable>| Replicated {
        engine: msg.engine.clone(),
        rank: msg.rank,
        epoch: msg.epoch,
        step: msg.step,
        tables,
        request_ids: Vec::new(),
        committed_tokens: Vec::new(),
    };
    let tail = Replicated {
        request_ids: msg.request_ids.clone(),
        committed_tokens: msg.committed_tokens.clone(),
        ..head(Vec::new())
    };
    // the last part carries the counts, so every part is packed as if it did
    let base = tail.encoded_len();
    let mut out: Vec<Replicated> = Vec::new();
    let mut part = head(Vec::new());
    let mut len = base;
    for table in msg.tables {
        let cost = table_cost(&table);
        if framed(len + cost) > frame && !part.tables.is_empty() {
            out.push(std::mem::replace(&mut part, head(Vec::new())));
            len = base;
        }
        if framed(base + cost) <= frame || table.slots.is_empty() {
            len += cost;
            part.tables.push(table);
            continue;
        }
        // one table over a frame: by runs of slots, each run with its rows
        let row = table.data.len() / table.slots.len();
        let mut at = 0;
        while at < table.slots.len() {
            let mut n = ((frame - framed(base)) / row.max(1)).clamp(1, table.slots.len() - at);
            let mut run = loop {
                let run = ReplicatedTable {
                    table: table.table,
                    dtype: table.dtype.clone(),
                    shape: table.shape.clone(),
                    slots: table.slots[at..at + n].to_vec(),
                    data: table.data[at * row..(at + n) * row].to_vec(),
                    // each row keeps its checksum (decision 0030)
                    crc32c: table.crc32c[at..at + n].to_vec(),
                };
                let over = framed(base + table_cost(&run)).saturating_sub(frame);
                if over == 0 || n == 1 {
                    break run;
                }
                n -= (over / row.max(1)).max(1);
            };
            if !part.tables.is_empty() {
                out.push(std::mem::replace(&mut part, head(Vec::new())));
            }
            at += run.slots.len();
            len = base + table_cost(&run);
            part.tables.push(std::mem::take(&mut run));
        }
    }
    part.request_ids = msg.request_ids;
    part.committed_tokens = msg.committed_tokens;
    out.push(part);
    out
}

/// One target of a rank: its queue, the room left in it, and the stop of its carrier.
#[derive(Clone)]
struct Target {
    tx: mpsc::UnboundedSender<Step>,
    room: Arc<Semaphore>,
    stop: CancellationToken,
}

impl Target {
    /// The target is gone: its carrier stops wherever it is, and every commit waiting for room
    /// in its queue goes on without it.
    fn remove(&self) {
        self.stop.cancel();
        self.room.close();
    }
}

/// A queued step with the room it holds until the target acknowledges it.
struct Step {
    msg: Replicated,
    _held: OwnedSemaphorePermit,
}

/// One replicated rank's targets and the queues feeding them.
struct RankReplication {
    senders: Vec<(String, Target)>,
}

/// The replicator: which ranks go where, and the tasks that carry them.
pub struct Replicator {
    tls: Option<Tls>,
    server_name: String,
    ranks: Mutex<BTreeMap<(String, u32), RankReplication>>,
    metrics: StoreMetrics,
    queue_bytes: AtomicUsize,
    /// How long a carrier waits for a reply, in milliseconds.
    reply_timeout_ms: AtomicU64,
    /// The node this store is on, as its daemon said: what a carrier calls itself (0026).
    node: std::sync::Mutex<String>,
    /// The store's tables, for the snapshot a carrier sends a replica that is not where it
    /// left it.
    storage: Arc<Mutex<Storage>>,
}

impl Replicator {
    /// A replicator dialling peers with `tls` (needed for TCP targets), verifying `server_name`
    /// on their certificates, each target's queue bounded at `queue_bytes`.
    #[must_use]
    pub fn new(
        tls: Option<Tls>,
        server_name: String,
        metrics: StoreMetrics,
        queue_bytes: usize,
        storage: Arc<Mutex<Storage>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            tls,
            server_name,
            ranks: Mutex::new(BTreeMap::new()),
            metrics,
            queue_bytes: AtomicUsize::new(queue_bytes),
            reply_timeout_ms: AtomicU64::new(
                u64::try_from(REPLY_TIMEOUT.as_millis()).unwrap_or(u64::MAX),
            ),
            node: std::sync::Mutex::new(String::new()),
            storage,
        })
    }

    /// The node this store is on; every carrier names it to its replica from now on.
    pub fn set_node(&self, node: &str) {
        if let Ok(mut n) = self.node.lock() {
            node.clone_into(&mut n);
        }
    }

    fn node(&self) -> String {
        self.node.lock().map(|n| n.clone()).unwrap_or_default()
    }

    /// Wait `timeout` for a replica's reply on every link from now on.
    pub fn set_reply_timeout(&self, timeout: Duration) {
        self.reply_timeout_ms.store(
            u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn reply_timeout(&self) -> Duration {
        Duration::from_millis(self.reply_timeout_ms.load(Ordering::Relaxed))
    }

    /// Bound every target's queue made from now on at `bytes`.
    pub fn set_queue_bytes(&self, bytes: usize) {
        self.queue_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Replicate `rank` of `engine` to `targets` from now on; an empty list stops it. A target
    /// already in place keeps its queue; a target no longer listed has its task dropped.
    ///
    /// # Errors
    ///
    /// A target that is not an address.
    pub async fn set_targets(
        self: &Arc<Self>,
        engine: &str,
        rank: u32,
        targets: &[String],
    ) -> Result<(), everserve_wire::transport::TransportError> {
        let parsed: Vec<(String, Address)> = targets
            .iter()
            .map(|t| Address::parse(t).map(|a| (t.clone(), a)))
            .collect::<Result<_, _>>()?;
        let mut ranks = self.ranks.lock().await;
        let key = (engine.to_owned(), rank);
        let mut senders: Vec<(String, Target)> = Vec::new();
        let old = ranks.remove(&key).map(|r| r.senders).unwrap_or_default();
        for (text, address) in parsed {
            if let Some(kept) = old.iter().find(|(t, _)| *t == text) {
                senders.push(kept.clone());
                continue;
            }
            let (tx, rx) = mpsc::unbounded_channel();
            let room = Arc::new(Semaphore::new(
                (self.queue_bytes.load(Ordering::Relaxed) / KIB).max(1),
            ));
            let stop = CancellationToken::new();
            tokio::spawn(Arc::clone(self).carry(
                engine.to_owned(),
                rank,
                text.clone(),
                address,
                rx,
                stop.clone(),
            ));
            senders.push((text, Target { tx, room, stop }));
        }
        for (text, target) in &old {
            if !senders.iter().any(|(t, _)| t == text) {
                target.remove();
                // its lag was its own: a series for a target that is gone would read as a lag
                let _ = self.metrics.replica_lag_steps.remove_label_values(&[
                    engine,
                    &rank.to_string(),
                    text,
                ]);
            }
        }
        info!(
            "{engine}: rank {rank} replicates to {:?}",
            senders.iter().map(|(t, _)| t).collect::<Vec<_>>()
        );
        if !senders.is_empty() {
            ranks.insert(key, RankReplication { senders });
        }
        Ok(())
    }

    /// Whether `rank` of `engine` has any target.
    pub async fn replicates(&self, engine: &str, rank: u32) -> bool {
        self.ranks
            .lock()
            .await
            .contains_key(&(engine.to_owned(), rank))
    }

    /// Queue what a commit wrote for every target of its rank, waiting for room in each
    /// target's queue first: the commit is acknowledged after this returns. The room is held
    /// until the target acknowledges the step, or is dropped from the targets.
    pub async fn queue(&self, msg: Replicated) {
        use prost::Message;

        let targets: Vec<(String, Target)> = {
            let ranks = self.ranks.lock().await;
            match ranks.get(&(msg.engine.clone(), msg.rank)) {
                Some(r) => r.senders.clone(),
                None => return,
            }
        };
        let kib = u32::try_from(msg.encoded_len().div_ceil(KIB)).unwrap_or(u32::MAX);
        for (target, t) in targets {
            // a step over the whole queue takes the whole queue: it goes alone, never waits forever
            let want =
                kib.min(u32::try_from(t.room.available_permits().max(1)).unwrap_or(u32::MAX));
            let t0 = std::time::Instant::now();
            let Ok(held) = Arc::clone(&t.room).acquire_many_owned(want).await else {
                continue; // the target was dropped while waiting
            };
            let waited = t0.elapsed().as_secs_f64();
            if waited > 0.0 {
                self.metrics
                    .replication_wait_seconds_total
                    .with_label_values(&[&msg.engine, &target])
                    .inc_by(waited);
            }
            let _ = t.tx.send(Step {
                msg: msg.clone(),
                _held: held,
            });
        }
    }

    /// One target's task: connect, bring the replica to where this store stands, send each
    /// queued step and wait for its acknowledgement, reconnect with backoff on any failure, and
    /// keep the lag gauge. Ends when the target is removed.
    async fn carry(
        self: Arc<Self>,
        engine: String,
        rank: u32,
        target: String,
        address: Address,
        mut rx: mpsc::UnboundedReceiver<Step>,
        removed: CancellationToken,
    ) {
        let lag = self.metrics.replica_lag_steps.with_label_values(&[
            &engine,
            &rank.to_string(),
            &target,
        ]);
        let mut link = Link {
            engine,
            rank,
            target,
            lag,
            pending: None,
            acked: None,
            served: false,
            reply_timeout: self.reply_timeout(),
            node: self.node(),
            idle_told: false,
            refused_at: None,
        };
        let mut wait = RECONNECT_MIN;
        let carried = async {
            loop {
                let ended = match connect(&address, self.tls.as_ref(), &self.server_name).await {
                    Ok(stream) => self.serve_link(stream, &mut link, &mut rx).await,
                    Err(e) => {
                        warn!(
                            "{}: rank {} replica {} unreachable: {e}",
                            link.engine, link.rank, link.target
                        );
                        LinkEnd::Failed
                    }
                };
                if std::mem::take(&mut link.served) {
                    wait = RECONNECT_MIN; // a link that carried a step earned a fresh backoff
                }
                match ended {
                    LinkEnd::Removed => return,
                    LinkEnd::Failed => {
                        self.metrics
                            .replication_link_failures_total
                            .with_label_values(&[&link.engine, &link.target])
                            .inc();
                        info!(
                            "{}: rank {} replica {}: reconnecting in {wait:?}",
                            link.engine, link.rank, link.target
                        );
                        tokio::time::sleep(jittered(wait)).await;
                        wait = (wait * 2).min(RECONNECT_MAX);
                    }
                }
            }
        };
        tokio::select! {
            () = carried => {}
            () = removed.cancelled() => {}
        }
    }

    /// One connection to the replica: the resync, then the queue until the link fails or the
    /// target is removed.
    async fn serve_link(
        &self,
        stream: everserve_wire::transport::Stream,
        link: &mut Link,
        rx: &mut mpsc::UnboundedReceiver<Step>,
    ) -> LinkEnd {
        let (mut rd, mut wr) = tokio::io::split(stream);
        if self.resync(&mut rd, &mut wr, link).await.is_err() {
            return LinkEnd::Failed;
        }
        link.idle_told = false;
        loop {
            let queued = if let Some(m) = link.pending.take() {
                m
            } else {
                // nothing queued: tell the replica where this carrier's rows end, once, so a
                // restore waiting on a staged replication knows the snapshot is whole (0030)
                if rx.is_empty() && !link.idle_told {
                    if let Some((epoch, step)) = link.acked
                        && tell_caught_up(&mut rd, &mut wr, link, epoch, step)
                            .await
                            .is_err()
                    {
                        return LinkEnd::Failed;
                    }
                    link.idle_told = true;
                }
                let Some(m) = rx.recv().await else {
                    let _ = wr.shutdown().await; // the target was dropped: say goodbye
                    return LinkEnd::Removed;
                };
                m
            };
            // a step the resync already carried, or older than what the replica has
            if link
                .acked
                .is_some_and(|a| (queued.msg.epoch, queued.msg.step) <= a)
            {
                drop(queued);
                continue;
            }
            // a replica that refused the rank as held is not covered: once this store has moved
            // past where it stood then, send the rank as held again before the step
            if link.refused_at.is_some() && self.moved_past_refusal(link).await {
                if self.resync(&mut rd, &mut wr, link).await.is_err() {
                    link.pending = Some(queued);
                    return LinkEnd::Failed;
                }
                if link
                    .acked
                    .is_some_and(|a| (queued.msg.epoch, queued.msg.step) <= a)
                {
                    drop(queued);
                    continue;
                }
            }
            let step = queued.msg.step;
            match send_step(&mut rd, &mut wr, &queued.msg, link.reply_timeout).await {
                Ok(Ok(())) => {
                    self.metrics
                        .replicated_total
                        .with_label_values(&[&link.engine, &link.target])
                        .inc();
                    link.lag.set(i64::try_from(rx.len()).unwrap_or(i64::MAX));
                    link.acked = Some((queued.msg.epoch, step));
                    link.served = true;
                    link.idle_told = false;
                    drop(queued); // its room returns to the queue
                }
                Ok(Err(error)) => {
                    // the replica holds none of that step: it is not covered, and the rank as held
                    // goes to it again before the next step, once this store has moved past it (a
                    // row that failed its checksum, decision 0030, or a step under a move's fence)
                    warn!(
                        "{}: rank {} replica {} refused step {step}: {error}",
                        link.engine, link.rank, link.target
                    );
                    link.acked = None;
                    link.refused_at = Some((queued.msg.epoch, step));
                }
                Err(e @ (wire::WireError::TooLarge(_) | wire::WireError::Decode(_))) => {
                    // not the link's fault: the step itself cannot cross, and it never will
                    self.metrics
                        .replication_dropped_total
                        .with_label_values(&[&link.engine, &link.target])
                        .inc();
                    warn!(
                        "{}: rank {} replica {} cannot take step {step}: {e}; dropped",
                        link.engine, link.rank, link.target
                    );
                }
                Err(e) => {
                    warn!(
                        "{}: rank {} replica {} link failed at step {step}: {e}",
                        link.engine, link.rank, link.target
                    );
                    link.pending = Some(queued);
                    return LinkEnd::Failed;
                }
            }
        }
    }

    /// Whether this store now stands past where it stood when the replica refused the rank.
    async fn moved_past_refusal(&self, link: &Link) -> bool {
        let (_, epoch, step, _) = self.storage.lock().await.state(&link.engine, link.rank);
        link.refused_at.is_some_and(|at| (epoch, step) > at)
    }

    /// The rank as this store holds it, streamed: parts of at most a frame, each run of rows
    /// copied under the storage lock only while it is read, so this store's own commits go on
    /// between parts (a whole-rank copy under the lock holds them for the copy's length, measured
    /// at 2.4 to 2.5 s at 32k tokens, before the first byte leaves). Every part carries the plan's
    /// label; the counts ride the last. Replies are read once every part is out, as for a step.
    async fn send_snapshot<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin>(
        &self,
        rd: &mut R,
        wr: &mut W,
        link: &Link,
        plan: SnapshotPlan,
    ) -> Result<Result<(), String>, wire::WireError> {
        use prost::Message;

        // a table's own fields (number, dtype, shape, the field tags and lengths) beside its rows,
        // and a row's slot id and checksum beside its bytes: generous, so a part never overflows
        const TABLE_OVERHEAD: usize = 256;
        const ROW_OVERHEAD: usize = 16;
        let head = |tables: Vec<ReplicatedTable>| Replicated {
            engine: link.engine.clone(),
            rank: link.rank,
            epoch: plan.epoch,
            step: plan.step,
            tables,
            request_ids: Vec::new(),
            committed_tokens: Vec::new(),
        };
        let base = Replicated {
            request_ids: plan.request_ids.clone(),
            committed_tokens: plan.committed_tokens.clone(),
            ..head(Vec::new())
        }
        .encoded_len();
        let mut part = head(Vec::new());
        let mut len = base;
        let mut sent = 0;
        for &(table, row_bytes) in &plan.tables {
            let mut from = Some(0);
            while let Some(at) = from {
                let room = wire::MAX_FRAME.saturating_sub(framed(len) + TABLE_OVERHEAD);
                let max_rows = room / (row_bytes + ROW_OVERHEAD);
                if max_rows == 0 {
                    if part.tables.is_empty() {
                        return Err(wire::WireError::Io(std::io::Error::other(format!(
                            "table {table}: a row of {row_bytes} bytes does not fit a frame"
                        ))));
                    }
                    wire::send(wr, &request(std::mem::replace(&mut part, head(Vec::new()))))
                        .await?;
                    sent += 1;
                    len = base;
                    continue;
                }
                let run = self.storage.lock().await.snapshot_run(
                    &link.engine,
                    link.rank,
                    table,
                    at,
                    max_rows,
                );
                let Some((run, next)) = run else { break };
                from = next;
                // a table with no written row is still named, once, as a whole snapshot names it
                if run.slots.is_empty() && at > 0 {
                    continue;
                }
                len += table_cost(&run);
                debug_assert!(
                    framed(len) <= wire::MAX_FRAME,
                    "a snapshot part over a frame"
                );
                part.tables.push(run);
            }
        }
        part.request_ids = plan.request_ids;
        part.committed_tokens = plan.committed_tokens;
        wire::send(wr, &request(part)).await?;
        sent += 1;
        for _ in 0..sent {
            let r = reply(rd, link.reply_timeout).await?;
            if !r.error.is_empty() {
                return Ok(Err(r.error));
            }
        }
        Ok(Ok(()))
    }

    /// Bring the replica to where this store stands, unless it is where this carrier left it:
    /// ask its (epoch, step), and send the rank as held when the answer differs.
    async fn resync<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin>(
        &self,
        rd: &mut R,
        wr: &mut W,
        link: &mut Link,
    ) -> Result<(), wire::WireError> {
        let (held, epoch, step, fence) = ask_state(
            rd,
            wr,
            &link.engine,
            link.rank,
            &link.node,
            link.reply_timeout,
        )
        .await?;
        if held && Some((epoch, step)) == link.acked {
            return Ok(());
        }
        let plan = self
            .storage
            .lock()
            .await
            .snapshot_plan(&link.engine, link.rank);
        let Some(plan) = plan else {
            // this store holds nothing for the rank yet: nothing to send, and nothing covered,
            // so the queue's first step goes whatever its number (the prefill is step 0)
            link.acked = None;
            return Ok(());
        };
        let at = (plan.epoch, plan.step);
        if plan.epoch < fence {
            // the replica fences the rank at a later epoch than this store holds it: a snapshot
            // now is refused after the whole transfer (a new holder's rank before its first
            // commit, measured: 5.4 GB and 13 s per take at 32k tokens). Hold it as a refusal,
            // sent once this store has moved past it
            info!(
                "{}: rank {} replica {} fences the rank at epoch {fence}, this store holds it at epoch {}: sent once this store commits at it",
                link.engine, link.rank, link.target, at.0
            );
            self.metrics
                .snapshots_total
                .with_label_values(&["held"])
                .inc();
            link.acked = None;
            link.refused_at = Some(at);
            return Ok(());
        }
        info!(
            "{}: rank {} replica {} {} at epoch {epoch} step {step}, this store at epoch {} step {}: sending the rank as held",
            link.engine,
            link.rank,
            link.target,
            if held {
                "holding rows"
            } else {
                "holding nothing"
            },
            at.0,
            at.1
        );
        if let Err(error) = self.send_snapshot(rd, wr, link, plan).await? {
            // not covered: the replica holds none of it. Sent again once this store has moved
            // on, before the next step, or a new holder's restored rows, refused under the rank's
            // move fence, never reach its sibling
            warn!(
                "{}: rank {} replica {} refused the rank as held: {error}",
                link.engine, link.rank, link.target
            );
            self.metrics
                .snapshots_total
                .with_label_values(&["refused"])
                .inc();
            link.acked = None;
            link.refused_at = Some(at);
            return Ok(());
        }
        self.metrics
            .snapshots_total
            .with_label_values(&["sent"])
            .inc();
        link.acked = Some(at);
        link.refused_at = None;
        Ok(())
    }
}

/// How a connection to a replica ended.
enum LinkEnd {
    /// The target was removed from the rank's list.
    Removed,
    /// The link failed; reconnect with backoff.
    Failed,
}

/// A carrier's state across connections to one target.
struct Link {
    engine: String,
    rank: u32,
    target: String,
    lag: everserve_telemetry::prometheus::IntGauge,
    /// A step the last link failed on, sent first on the next.
    pending: Option<Step>,
    /// What this carrier knows the replica holds through: the (epoch, step) of the last step it
    /// acknowledged, or of the rank as held when that carried rows; None before any. A queued
    /// step at or before it is skipped as already there; nothing is skipped on None.
    acked: Option<(u64, u64)>,
    /// Whether the current link carried a step, which resets the reconnect backoff.
    served: bool,
    /// How long to wait for each of the replica's replies.
    reply_timeout: Duration,
    /// The node this store is on, named to the replica in the first request.
    node: String,
    /// Whether the replica was told this carrier's queue is empty since its last step (decision
    /// 0030): told once per idle period, cleared by every step sent.
    idle_told: bool,
    /// Where this store stood when the replica last refused the rank as held: the rank is sent
    /// again only once this store has moved past it, never in a loop at the same state.
    refused_at: Option<(u64, u64)>,
}

/// One reply from the replica, within `timeout`: a replica that answers nothing has a link
/// that is gone, whatever the socket says.
async fn reply<R: tokio::io::AsyncRead + Unpin>(
    rd: &mut R,
    timeout: Duration,
) -> Result<MirrorReply, wire::WireError> {
    tokio::time::timeout(timeout, wire::recv::<_, MirrorReply>(rd))
        .await
        .unwrap_or_else(|_| {
            Err(wire::WireError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "no reply from the replica",
            )))
        })
}

/// Where the replica stands for `rank`: whether it holds any row, its (epoch, step), and the
/// epoch it fences the rank at.
async fn ask_state<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin>(
    rd: &mut R,
    wr: &mut W,
    engine: &str,
    rank: u32,
    node: &str,
    timeout: Duration,
) -> Result<(bool, u64, u64, u64), wire::WireError> {
    let req = MirrorRequest {
        kind: Some(mirror_request::Kind::ReplicaState(ReplicaStateQuery {
            engine: engine.to_owned(),
            rank,
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            node: node.to_owned(),
        })),
    };
    wire::send(wr, &req).await?;
    match reply(rd, timeout).await?.kind {
        Some(mirror_reply::Kind::ReplicaState(s)) => Ok((s.held, s.epoch, s.step, s.fence_epoch)),
        other => Err(wire::WireError::Io(std::io::Error::other(format!(
            "not a replica state: {other:?}"
        )))),
    }
}

/// Tell the replica this carrier has nothing more queued for the rank through (`epoch`, `step`)
/// (decision 0030). The replica's refusal is logged, not fatal; a failed link is.
async fn tell_caught_up<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin>(
    rd: &mut R,
    wr: &mut W,
    link: &Link,
    epoch: u64,
    step: u64,
) -> Result<(), wire::WireError> {
    let req = MirrorRequest {
        kind: Some(mirror_request::Kind::CaughtUp(CaughtUp {
            engine: link.engine.clone(),
            rank: link.rank,
            epoch,
            step,
        })),
    };
    wire::send(wr, &req).await?;
    let answer = reply(rd, link.reply_timeout).await?;
    if !answer.error.is_empty() {
        warn!(
            "{}: rank {} replica {} refused the caught-up mark: {}",
            link.engine, link.rank, link.target, answer.error
        );
    }
    Ok(())
}

/// `part` as the request that carries it.
fn request(part: Replicated) -> MirrorRequest {
    MirrorRequest {
        kind: Some(mirror_request::Kind::Replicated(part)),
    }
}

/// One step to the replica: every part out, then every reply in, in order. `Ok(Ok(()))` when
/// the replica took it, `Ok(Err(why))` when it refused a part, `Err` when the link failed.
async fn send_step<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin>(
    rd: &mut R,
    wr: &mut W,
    msg: &Replicated,
    timeout: Duration,
) -> Result<Result<(), String>, wire::WireError> {
    let mut sent = 0;
    for part in parts(msg.clone(), wire::MAX_FRAME) {
        wire::send(wr, &request(part)).await?;
        sent += 1;
    }
    for _ in 0..sent {
        let r = reply(rd, timeout).await?;
        if !r.error.is_empty() {
            return Ok(Err(r.error));
        }
    }
    Ok(Ok(()))
}

#[cfg(test)]
mod tests {
    use everserve_wire::proto::everserve::mirror::v1::ReplicatedTable;
    use prost::Message;

    use super::*;

    fn table(id: u32, slots: usize, row: usize) -> ReplicatedTable {
        let data: Vec<u8> = (0..slots * row)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        ReplicatedTable {
            table: id,
            dtype: "uint8".into(),
            shape: vec![u32::try_from(row).unwrap()],
            slots: (0..slots).map(|s| i64::try_from(s).unwrap()).collect(),
            crc32c: data.chunks(row).map(crc32c::crc32c).collect(),
            data,
        }
    }

    fn step(tables: Vec<ReplicatedTable>) -> Replicated {
        Replicated {
            engine: "e".into(),
            rank: 1,
            epoch: 2,
            step: 7,
            tables,
            request_ids: vec!["r1".into(), "r2".into()],
            committed_tokens: vec![10, 20],
        }
    }

    #[test]
    fn a_step_that_fits_is_one_part_itself() {
        let msg = step(vec![table(0, 4, 16), table(1, 4, 16)]);
        assert_eq!(parts(msg.clone(), wire::MAX_FRAME), vec![msg]);
    }

    #[test]
    fn a_step_over_the_budget_goes_as_parts_under_it_with_the_counts_last() {
        let msg = step((0..6).map(|t| table(t, 100, 1000)).collect()); // 600 KB
        let out = parts(msg.clone(), 250_000);
        assert!(out.len() >= 3, "{}", out.len());
        for p in &out {
            assert!(framed(p.encoded_len()) <= 250_000, "{}", p.encoded_len());
            assert_eq!((p.engine.as_str(), p.rank, p.epoch, p.step), ("e", 1, 2, 7));
        }
        let (last, before) = out.split_last().unwrap();
        assert!(
            before
                .iter()
                .all(|p| p.request_ids.is_empty() && p.committed_tokens.is_empty())
        );
        assert_eq!(
            (&last.request_ids, &last.committed_tokens),
            (&msg.request_ids, &msg.committed_tokens)
        );
        let tables: Vec<_> = out.into_iter().flat_map(|p| p.tables).collect();
        assert_eq!(tables, msg.tables, "tables whole and in order");
    }

    #[test]
    fn a_table_over_the_budget_is_split_by_slots_with_each_runs_rows() {
        let big = table(3, 1000, 1000); // 1 MB in one table
        let msg = step(vec![table(0, 2, 8), big.clone()]);
        let out = parts(msg, 300_000);
        assert!(out.len() >= 4, "{}", out.len());
        let runs: Vec<_> = out
            .iter()
            .flat_map(|p| p.tables.iter())
            .filter(|t| t.table == 3)
            .collect();
        assert!(runs.len() >= 4);
        for r in &runs {
            assert_eq!(r.data.len(), r.slots.len() * 1000);
            assert_eq!((r.dtype.as_str(), &r.shape), ("uint8", &big.shape));
            // each row keeps its own checksum through the split (decision 0030)
            assert_eq!(
                r.crc32c,
                r.data.chunks(1000).map(crc32c::crc32c).collect::<Vec<_>>()
            );
        }
        let slots: Vec<i64> = runs.iter().flat_map(|r| r.slots.iter().copied()).collect();
        let data: Vec<u8> = runs.iter().flat_map(|r| r.data.iter().copied()).collect();
        assert_eq!((slots, data), (big.slots, big.data));
        for p in &out {
            assert!(framed(p.encoded_len()) <= 300_000, "{}", p.encoded_len());
        }
    }

    #[test]
    fn a_part_is_packed_to_the_frame_with_no_slack_to_spare() {
        // a frame just big enough for two tables with the counts: the third starts a new part
        let msg = step(vec![
            table(0, 10, 100),
            table(1, 10, 100),
            table(2, 10, 100),
        ]);
        let two = {
            let p = Replicated {
                tables: msg.tables[..2].to_vec(),
                ..msg.clone()
            };
            framed(p.encoded_len())
        };
        let out = parts(msg.clone(), two);
        assert_eq!(
            out.iter().map(|p| p.tables.len()).collect::<Vec<_>>(),
            vec![2, 1]
        );
        let out = parts(msg, two - 1);
        assert_eq!(
            out.iter().map(|p| p.tables.len()).collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
    }

    #[tokio::test]
    async fn a_snapshot_over_a_frame_streams_in_parts_that_each_fit_and_the_replica_ends_equal() {
        use everserve_wire::proto::everserve::mirror::v1::StagedStep;

        // rows of 16 KiB (shape [2, 4096] of bfloat16), 1,100 written in one table: about
        // 17.6 MB, over a frame, so the snapshot must stream as more than one part
        const ROW: usize = 16 * 1024;
        const SLOTS: usize = 1200;
        const WRITTEN: usize = 1100;
        let dir = tempfile::tempdir().expect("tmp");
        let mut src = Storage::new(dir.path().join("src"), SLOTS, 64 << 20);
        let d = src.open("e", 0).expect("open");
        let mut bytes = Vec::with_capacity(WRITTEN * ROW);
        for s in 0..WRITTEN {
            bytes.extend(std::iter::repeat_n(
                u8::try_from(s % 251).expect("byte"),
                ROW,
            ));
        }
        std::fs::write(d.join("big.rows"), &bytes).expect("stage");
        let written: Vec<i64> = (0..i64::try_from(WRITTEN).expect("slots")).collect();
        let staged = StagedStep {
            stage: "big.rows".into(),
            base_offset: 0,
            layer_stride: 0,
            head_stride: 0,
            layers: vec![0],
            heads: vec![0],
            table_stride: 1,
            slots: written.clone(),
            dtype: "bfloat16".into(),
            shape: vec![2, 4096],
        };
        src.commit("e", 0, 1, 7, Some(&staged), &[("r".into(), 9)])
            .expect("commit");
        let storage = Arc::new(Mutex::new(src));
        let telemetry = everserve_telemetry::Telemetry::new();
        let rep = Replicator::new(
            None,
            "store".into(),
            StoreMetrics::register(&telemetry),
            1 << 30,
            Arc::clone(&storage),
        );

        let (near, far) = tokio::io::duplex(1 << 20);
        let replica = fake_replica(far, dir.path().join("replica"), SLOTS);
        let (mut rd, mut wr) = tokio::io::split(near);
        let mut link = Link {
            engine: "e".into(),
            rank: 0,
            target: "replica".into(),
            lag: rep
                .metrics
                .replica_lag_steps
                .with_label_values(&["e", "0", "replica"]),
            pending: None,
            acked: None,
            served: false,
            reply_timeout: REPLY_TIMEOUT,
            node: "node-src".into(),
            idle_told: false,
            refused_at: None,
        };
        rep.resync(&mut rd, &mut wr, &mut link)
            .await
            .expect("resync");
        let (held, parts) = replica.await.expect("join");
        assert!(parts >= 2, "{parts} part(s) for a rank over a frame");
        assert_eq!(link.acked, Some((1, 7)));
        let src = storage.lock().await;
        assert_eq!(
            held.rows("e", 0, 0, &written),
            src.rows("e", 0, 0, &written)
        );
        assert_eq!(held.committed_of("e", 0), vec![("r".to_owned(), 9)]);
    }

    /// A replica on `far` that answers a resync like a store: it holds nothing, applies every
    /// part, checks each fits a frame, and ends with what it holds and how many parts it took.
    fn fake_replica(
        far: tokio::io::DuplexStream,
        dir: std::path::PathBuf,
        slots: usize,
    ) -> tokio::task::JoinHandle<(Storage, usize)> {
        use everserve_wire::proto::everserve::mirror::v1::{Done, ReplicaState};

        tokio::spawn(async move {
            let (mut rd, mut wr) = tokio::io::split(far);
            let mut held = Storage::new(dir, slots, 64 << 20);
            let mut parts = 0;
            loop {
                let req: MirrorRequest = wire::recv(&mut rd).await.expect("request");
                let (kind, last) = match req.kind {
                    Some(mirror_request::Kind::ReplicaState(_)) => (
                        mirror_reply::Kind::ReplicaState(ReplicaState::default()),
                        false,
                    ),
                    Some(mirror_request::Kind::Replicated(part)) => {
                        assert!(
                            framed(part.encoded_len()) <= wire::MAX_FRAME,
                            "a part over a frame"
                        );
                        parts += 1;
                        held.apply_replicated(&part).expect("apply");
                        (
                            mirror_reply::Kind::Done(Done {}),
                            !part.request_ids.is_empty(),
                        )
                    }
                    other => panic!("{other:?}"),
                };
                let reply = MirrorReply {
                    kind: Some(kind),
                    ..MirrorReply::default()
                };
                wire::send(&mut wr, &reply).await.expect("reply");
                if last {
                    return (held, parts);
                }
            }
        })
    }
}
