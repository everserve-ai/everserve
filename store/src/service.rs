// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The mirror service over the frame protocol: one `MirrorRequest` in, one `MirrorReply` out, in
//! order, per connection. The socket is the control plane; rows move through shared memory
//! (decision 0011). Every worker of an engine holds its own connection; the storage is shared
//! behind one lock because a commit is a few hundred row copies and the lock is held only for them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use everserve_wire::frame::{self, WireError};
use everserve_wire::proto::everserve::mirror::v1::{
    CommitAck, CommittedQuery, CommittedReply, Done, Evicted, Finished, MirrorReply, MirrorRequest,
    Open, Opened, Prepare, ReplicaState, ReplicaStateQuery, Replicated, RestoreReply,
    RestoreRequest, RowsQuery, RowsReply, SlotRow, mirror_reply, mirror_request,
};
use everserve_wire::transport::Peer;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cache::{CacheError, CompileCaches};

/// How long a listener waits after an accept that failed before it accepts again: a listener
/// that failed for want of a descriptor loops hot otherwise.
const ACCEPT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);

/// A request the storage answers under its lock: what reaches the second half of
/// [`Service::answer_on`] once the protocol check, the peer gate, the cache and a commit have
/// had theirs. Named apart so that half matches nothing it cannot be given.
enum Stored {
    Replicated(Replicated),
    Open(Open),
    Prepare(Prepare),
    Restore(RestoreRequest),
    Rows(RowsQuery),
    ReplicaState(ReplicaStateQuery),
    Committed(CommittedQuery),
    Evicted(Evicted),
    Finished(Finished),
}

impl Stored {
    /// `kind` as the storage answers it, or `kind` back when it is answered before the storage.
    fn of(kind: mirror_request::Kind) -> Result<Self, Box<mirror_request::Kind>> {
        Ok(match kind {
            mirror_request::Kind::Replicated(m) => Self::Replicated(m),
            mirror_request::Kind::Open(o) => Self::Open(o),
            mirror_request::Kind::Prepare(p) => Self::Prepare(p),
            mirror_request::Kind::Restore(r) => Self::Restore(r),
            mirror_request::Kind::Rows(q) => Self::Rows(q),
            mirror_request::Kind::ReplicaState(q) => Self::ReplicaState(q),
            mirror_request::Kind::Committed(q) => Self::Committed(q),
            mirror_request::Kind::Evicted(e) => Self::Evicted(e),
            mirror_request::Kind::Finished(f) => Self::Finished(f),
            other => return Err(Box::new(other)),
        })
    }
}
use crate::metrics::StoreMetrics;
use crate::replication::Replicator;
use crate::storage::Storage;

/// The service: shared storage plus the accept loop.
pub struct Service {
    storage: Arc<Mutex<Storage>>,
    metrics: StoreMetrics,
    replicator: Arc<Replicator>,
    /// Which node each replicated rank comes from, as the coordinator said through the daemon
    /// (decision 0026); a rank not here is taken from nobody.
    expected: Mutex<BTreeMap<(String, u32), String>>,
    /// The epoch each replicated rank is fenced at: a step of an older epoch is refused. Set by
    /// every expectation and never lowered, so a source whose daemon died with its worker and
    /// keeps draining an older epoch's backlog cannot write a replica the spare restores from.
    fenced: Mutex<BTreeMap<(String, u32), u64>>,
    /// The node's compile caches (decision 0043); none when the store keeps none.
    caches: Option<CompileCaches>,
}

/// One connection's standing as a replica sees it: who is at the other end, and the one rank
/// it may carry once its first request was accepted (decision 0026).
pub struct LinkState {
    peer: Peer,
    bound: Option<(String, u32)>,
    /// The connection is a reader on this node's worker socket, not a peer store's link: it may
    /// ask where the store stands without binding (decision 0030). A peer over a Unix socket is
    /// `Peer::Local` too, so the socket decides, not the peer.
    reader: bool,
    /// The ranks this connection opened, each with the staging prefix it was given: their
    /// staging files go when the connection ends.
    opened: Vec<(String, u32, String)>,
}

impl LinkState {
    /// A peer store's connection from `peer`, bound to nothing yet.
    #[must_use]
    pub fn from(peer: Peer) -> Self {
        Self {
            peer,
            bound: None,
            reader: false,
            opened: Vec::new(),
        }
    }

    /// A connection on this node's worker socket, bound to nothing: a worker's, or a reader's.
    #[must_use]
    pub fn local() -> Self {
        Self {
            peer: Peer::Local,
            bound: None,
            reader: true,
            opened: Vec::new(),
        }
    }
}

impl Service {
    /// A service over `storage`, reporting through `telemetry`.
    #[must_use]
    pub fn new(storage: Storage, telemetry: &everserve_telemetry::Telemetry) -> Arc<Self> {
        Self::with_tls(storage, telemetry, None, "store", None)
    }

    /// As [`Service::new`], dialling replica targets over TCP with `tls`, verifying
    /// `peer_name` on their certificates, and keeping the node's compile caches in `caches`.
    #[must_use]
    pub fn with_tls(
        storage: Storage,
        telemetry: &everserve_telemetry::Telemetry,
        tls: Option<everserve_wire::transport::Tls>,
        peer_name: &str,
        caches: Option<CompileCaches>,
    ) -> Arc<Self> {
        let metrics = StoreMetrics::register(telemetry);
        let queue_bytes = storage.budget();
        let storage = Arc::new(Mutex::new(storage));
        Arc::new(Self {
            replicator: Replicator::new(
                tls,
                peer_name.to_owned(),
                metrics.clone(),
                queue_bytes,
                Arc::clone(&storage),
            ),
            storage,
            metrics,
            expected: Mutex::new(BTreeMap::new()),
            fenced: Mutex::new(BTreeMap::new()),
            caches,
        })
    }

    /// Bound each replication target's queue at `bytes` instead of the table budget.
    pub fn set_replication_queue_bytes(&self, bytes: usize) {
        self.replicator.set_queue_bytes(bytes);
    }

    /// Wait `timeout` for a replica's reply instead of [`crate::replication::REPLY_TIMEOUT`].
    pub fn set_replication_reply_timeout(&self, timeout: std::time::Duration) {
        self.replicator.set_reply_timeout(timeout);
    }

    /// Serve peers (other stores replicating to this one) on `listener` until `cancel` fires.
    /// Only the replication protocol: a source's state query, rows and caught-up mark, and a
    /// compile cache's manifest and reads. A Unix listener on a node, TLS across nodes.
    ///
    /// # Errors
    ///
    /// The accept error.
    pub async fn serve_peers(
        self: Arc<Self>,
        listener: everserve_wire::transport::Listener,
        cancel: CancellationToken,
    ) -> std::io::Result<()> {
        info!(
            "mirror replication listening on {}",
            listener
                .local_address()
                .map_or_else(|_| "?".to_owned(), |a| a.to_string())
        );
        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                accepted = listener.accept() => {
                    // a source store claims no name here: its identity is the CA's word
                    let (stream, peer) = match accepted {
                        Ok(a) => (a.stream, a.peer),
                        Err(e) => {
                            warn!("replication accept: {e}; listening on");
                            tokio::time::sleep(ACCEPT_BACKOFF).await;
                            continue;
                        }
                    };
                    let me = Arc::clone(&self);
                    let conn_cancel = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            () = conn_cancel.cancelled() => {}
                            result = me.handle_stream(stream, LinkState::from(peer)) => {
                                if let Err(e) = result {
                                    warn!("replication connection ended: {e}");
                                }
                            }
                        }
                    });
                }
            }
        }
        Ok(())
    }

    /// Bind `path` and serve the node's processes there until `cancel` fires. The socket is the
    /// wire crate's listener: a path that exists is replaced, and the path is left in place on
    /// stop (the listener says why).
    ///
    /// # Errors
    ///
    /// The bind error, if the socket cannot be created.
    pub async fn serve(
        self: Arc<Self>,
        path: PathBuf,
        cancel: CancellationToken,
    ) -> std::io::Result<()> {
        let listener = everserve_wire::transport::Listener::bind(
            &everserve_wire::transport::Address::Unix(path.clone()),
            None,
        )
        .await
        .map_err(std::io::Error::other)?;
        info!("mirror listening on {}", path.display());
        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                accepted = listener.accept() => {
                    // an accept that fails (the process out of descriptors, for one) is the
                    // listener's to outlive: the node's mirror does not end on it
                    let stream = match accepted {
                        Ok(a) => a.stream,
                        Err(e) => {
                            warn!("mirror accept: {e}; listening on");
                            tokio::time::sleep(ACCEPT_BACKOFF).await;
                            continue;
                        }
                    };
                    let me = Arc::clone(&self);
                    let conn_cancel = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            () = conn_cancel.cancelled() => {}
                            result = me.handle(stream) => {
                                if let Err(e) = result {
                                    warn!("mirror connection ended: {e}");
                                }
                            }
                        }
                    });
                }
            }
        }
        // the socket's path stays: a store started again may have bound it already, and its next
        // bind replaces a leftover (wire's Listener says why)
        info!("mirror stopped listening");
        Ok(())
    }

    async fn handle(
        self: Arc<Self>,
        stream: everserve_wire::transport::Stream,
    ) -> Result<(), WireError> {
        self.handle_stream(stream, LinkState::local()).await
    }

    /// Serve one connection: a peer store's link (`LinkState::from`) or a connection on this
    /// node's worker socket (`LinkState::local`), which decides what may go unbound.
    async fn handle_stream(
        self: Arc<Self>,
        stream: everserve_wire::transport::Stream,
        mut link: LinkState,
    ) -> Result<(), WireError> {
        let result = self.exchange(stream, &mut link).await;
        self.end_link(&link).await;
        result
    }

    /// Remove the staging files of every rank `link` opened, whichever way it ended: a worker
    /// killed mid-step never deletes its own, and what each leaves stays until shared memory
    /// cannot hold the store's budget.
    async fn end_link(&self, link: &LinkState) {
        if link.opened.is_empty() {
            return;
        }
        let mut st = self.storage.lock().await;
        for (engine, rank, prefix) in &link.opened {
            let removed = st.end_link(engine, *rank, prefix);
            if removed > 0 {
                info!(
                    "{engine} rank {rank}: removed {removed} staging files of an ended connection"
                );
            }
        }
    }

    /// Open a rank for `link` and give the connection its staging prefix, kept on the link so
    /// the files named from it go when the connection ends.
    fn open(st: &mut Storage, o: &Open, link: &mut LinkState) -> Vec<MirrorReply> {
        match st.open(&o.engine, o.rank) {
            Ok(dir) => {
                let stage_prefix = st.stage_prefix();
                link.opened
                    .push((o.engine.clone(), o.rank, stage_prefix.clone()));
                vec![reply(mirror_reply::Kind::Opened(Opened {
                    rank_dir: dir.to_string_lossy().into_owned(),
                    stage_prefix,
                }))]
            }
            Err(e) => vec![error(&e.to_string())],
        }
    }

    /// Requests and replies on one connection until it ends.
    async fn exchange(
        &self,
        stream: everserve_wire::transport::Stream,
        link: &mut LinkState,
    ) -> Result<(), WireError> {
        let (mut rd, mut wr) = tokio::io::split(stream);
        loop {
            let req: MirrorRequest = match frame::recv(&mut rd).await {
                Ok(r) => r,
                Err(WireError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    let _ = wr.shutdown().await; // close_notify back: a clean end both ways
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            for reply in self.answer_on(req, link).await {
                frame::send(&mut wr, &reply).await?;
            }
        }
    }

    /// A restore: every table of the rank with its written windows since `r.since_step`, its
    /// counts and its step, with the lock wait (`waited`) and the work logged and observed.
    fn restore(
        &self,
        st: &Storage,
        r: &RestoreRequest,
        waited: std::time::Duration,
    ) -> Vec<MirrorReply> {
        let work = std::time::Instant::now();
        let (tables, step) = st.restore(&r.engine, r.rank, r.since_step);
        let (request_ids, committed_tokens): (Vec<String>, Vec<u64>) =
            st.committed_of(&r.engine, r.rank).into_iter().unzip();
        let rows: i64 = tables
            .iter()
            .flat_map(|t| t.written.iter().map(|w| w.count))
            .sum();
        self.metrics
            .restore_rows_total
            .inc_by(u64::try_from(rows).unwrap_or(u64::MAX));
        let work = work.elapsed();
        self.metrics
            .restore_seconds
            .with_label_values(&["lock"])
            .observe(waited.as_secs_f64());
        self.metrics
            .restore_seconds
            .with_label_values(&["work"])
            .observe(work.as_secs_f64());
        info!(
            "{} rank {}: restore since step {}: {rows} rows, lock {:.1} ms, work {:.1} ms",
            r.engine,
            r.rank,
            r.since_step,
            waited.as_secs_f64() * 1e3,
            work.as_secs_f64() * 1e3
        );
        vec![reply(mirror_reply::Kind::Restore(RestoreReply {
            tables,
            request_ids,
            committed_tokens,
            step,
        }))]
    }

    /// Where this store stands for a rank, and the epoch it fences the rank at.
    async fn replica_state(&self, st: &Storage, q: &ReplicaStateQuery) -> Vec<MirrorReply> {
        let (held, epoch, step, caught_up) = st.state(&q.engine, q.rank);
        // the fence is never held while the storage is awaited, so taking it here is safe
        let fence_epoch = self
            .fenced
            .lock()
            .await
            .get(&(q.engine.clone(), q.rank))
            .copied()
            .unwrap_or(0);
        vec![reply(mirror_reply::Kind::ReplicaState(ReplicaState {
            epoch,
            step,
            held,
            caught_up,
            fence_epoch,
        }))]
    }

    /// What a replica decides before the storage is touched (decision 0026): the control
    /// requests from its daemon (`Replicate`, `Expect`) are answered here; a source's first
    /// request binds the link or is refused; rows must arrive on a link bound to their rank.
    /// None when the request goes on to the storage.
    async fn gate(
        &self,
        kind: &mirror_request::Kind,
        link: &mut LinkState,
    ) -> Option<Vec<MirrorReply>> {
        match kind {
            mirror_request::Kind::Replicate(rep) => {
                self.replicator.set_node(&rep.node);
                Some(
                    match self
                        .replicator
                        .set_targets(&rep.engine, rep.rank, &rep.targets)
                        .await
                    {
                        Ok(()) => vec![reply(mirror_reply::Kind::Done(Done {}))],
                        Err(e) => vec![error(&e.to_string())],
                    },
                )
            }
            mirror_request::Kind::Expect(e) => {
                let key = (e.engine.clone(), e.rank);
                {
                    let mut fenced = self.fenced.lock().await;
                    let fence = fenced.entry(key.clone()).or_insert(0);
                    *fence = (*fence).max(e.epoch);
                }
                let mut expected = self.expected.lock().await;
                if e.from_node.is_empty() {
                    expected.remove(&key);
                } else {
                    expected.insert(key, e.from_node.clone());
                }
                Some(vec![reply(mirror_reply::Kind::Done(Done {}))])
            }
            // a reader on this node asks where the store stands (a spare's restore waiting for a
            // staged replication, decision 0030); a source's first request binds the link
            mirror_request::Kind::ReplicaState(_) if link.reader => None,
            mirror_request::Kind::ReplicaState(q) => {
                self.bind(link, q).await.err().map(|why| vec![error(&why)])
            }
            mirror_request::Kind::CaughtUp(c)
                if link.bound.as_ref() != Some(&(c.engine.clone(), c.rank)) =>
            {
                Some(vec![error(&format!(
                    "caught up for rank {} of {} on a link not bound to it",
                    c.rank, c.engine
                ))])
            }
            mirror_request::Kind::CaughtUp(c) => {
                let marked = self
                    .storage
                    .lock()
                    .await
                    .mark_caught_up(&c.engine, c.rank, c.epoch, c.step);
                tracing::debug!(
                    "rank {} of {}: source caught up at epoch {} step {}: {}",
                    c.rank,
                    c.engine,
                    c.epoch,
                    c.step,
                    if marked {
                        "marked"
                    } else {
                        "not where this store stands"
                    }
                );
                Some(vec![reply(mirror_reply::Kind::Done(Done {}))])
            }
            mirror_request::Kind::Replicated(msg)
                if link.bound.as_ref() != Some(&(msg.engine.clone(), msg.rank)) =>
            {
                Some(vec![error(&format!(
                    "rows for rank {} of {} on a link not bound to it",
                    msg.rank, msg.engine
                ))])
            }
            mirror_request::Kind::Replicated(msg) => {
                let fence = self
                    .fenced
                    .lock()
                    .await
                    .get(&(msg.engine.clone(), msg.rank))
                    .copied()
                    .unwrap_or(0);
                (msg.epoch < fence).then(|| {
                    vec![error(&format!(
                        "step {} at epoch {} of rank {} of {} is older than the rank's move at epoch {fence}",
                        msg.step, msg.epoch, msg.rank, msg.engine
                    ))]
                })
            }
            _ => None,
        }
    }

    /// A compile cache request (decision 0043), answered without the storage: a fetch only from
    /// this node's daemon on its socket, a manifest or a read from a peer store. None for any
    /// other request.
    async fn cache(&self, kind: &mirror_request::Kind) -> Option<Vec<MirrorReply>> {
        let answer = match kind {
            mirror_request::Kind::CacheFetch(f) => match self.caches.as_ref() {
                None => Err(CacheError::NoDirectory),
                Some(c) => c.fetch(f).await.map(|()| mirror_reply::Kind::Done(Done {})),
            },
            mirror_request::Kind::CacheManifest(q) => match self.caches.as_ref() {
                None => Err(CacheError::NoDirectory),
                Some(c) => c.manifest(q).map(mirror_reply::Kind::CacheManifest),
            },
            mirror_request::Kind::CacheRead(r) => match self.caches.as_ref() {
                None => Err(CacheError::NoDirectory),
                Some(c) => c.read(r).await.map(mirror_reply::Kind::CacheChunk),
            },
            _ => return None,
        };
        Some(vec![match answer {
            Ok(kind) => reply(kind),
            Err(e) => {
                warn!("compile cache: {e}");
                error(&e.to_string())
            }
        }])
    }

    /// What is answered before the storage's lock is taken for the rest: refusals of the protocol
    /// and of a peer's asking, the gate, the compile caches, and a commit, whose replication is
    /// queued after the lock. Any other kind back.
    async fn before_storage(
        &self,
        kind: mirror_request::Kind,
        link: &mut LinkState,
    ) -> Result<Vec<MirrorReply>, Stored> {
        if let Some(refused) = refused_protocol(&kind) {
            return Ok(refused);
        }
        if !link.reader && !peer_may_ask(&kind) {
            return Ok(vec![error(&format!(
                "{} is asked on the node's socket, not by a peer store",
                describe(&kind)
            ))]);
        }
        if let Some(replies) = self.gate(&kind, link).await {
            return Ok(replies);
        }
        if let Some(replies) = self.cache(&kind).await {
            return Ok(replies);
        }
        let c = match Stored::of(kind) {
            Ok(stored) => return Err(stored),
            Err(other) => match *other {
                mirror_request::Kind::Commit(c) => c,
                // a kind the gate or the cache answers, which neither did: answered as such,
                // never taken as something else
                other => {
                    return Ok(vec![error(&format!(
                        "{} is not answered here",
                        describe(&other)
                    ))]);
                }
            },
        };
        let (replies, to_replicate) = {
            let mut st = self.storage.lock().await;
            self.on_commit(&mut st, c).await
        };
        if let Some(msg) = to_replicate {
            self.replicator.queue(msg).await;
        }
        Ok(replies)
    }

    /// Bind `link` to the rank a source's first request names, or say why not: the source's
    /// certificate must be valid for the node it names, and the coordinator must have said the
    /// rank comes from that node (decision 0026).
    async fn bind(&self, link: &mut LinkState, q: &ReplicaStateQuery) -> Result<(), String> {
        let key = (q.engine.clone(), q.rank);
        if !link.peer.is_named(&q.node) {
            return Err(format!(
                "rank {} of {}: certificate not valid for node {:?}",
                q.rank, q.engine, q.node
            ));
        }
        match self.expected.lock().await.get(&key) {
            Some(from) if *from == q.node => {
                link.bound = Some(key);
                Ok(())
            }
            Some(from) => Err(format!(
                "rank {} of {} replicates here from {from}, not {}",
                q.rank, q.engine, q.node
            )),
            None => Err(format!(
                "rank {} of {} is not expected here from anyone",
                q.rank, q.engine
            )),
        }
    }

    /// Decision 0020: create and touch a rank's tables before it serves, and log how long it took.
    fn prepare(&self, st: &mut Storage, pr: &Prepare) -> Vec<MirrorReply> {
        let started = std::time::Instant::now();
        match st.prepare(
            &pr.engine, pr.rank, pr.slots, &pr.tables, &pr.dtype, &pr.shape,
        ) {
            Ok(()) => {
                // decision 0020: registration pays the tables' page faults; say how long
                info!(
                    engine = %pr.engine,
                    rank = pr.rank,
                    tables = pr.tables.len(),
                    ms = started.elapsed().as_secs_f64() * 1e3,
                    "prepared"
                );
                self.metrics
                    .bytes_in_use
                    .set(i64::try_from(st.in_use()).unwrap_or(i64::MAX));
                vec![reply(mirror_reply::Kind::Done(Done {}))]
            }
            Err(e) => {
                self.metrics
                    .refusals_total
                    .with_label_values(&[e.reason()])
                    .inc();
                vec![error(&e.to_string())]
            }
        }
    }

    /// Apply one commit and, for a replicated rank, queue what it wrote for the targets.
    async fn on_commit(
        &self,
        st: &mut Storage,
        c: everserve_wire::proto::everserve::mirror::v1::Commit,
    ) -> (Vec<MirrorReply>, Option<Replicated>) {
        let committed: Vec<(String, u64)> = c
            .request_ids
            .iter()
            .cloned()
            .zip(c.committed_tokens.iter().copied())
            .collect();
        match st.commit(
            &c.engine,
            c.rank,
            c.epoch,
            c.step,
            c.staged.as_ref(),
            &committed,
        ) {
            Ok(()) => {
                self.metrics.commits_total.inc();
                // what the replicas get, queued by the caller once the storage lock is gone:
                // waiting for room in a target's queue must not hold the store
                let to_replicate = if self.replicator.replicates(&c.engine, c.rank).await {
                    st.replicated(
                        &c.engine,
                        c.rank,
                        c.epoch,
                        c.step,
                        c.staged.as_ref(),
                        &committed,
                    )
                } else {
                    None
                };
                self.metrics
                    .bytes_in_use
                    .set(i64::try_from(st.in_use()).unwrap_or(i64::MAX));
                (
                    vec![reply(mirror_reply::Kind::CommitAck(CommitAck {
                        step: c.step,
                    }))],
                    to_replicate,
                )
            }
            Err(e) => {
                self.metrics
                    .refusals_total
                    .with_label_values(&[e.reason()])
                    .inc();
                (vec![error(&e.to_string())], None)
            }
        }
    }

    /// The replies to one request on a local, unbound connection: what a worker's socket is.
    /// Public so a test can drive the service without a socket.
    pub async fn answer(&self, req: MirrorRequest) -> Vec<MirrorReply> {
        self.answer_on(req, &mut LinkState::local()).await
    }

    /// The replies to one request on `link`. A replica's first request from a source
    /// (`ReplicaState`) binds the link to one rank when the source's certificate is valid for
    /// the node it names and the coordinator said the rank comes from that node; rows for any
    /// other rank, or on an unbound link, are refused (decision 0026). A peer store's link carries
    /// the replication protocol and nothing else.
    pub async fn answer_on(&self, req: MirrorRequest, link: &mut LinkState) -> Vec<MirrorReply> {
        let Some(kind) = req.kind else {
            return vec![error("empty request")];
        };
        tracing::debug!("request {}", describe(&kind));
        let stored = match self.before_storage(kind, link).await {
            Ok(replies) => return replies,
            Err(stored) => stored,
        };
        let asked = std::time::Instant::now();
        let mut st = self.storage.lock().await;
        let waited = asked.elapsed();
        match stored {
            Stored::Replicated(msg) => match st.apply_replicated(&msg) {
                Ok(()) => {
                    self.metrics.replica_applied_total.inc();
                    self.metrics
                        .bytes_in_use
                        .set(i64::try_from(st.in_use()).unwrap_or(i64::MAX));
                    vec![reply(mirror_reply::Kind::Done(Done {}))]
                }
                Err(e) => {
                    self.metrics
                        .refusals_total
                        .with_label_values(&[e.reason()])
                        .inc();
                    vec![error(&e.to_string())]
                }
            },
            Stored::Open(o) => Self::open(&mut st, &o, link),
            Stored::Prepare(pr) => self.prepare(&mut st, &pr),
            Stored::Restore(r) => self.restore(&st, &r, waited),
            Stored::Rows(q) => rows_reply(&st, &q),
            Stored::ReplicaState(q) => self.replica_state(&st, &q).await,
            Stored::Committed(q) => {
                let counts = st.committed(&q.engine, &q.request_ids);
                vec![reply(mirror_reply::Kind::Committed(CommittedReply {
                    request_ids: q.request_ids,
                    committed_tokens: counts,
                }))]
            }
            Stored::Evicted(e) => {
                st.evicted(&e.engine, &e.request_id, e.kept_tokens);
                vec![reply(mirror_reply::Kind::Done(Done {}))]
            }
            Stored::Finished(f) => {
                st.finished(&f.engine, &f.request_ids);
                vec![reply(mirror_reply::Kind::Done(Done {}))]
            }
        }
    }
}

/// The reply to a rows query: each slot's written step and bytes under the lock, or why not.
fn rows_reply(st: &Storage, q: &RowsQuery) -> Vec<MirrorReply> {
    match st.rows(&q.engine, q.rank, q.table, &q.slots) {
        Some(held) => vec![reply(mirror_reply::Kind::Rows(RowsReply {
            epoch: held.epoch,
            step: held.step,
            rows: held
                .rows
                .into_iter()
                .map(|(slot, written_step, data, crc32c)| SlotRow {
                    slot,
                    written_step,
                    data,
                    crc32c,
                })
                .collect(),
        }))],
        None => vec![error(&format!(
            "rank {} of {} has no table {} here",
            q.rank, q.engine, q.table
        ))],
    }
}

/// The refusal of a first request from another protocol (decision 0027), or None when the
/// request is not a first request or carries this build's: a worker's `Open`, a source store's
/// `ReplicaState`.
/// Whether a peer store's link may ask `kind`: what a source sends its replica (decision 0026),
/// and a compile cache's manifest and reads (decision 0043). Everything else is the node's own:
/// its workers' commits and restores, and its daemon's word on targets and expectations, which
/// a peer could otherwise use to choose whose rows this store takes. A peer over a Unix socket
/// is `Peer::Local` too, so the socket decides, not the peer.
fn peer_may_ask(kind: &mirror_request::Kind) -> bool {
    matches!(
        kind,
        mirror_request::Kind::ReplicaState(_)
            | mirror_request::Kind::Replicated(_)
            | mirror_request::Kind::CaughtUp(_)
            | mirror_request::Kind::CacheManifest(_)
            | mirror_request::Kind::CacheRead(_)
    )
}

fn refused_protocol(kind: &mirror_request::Kind) -> Option<Vec<MirrorReply>> {
    let (sent, oldest) = match kind {
        mirror_request::Kind::Open(o) => (o.protocol, o.protocol_oldest),
        mirror_request::Kind::ReplicaState(q) => (q.protocol, q.protocol_oldest),
        _ => return None,
    };
    everserve_wire::check_protocol(sent, oldest)
        .err()
        .map(|e| vec![error(&e.to_string())])
}

/// One line per request at debug level, so a stalled client can be matched against what the
/// store received: the kind, the engine and the rank (or the request's step).
fn describe(kind: &mirror_request::Kind) -> String {
    match kind {
        mirror_request::Kind::CacheFetch(f) => format!(
            "cache fetch {} rank {} digest {} from {:?}",
            f.key, f.rank, f.digest, f.holders
        ),
        mirror_request::Kind::CacheManifest(q) => {
            format!(
                "cache manifest {} rank {} digest {}",
                q.key, q.rank, q.digest
            )
        }
        mirror_request::Kind::CacheRead(r) => {
            format!(
                "cache read {} rank {} {} at {}",
                r.key, r.rank, r.path, r.offset
            )
        }
        mirror_request::Kind::CaughtUp(c) => format!(
            "caught up {} rank {} at epoch {} step {}",
            c.engine, c.rank, c.epoch, c.step
        ),
        mirror_request::Kind::Open(o) => format!("open {} rank {}", o.engine, o.rank),
        mirror_request::Kind::Prepare(p) => {
            format!(
                "prepare {} rank {} tables {}",
                p.engine,
                p.rank,
                p.tables.len()
            )
        }
        mirror_request::Kind::Commit(c) => format!(
            "commit {} rank {} epoch {} step {} rows {}",
            c.engine,
            c.rank,
            c.epoch,
            c.step,
            c.staged.as_ref().map_or(0, |s| s.slots.len())
        ),
        mirror_request::Kind::Restore(r) => {
            format!(
                "restore {} rank {} since {}",
                r.engine, r.rank, r.since_step
            )
        }
        mirror_request::Kind::ReplicaState(q) => {
            format!("replica state {} rank {}", q.engine, q.rank)
        }
        mirror_request::Kind::Committed(q) => {
            format!("committed {} ({})", q.engine, q.request_ids.len())
        }
        mirror_request::Kind::Evicted(e) => format!("evicted {} {}", e.engine, e.request_id),
        mirror_request::Kind::Finished(f) => {
            format!("finished {} ({})", f.engine, f.request_ids.len())
        }
        mirror_request::Kind::Expect(e) => format!(
            "expect {} rank {} from {:?} at epoch {}",
            e.engine, e.rank, e.from_node, e.epoch
        ),
        mirror_request::Kind::Rows(q) => format!(
            "rows {} rank {} table {} ({} slots)",
            q.engine,
            q.rank,
            q.table,
            q.slots.len()
        ),
        mirror_request::Kind::Replicate(r) => {
            format!("replicate {} rank {} to {:?}", r.engine, r.rank, r.targets)
        }
        mirror_request::Kind::Replicated(m) => format!(
            "replicated {} rank {} epoch {} step {} tables {}",
            m.engine,
            m.rank,
            m.epoch,
            m.step,
            m.tables.len()
        ),
    }
}

fn reply(kind: mirror_reply::Kind) -> MirrorReply {
    MirrorReply {
        kind: Some(kind),
        error: String::new(),
    }
}

fn error(msg: &str) -> MirrorReply {
    MirrorReply {
        kind: None,
        error: msg.to_owned(),
    }
}
