// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The mirror over a Unix socket with rows in shared memory: open, commit from a staging file,
//! idempotent overwrite, padding, restore by table reference, committed counts, eviction, finish,
//! stale epochs, bad shapes and names, the memory budget, and the metrics that count them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use everserve_store::service::Service;
use everserve_store::storage::{Storage, StorageError};
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::proto::everserve::mirror::v1::{
    Commit, CommittedQuery, Evicted, Finished, MirrorReply, MirrorRequest, Open, RestoreRequest,
    StagedStep, mirror_reply, mirror_request,
};
use tokio::net::UnixStream;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const ROW: usize = 16; // shape [2, 4] of bfloat16

/// Write one table's rows of `ROW` bytes, each filled with its value, as `name` in `rank_dir`;
/// return the step descriptor naming that one table (`layer`, head slot 0, table stride 1).
fn stage(rank_dir: &Path, name: &str, layer: u32, slots: &[i64], fills: &[u8]) -> StagedStep {
    assert_eq!(slots.len(), fills.len());
    let mut bytes = Vec::new();
    for &f in fills {
        bytes.extend(std::iter::repeat_n(f, ROW));
    }
    std::fs::write(rank_dir.join(name), bytes).expect("stage file");
    StagedStep {
        stage: name.to_owned(),
        base_offset: 0,
        layer_stride: 0,
        head_stride: 0,
        layers: vec![layer],
        heads: vec![0],
        table_stride: 1,
        slots: slots.to_vec(),
        dtype: "bfloat16".into(),
        shape: vec![2, 4],
    }
}

/// Two tables (layers 0 and 1, one head) in one file: layer 1's rows follow layer 0's.
fn stage_two(rank_dir: &Path, name: &str, slots: &[i64], fill0: u8, fill1: u8) -> StagedStep {
    let mut bytes = Vec::new();
    for f in [fill0, fill1] {
        for _ in slots {
            bytes.extend(std::iter::repeat_n(f, ROW));
        }
    }
    std::fs::write(rank_dir.join(name), bytes).expect("stage file");
    StagedStep {
        stage: name.to_owned(),
        base_offset: 0,
        layer_stride: (slots.len() * ROW) as u64,
        head_stride: 0,
        layers: vec![0, 1],
        heads: vec![0],
        table_stride: 1,
        slots: slots.to_vec(),
        dtype: "bfloat16".into(),
        shape: vec![2, 4],
    }
}

fn commit(epoch: u64, step: u64, staged: StagedStep, committed: &[(&str, u64)]) -> MirrorRequest {
    MirrorRequest {
        kind: Some(mirror_request::Kind::Commit(Commit {
            engine: "e".into(),
            rank: 2,
            epoch,
            step,
            staged: Some(staged),
            request_ids: committed.iter().map(|(id, _)| (*id).to_owned()).collect(),
            committed_tokens: committed.iter().map(|(_, n)| *n).collect(),
        })),
    }
}

fn query(ids: &[&str]) -> MirrorRequest {
    MirrorRequest {
        kind: Some(mirror_request::Kind::Committed(CommittedQuery {
            engine: "e".into(),
            request_ids: ids.iter().map(|s| (*s).to_owned()).collect(),
        })),
    }
}

async fn ask(s: &mut UnixStream, req: &MirrorRequest) -> MirrorReply {
    frame::send(s, req).await.expect("send");
    frame::recv(s).await.expect("recv")
}

struct Running {
    service: Arc<Service>,
    telemetry: Telemetry,
    sock: PathBuf,
    cancel: CancellationToken,
    server: JoinHandle<std::io::Result<()>>,
    client: UnixStream,
    rank_dir: PathBuf,
    dir: tempfile::TempDir,
}

async fn start() -> Running {
    start_with(None).await
}

/// As [`start`], with each replication target's queue bounded at `queue_bytes` when given.
async fn start_with(queue_bytes: Option<usize>) -> Running {
    let dir = tempfile::tempdir().expect("tmp");
    let sock = dir.path().join("m.sock");
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 64, 1 << 20),
        &telemetry,
    );
    if let Some(bytes) = queue_bytes {
        service.set_replication_queue_bytes(bytes);
    }
    let cancel = CancellationToken::new();
    let server = tokio::spawn(Arc::clone(&service).serve(sock.clone(), cancel.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let mut client = UnixStream::connect(&sock).await.expect("connect");
    let open = MirrorRequest {
        kind: Some(mirror_request::Kind::Open(Open {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "e".into(),
            rank: 2,
        })),
    };
    let opened = ask(&mut client, &open).await;
    let Some(mirror_reply::Kind::Opened(o)) = opened.kind else {
        panic!("{opened:?}")
    };
    let rank_dir = PathBuf::from(o.rank_dir);
    assert!(
        rank_dir.is_dir() && rank_dir.ends_with("e/rank2"),
        "{}",
        rank_dir.display()
    );
    Running {
        service: Arc::clone(&service),
        telemetry,
        sock,
        cancel,
        server,
        client,
        rank_dir,
        dir,
    }
}

async fn stop(r: Running) {
    r.cancel.cancel();
    r.server.await.expect("join").expect("serve");
    // stopped: nothing listens at the socket's path any more (the path itself stays, for the next
    // bind to replace)
    assert!(std::os::unix::net::UnixStream::connect(&r.sock).is_err());
}

fn table_rows(path: &str, slots: &[usize]) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).expect("table");
    slots
        .iter()
        .map(|&s| bytes[s * ROW..(s + 1) * ROW].to_vec())
        .collect()
}

#[tokio::test]
async fn commit_restore_and_request_lifecycle_over_the_socket() {
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;

    let ack = ask(
        s,
        &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)]),
    )
    .await;
    assert!(
        matches!(ack.kind, Some(mirror_reply::Kind::CommitAck(ref a)) if a.step == 10),
        "{ack:?}"
    );
    // idempotent overwrite of slot 9, a padding row that is skipped, and a new slot; a staging
    // file is sized once, so a step of another size is another file
    let ack = ask(
        s,
        &commit(
            1,
            11,
            stage(&d, "s2", 0, &[9, -1, 12], &[1, 7, 1]),
            &[("a", 41), ("b", 5)],
        ),
    )
    .await;
    assert!(ack.error.is_empty(), "{ack:?}");

    // restore names the tables and their written slots; the rows are read from the files
    let req = MirrorRequest {
        kind: Some(mirror_request::Kind::Restore(RestoreRequest {
            engine: "e".into(),
            rank: 2,
            since_step: 0,
        })),
    };
    let reply = ask(s, &req).await;
    let Some(mirror_reply::Kind::Restore(rr)) = reply.kind else {
        panic!("{reply:?}")
    };
    assert_eq!(rr.tables.len(), 2);
    assert_eq!(
        rr.step, 11,
        "the watermark is the rank's last committed step"
    );
    assert_eq!(rr.request_ids.len(), rr.committed_tokens.len());
    let t0 = &rr.tables[0];
    assert_eq!(t0.layer, 0);
    let written: Vec<(i64, i64)> = t0.written.iter().map(|w| (w.start, w.count)).collect();
    assert_eq!(written, vec![(3, 1), (9, 1), (12, 1)]);
    assert!(Path::new(&t0.path).starts_with(&d), "{}", t0.path);
    assert_eq!(
        std::fs::metadata(&t0.path).expect("table").len(),
        (64 * ROW) as u64
    );
    for row in table_rows(&t0.path, &[3, 9, 12]) {
        assert!(row.iter().all(|&x| x == 1), "{row:?}");
    }
    assert!(
        table_rows(&t0.path, &[0, 4])
            .iter()
            .all(|row| row.iter().all(|&x| x == 0))
    );
    let t1 = &rr.tables[1];
    assert_eq!(t1.layer, 1);
    assert_eq!(
        t1.written
            .iter()
            .map(|w| (w.start, w.count))
            .collect::<Vec<_>>(),
        vec![(3, 1), (9, 1)]
    );
    assert!(
        table_rows(&t1.path, &[3])
            .iter()
            .all(|row| row.iter().all(|&x| x == 2))
    );

    let text = r.telemetry.render();
    assert!(text.contains("everserve_store_commits_total 2"), "{text}");
    assert!(
        text.contains("everserve_store_restore_rows_total 5"),
        "{text}"
    );
    assert!(text.contains("everserve_store_bytes_in_use"), "{text}");
    stop(r).await;
}

#[tokio::test]
async fn an_incremental_restore_names_only_the_slots_written_after_the_watermark() {
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;
    assert!(
        ask(s, &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[]))
            .await
            .error
            .is_empty()
    );
    assert!(
        ask(
            s,
            &commit(1, 11, stage(&d, "s2", 0, &[9, 12], &[1, 1]), &[])
        )
        .await
        .error
        .is_empty()
    );
    // only the slots step 11 wrote, and the watermark to continue from
    let since = MirrorRequest {
        kind: Some(mirror_request::Kind::Restore(RestoreRequest {
            engine: "e".into(),
            rank: 2,
            since_step: 10,
        })),
    };
    let reply = ask(s, &since).await;
    let Some(mirror_reply::Kind::Restore(delta)) = reply.kind else {
        panic!("{reply:?}")
    };
    let w0: Vec<(i64, i64)> = delta.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(
        w0,
        vec![(9, 1), (12, 1)],
        "step 11 rewrote slot 9 and wrote slot 12"
    );
    assert!(
        delta.tables[1].written.is_empty(),
        "layer 1 was not written after step 10"
    );
    assert_eq!(delta.step, 11);
    stop(r).await;
}

#[tokio::test]
async fn counts_are_queried_lowered_by_eviction_and_forgotten_on_finish() {
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;
    let ack = ask(
        s,
        &commit(
            1,
            10,
            stage(&d, "s0", 0, &[3], &[1]),
            &[("a", 41), ("b", 5)],
        ),
    )
    .await;
    assert!(ack.error.is_empty(), "{ack:?}");
    let reply = ask(s, &query(&["a", "b", "zz"])).await;
    let Some(mirror_reply::Kind::Committed(c)) = reply.kind else {
        panic!("{reply:?}")
    };
    assert_eq!(c.committed_tokens, vec![41, 5, 0]);
    let evicted = MirrorRequest {
        kind: Some(mirror_request::Kind::Evicted(Evicted {
            engine: "e".into(),
            request_id: "a".into(),
            kept_tokens: 16,
        })),
    };
    assert!(matches!(
        ask(s, &evicted).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
    let finished = MirrorRequest {
        kind: Some(mirror_request::Kind::Finished(Finished {
            engine: "e".into(),
            request_ids: vec!["b".into()],
        })),
    };
    assert!(matches!(
        ask(s, &finished).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
    let reply = ask(s, &query(&["a", "b"])).await;
    let Some(mirror_reply::Kind::Committed(c)) = reply.kind else {
        panic!("{reply:?}")
    };
    assert_eq!(c.committed_tokens, vec![16, 0]);
    stop(r).await;
}

#[tokio::test]
async fn refusals_are_errors_not_silence_and_are_counted() {
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;
    assert!(
        ask(s, &commit(1, 10, stage(&d, "s0", 0, &[3], &[1]), &[]))
            .await
            .error
            .is_empty()
    );
    let reply = ask(s, &commit(0, 12, stage(&d, "s0", 0, &[1], &[1]), &[])).await;
    assert!(reply.error.contains("stale epoch"), "{reply:?}");
    let mut bad = stage(&d, "s0", 0, &[1], &[1]);
    bad.shape = vec![2, 8];
    let reply = ask(s, &commit(1, 12, bad, &[])).await;
    assert!(reply.error.contains("differs"), "{reply:?}");
    let mut escaped = stage(&d, "s0", 0, &[1], &[1]);
    escaped.stage = "../s0".into();
    let reply = ask(s, &commit(1, 12, escaped, &[])).await;
    assert!(reply.error.contains("not a file name"), "{reply:?}");
    let mut short = stage(&d, "s0", 0, &[1], &[1]);
    short.slots = vec![1, 2, 3]; // three rows named, one staged
    let reply = ask(s, &commit(1, 12, short, &[])).await;
    assert!(reply.error.contains("needed"), "{reply:?}");
    let text = r.telemetry.render();
    for reason in ["stale_epoch", "shape", "stage_name", "size"] {
        assert!(
            text.contains(&format!(
                "everserve_store_refusals_total{{reason=\"{reason}\"}} 1"
            )),
            "{reason}: {text}"
        );
    }
    assert!(text.contains("everserve_store_commits_total 1"), "{text}");
    stop(r).await;
}

#[test]
fn budget_slots_and_sizes_are_enforced_before_any_write() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, ROW * 8); // exactly one layer
    let d = st.open("e", 0).expect("open");
    let one = |name: &str, layer, slots: &[i64], fill| {
        let fills = vec![fill; slots.len()];
        stage(&d, name, layer, slots, &fills)
    };
    st.commit("e", 0, 1, 1, Some(&one("a", 0, &[1], 7)), &[])
        .expect("first layer fits");
    assert_eq!(st.in_use(), ROW * 8);
    let err = st
        .commit("e", 0, 1, 2, Some(&one("b", 1, &[1], 7)), &[])
        .unwrap_err();
    assert!(matches!(err, StorageError::Budget { .. }), "{err}");
    assert_eq!(
        st.restore("e", 0, 0).0.len(),
        1,
        "the refused commit wrote nothing"
    );
    let err = st
        .commit("e", 0, 1, 3, Some(&one("c", 0, &[8], 7)), &[])
        .unwrap_err();
    assert!(matches!(err, StorageError::Slot { slot: 8, .. }), "{err}");
    let mut odd = one("d", 0, &[2], 7);
    odd.dtype = "complex128".into();
    let err = st.commit("e", 0, 1, 3, Some(&odd), &[]).unwrap_err();
    assert!(matches!(err, StorageError::Dtype { .. }), "{err}");
    // a descriptor naming a bad slot in any position writes nothing
    let err = st
        .commit("e", 0, 1, 3, Some(&one("e", 0, &[2, 9], 7)), &[])
        .unwrap_err();
    assert!(matches!(err, StorageError::Slot { slot: 9, .. }), "{err}");
    let tables = st.layers("e", 0);
    assert_eq!(
        tables[0].1.written_slots(),
        vec![1],
        "slot 2 was not written"
    );
    assert!(tables[0].1.row(1).iter().all(|&x| x == 7));
}

#[test]
fn each_rank_keeps_the_counts_its_own_rows_are_complete_through() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let d1 = st.open("e", 1).expect("open");
    let d0 = st.open("e", 0).expect("open");
    let a = |n| [("a".to_owned(), n)];
    // rank 1 commits step 10 and stops; rank 0 goes on to step 11
    st.commit(
        "e",
        1,
        0,
        10,
        Some(&stage(&d1, "s", 0, &[1], &[1])),
        &a(2058),
    )
    .expect("commit");
    st.commit(
        "e",
        0,
        0,
        10,
        Some(&stage(&d0, "s", 0, &[1], &[1])),
        &a(2058),
    )
    .expect("commit");
    st.commit(
        "e",
        0,
        0,
        11,
        Some(&stage(&d0, "s", 0, &[2], &[1])),
        &a(2059),
    )
    .expect("commit");
    assert_eq!(st.committed("e", &["a".to_owned()]), vec![2059]);
    assert_eq!(st.committed_of("e", 1), vec![("a".to_owned(), 2058)]);
    assert_eq!(st.committed_of("e", 0), vec![("a".to_owned(), 2059)]);
    // eviction lowers, finish forgets, on every rank
    st.evicted("e", "a", 16);
    assert_eq!(st.committed_of("e", 0), vec![("a".to_owned(), 16)]);
    assert_eq!(st.committed_of("e", 1), vec![("a".to_owned(), 16)]);
    st.finished("e", &["a".to_owned()]);
    assert!(st.committed_of("e", 0).is_empty() && st.committed_of("e", 1).is_empty());
}

#[test]
fn reopening_a_rank_forgets_its_earlier_staging_files() {
    // a replaced worker creates new staging files; the store must not keep serving a stale map
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let d = st.open("e", 0).expect("open");
    st.commit("e", 0, 1, 1, Some(&stage(&d, "s", 0, &[1], &[1])), &[])
        .expect("commit");
    std::fs::remove_file(d.join("s")).expect("rm");
    let d = st.open("e", 0).expect("reopen");
    let err = st
        .commit(
            "e",
            0,
            2,
            2,
            Some(&StagedStep {
                stage: "s".into(),
                base_offset: 0,
                layer_stride: 0,
                head_stride: 0,
                layers: vec![0],
                heads: vec![0],
                table_stride: 1,
                slots: vec![2],
                dtype: "bfloat16".into(),
                shape: vec![2, 4],
            }),
            &[],
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Io { .. }), "{err}");
    st.commit("e", 0, 2, 2, Some(&stage(&d, "s", 0, &[2], &[3])), &[])
        .expect("the new file is mapped");
    assert!(st.layers("e", 0)[0].1.row(2).iter().all(|&x| x == 3));
}

#[test]
fn a_store_holding_fewer_slots_than_its_engine_writes_refuses_the_prepare() {
    // a store of 4,096 slot ids under an engine writing millions dies at its first busy step;
    // the prepare says so, before it serves
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 4096, ROW * 4096 * 2);
    st.open("e", 0).expect("open");
    let err = st
        .prepare("e", 0, 6_014_144, &[0], "bfloat16", &[2, 4])
        .unwrap_err();
    assert!(
        matches!(
            err,
            StorageError::Undersized {
                wants: 6_014_144,
                holds: 4096,
                ..
            }
        ),
        "{err}"
    );
    assert!(err.to_string().contains("node.store.slots"), "{err}");
    assert_eq!(err.reason(), "undersized");
    assert_eq!(st.in_use(), 0, "nothing made");
    // an engine that states no count is refused too; one that fits is prepared
    assert!(st.prepare("e", 0, 0, &[0], "bfloat16", &[2, 4]).is_err());
    st.prepare("e", 0, 4096, &[0], "bfloat16", &[2, 4])
        .expect("fits");
}

#[test]
fn prepared_tables_exist_touched_and_within_the_budget_before_the_first_commit() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 64, ROW * 64 * 3); // room for three tables
    let d = st.open("e", 0).expect("open");
    st.prepare("e", 0, 64, &[0, 1], "bfloat16", &[2, 4])
        .expect("prepare");
    assert_eq!(st.in_use(), ROW * 64 * 2);
    let tables = st.layers("e", 0);
    assert_eq!(
        tables.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(tables.iter().all(|(_, l)| l.written_slots().is_empty()));
    assert_eq!(
        std::fs::metadata(d.join("table0.rows"))
            .expect("file")
            .len(),
        (ROW * 64) as u64
    );
    // preparing again is idempotent; a different shape for a known table is refused
    st.prepare("e", 0, 64, &[0, 1], "bfloat16", &[2, 4])
        .expect("again");
    assert_eq!(st.in_use(), ROW * 64 * 2);
    let err = st
        .prepare("e", 0, 64, &[1], "bfloat16", &[2, 8])
        .unwrap_err();
    assert!(matches!(err, StorageError::Shape { layer: 1, .. }), "{err}");
    // a commit into a prepared table adds nothing to the budget; one more table would exceed it
    st.commit(
        "e",
        0,
        1,
        1,
        Some(&stage(&d, "s", 0, &[3, 4, 5, 9], &[1, 1, 1, 1])),
        &[],
    )
    .expect("commit");
    assert_eq!(st.in_use(), ROW * 64 * 2);
    let err = st
        .prepare("e", 0, 64, &[2, 3], "bfloat16", &[2, 4])
        .unwrap_err();
    assert!(matches!(err, StorageError::Budget { .. }), "{err}");
    // consecutive slots came over as one run and the odd one on its own
    let l0 = &st.layers("e", 0)[0].1;
    assert_eq!(l0.written_slots(), vec![3, 4, 5, 9]);
    assert!(l0.row(4).iter().all(|&x| x == 1) && l0.row(9).iter().all(|&x| x == 1));
}

/// A second store serving peers on a Unix listener, for replication tests.
struct Replica {
    telemetry: Telemetry,
    peer: PathBuf,
    cancel: CancellationToken,
    dir: tempfile::TempDir,
}

async fn replica() -> Replica {
    let dir = tempfile::tempdir().expect("tmp");
    let peer = dir.path().join("peer.sock");
    replica_in(dir, peer).await
}

/// A replica listening at `peer`, fresh, as a restarted store on the same address.
async fn replica_at(peer: &std::path::Path) -> Replica {
    replica_in(tempfile::tempdir().expect("tmp"), peer.to_path_buf()).await
}

async fn replica_in(dir: tempfile::TempDir, peer: PathBuf) -> Replica {
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 64, 1 << 20),
        &telemetry,
    );
    let listener = everserve_wire::transport::Listener::bind(
        &everserve_wire::transport::Address::Unix(peer.clone()),
        None,
    )
    .await
    .expect("bind");
    let cancel = CancellationToken::new();
    tokio::spawn(Arc::clone(&service).serve_peers(listener, cancel.clone()));
    // the replica's own worker socket, so a restore can be read from it like any store
    let sock = dir.path().join("m.sock");
    tokio::spawn(service.serve(sock, cancel.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    Replica {
        telemetry,
        peer,
        cancel,
        dir,
    }
}

async fn wait_metric(t: &Telemetry, line: &str) {
    for _ in 0..300 {
        if t.render().contains(line) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("metric line {line:?} never appeared:\n{}", t.render());
}

/// A source store replicating rank 2 to a replica, with two steps committed and applied.
async fn replicated_pair() -> (Replica, Running) {
    let rep = replica().await;
    expect(&rep, "node-src").await;
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;
    let replicate = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicate(
            everserve_wire::proto::everserve::mirror::v1::Replicate {
                node: "node-src".into(),
                engine: "e".into(),
                rank: 2,
                targets: vec![format!("unix:{}", rep.peer.display())],
            },
        )),
    };
    assert!(matches!(
        ask(s, &replicate).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
    assert!(
        ask(
            s,
            &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)])
        )
        .await
        .error
        .is_empty()
    );
    assert!(
        ask(
            s,
            &commit(
                1,
                11,
                stage(&d, "s2", 0, &[9, -1, 12], &[5, 7, 5]),
                &[("a", 41)]
            )
        )
        .await
        .error
        .is_empty()
    );
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 2").await;
    wait_metric(
        &r.telemetry,
        "everserve_store_replicated_total{engine=\"e\",target=",
    )
    .await;
    (rep, r)
}

/// Tell the replica, as its daemon would, that rank 2 of "e" comes from `from` (decision 0026).
async fn expect(rep: &Replica, from: &str) {
    expect_at(rep, from, 0).await;
}

/// [`expect`] with the rank fenced at `epoch`, as after a move to `from`.
async fn expect_at(rep: &Replica, from: &str, epoch: u64) {
    let mut c = UnixStream::connect(rep.dir.path().join("m.sock"))
        .await
        .expect("replica");
    let req = MirrorRequest {
        kind: Some(mirror_request::Kind::Expect(
            everserve_wire::proto::everserve::mirror::v1::Expect {
                engine: "e".into(),
                rank: 2,
                from_node: from.into(),
                epoch,
            },
        )),
    };
    assert!(matches!(
        ask(&mut c, &req).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
}

async fn restore_from(
    rep: &Replica,
    since_step: u64,
) -> everserve_wire::proto::everserve::mirror::v1::RestoreReply {
    let mut c = UnixStream::connect(rep.dir.path().join("m.sock"))
        .await
        .expect("replica");
    let restore = MirrorRequest {
        kind: Some(mirror_request::Kind::Restore(RestoreRequest {
            engine: "e".into(),
            rank: 2,
            since_step,
        })),
    };
    let reply = ask(&mut c, &restore).await;
    let Some(mirror_reply::Kind::Restore(rr)) = reply.kind else {
        panic!("{reply:?}")
    };
    rr
}

#[tokio::test]
async fn a_replicated_rank_restores_from_the_replica_with_the_same_rows_and_counts() {
    let (rep, r) = replicated_pair().await;
    let rr = restore_from(&rep, 0).await;
    assert_eq!(rr.step, 11);
    assert_eq!(rr.request_ids, vec!["a".to_owned()]);
    assert_eq!(rr.committed_tokens, vec![41]);
    assert_eq!(rr.tables.len(), 2);
    let w0: Vec<(i64, i64)> = rr.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(w0, vec![(3, 1), (9, 1), (12, 1)]);
    assert!(
        table_rows(&rr.tables[0].path, &[9, 12])
            .iter()
            .all(|row| row.iter().all(|&x| x == 5)),
        "step 11 rewrote slot 9 and wrote 12 on the replica"
    );
    assert!(
        table_rows(&rr.tables[0].path, &[3])
            .iter()
            .all(|row| row.iter().all(|&x| x == 1))
    );
    assert!(
        table_rows(&rr.tables[1].path, &[3])
            .iter()
            .all(|row| row.iter().all(|&x| x == 2))
    );
    assert!(
        rr.tables[0]
            .path
            .starts_with(rep.dir.path().to_str().expect("utf8")),
        "the replica's own file"
    );
    rep.cancel.cancel();
    stop(r).await;
}

#[tokio::test]
async fn an_incremental_restore_from_the_replica_continues_from_the_sources_watermark() {
    let (rep, r) = replicated_pair().await;
    let delta = restore_from(&rep, 10).await;
    let d0: Vec<(i64, i64)> = delta.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(d0, vec![(9, 1), (12, 1)]);
    assert!(delta.tables[1].written.is_empty());
    assert_eq!(delta.step, 11);
    let text = r.telemetry.render();
    assert!(
        text.contains("everserve_store_replica_lag_steps{engine=\"e\",rank=\"2\",target="),
        "{text}"
    );
    rep.cancel.cancel();
    stop(r).await;
}

#[test]
fn a_replicated_step_older_than_the_ranks_last_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let msg = |step: u64, fill: u8| everserve_wire::proto::everserve::mirror::v1::Replicated {
        engine: "e".into(),
        rank: 1,
        epoch: 3,
        step,
        tables: vec![
            everserve_wire::proto::everserve::mirror::v1::ReplicatedTable {
                table: 0,
                dtype: "bfloat16".into(),
                shape: vec![2, 4],
                slots: vec![2],
                data: vec![fill; ROW],
                crc32c: vec![crc32c::crc32c(&[fill; ROW])],
            },
        ],
        request_ids: vec!["q".into()],
        committed_tokens: vec![step],
    };
    st.apply_replicated(&msg(5, 1)).expect("first");
    let err = st.apply_replicated(&msg(4, 9)).unwrap_err();
    assert!(matches!(err, StorageError::StaleEpoch { .. }), "{err}");
    assert!(
        st.layers("e", 1)[0].1.row(2).iter().all(|&x| x == 1),
        "the stale step wrote nothing"
    );
    st.apply_replicated(&msg(6, 2)).expect("newer");
    assert_eq!(st.committed_of("e", 1), vec![("q".to_owned(), 6)]);
    assert_eq!(st.restore("e", 1, 5).0[0].written.len(), 1);
}

/// A peer that accepts replication links and reads every frame but never answers one.
fn silent_peer(dir: &std::path::Path) -> PathBuf {
    let peer = dir.join("silent.sock");
    let listener = tokio::net::UnixListener::bind(&peer).expect("bind");
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut sink = vec![0u8; 1 << 16];
                while tokio::io::AsyncReadExt::read(&mut stream, &mut sink)
                    .await
                    .is_ok_and(|n| n > 0)
                {}
            });
        }
    });
    peer
}

#[tokio::test]
async fn a_replica_that_cannot_keep_up_holds_the_commit_and_no_step_is_dropped() {
    // a queue that drops its oldest step once 64 behind leaves the replica's counts past rows it
    // never got: the queue is bytes, and a commit with no room waits for the replica
    let mut r = start_with(Some(1024)).await; // one step's worth of room
    let d = r.rank_dir.clone();
    let peer = silent_peer(r.dir.path());
    let s = &mut r.client;
    let replicate = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicate(
            everserve_wire::proto::everserve::mirror::v1::Replicate {
                node: "node-src".into(),
                engine: "e".into(),
                rank: 2,
                targets: vec![format!("unix:{}", peer.display())],
            },
        )),
    };
    assert!(matches!(
        ask(s, &replicate).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
    // the first step takes the room and sits unacknowledged at the silent peer
    assert!(
        ask(
            s,
            &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)])
        )
        .await
        .error
        .is_empty()
    );
    // the second waits for room: no acknowledgement, no drop
    let second = commit(
        1,
        11,
        stage(&d, "s2", 0, &[9, -1, 12], &[5, 7, 5]),
        &[("a", 41)],
    );
    everserve_wire::frame::send(s, &second).await.expect("send");
    let held = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        everserve_wire::frame::recv::<_, MirrorReply>(s),
    )
    .await;
    assert!(
        held.is_err(),
        "the commit was acknowledged with no room: {held:?}"
    );
    assert!(
        !r.telemetry
            .render()
            .contains("everserve_store_replication_dropped_total{"),
        "a step was dropped:\n{}",
        r.telemetry.render()
    );
    // the target is removed (as the coordinator does for a lost node): the room returns and
    // the commit goes through
    let mut other = UnixStream::connect(&r.sock).await.expect("connect");
    let clear = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicate(
            everserve_wire::proto::everserve::mirror::v1::Replicate {
                node: "node-src".into(),
                engine: "e".into(),
                rank: 2,
                targets: vec![],
            },
        )),
    };
    assert!(matches!(
        ask(&mut other, &clear).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
    let ack = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        everserve_wire::frame::recv::<_, MirrorReply>(s),
    )
    .await
    .expect("acknowledged once the target was removed")
    .expect("reply");
    assert!(ack.error.is_empty(), "{ack:?}");
    wait_metric(
        &r.telemetry,
        "everserve_store_replication_wait_seconds_total{engine=\"e\",target=",
    )
    .await;
}

/// Two steps committed for rank 2 with no replica yet: the rows a late target has to be given.
async fn committed_alone() -> Running {
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let s = &mut r.client;
    assert!(
        ask(
            s,
            &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)])
        )
        .await
        .error
        .is_empty()
    );
    assert!(
        ask(
            s,
            &commit(
                1,
                11,
                stage(&d, "s2", 0, &[9, -1, 12], &[5, 7, 5]),
                &[("a", 41)]
            )
        )
        .await
        .error
        .is_empty()
    );
    r
}

async fn replicate_to(r: &mut Running, peer: &std::path::Path) {
    let replicate = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicate(
            everserve_wire::proto::everserve::mirror::v1::Replicate {
                node: "node-src".into(),
                engine: "e".into(),
                rank: 2,
                targets: vec![format!("unix:{}", peer.display())],
            },
        )),
    };
    assert!(matches!(
        ask(&mut r.client, &replicate).await.kind,
        Some(mirror_reply::Kind::Done(_))
    ));
}

#[tokio::test]
async fn a_replica_reads_caught_up_once_the_sources_queue_is_idle() {
    // decision 0030: a spare's restore waits on its own store's state; the carrier tells the
    // replica when it has nothing more queued, and a local reader asks without binding
    let (rep, r) = replicated_pair().await;
    let mut local = UnixStream::connect(rep.dir.path().join("m.sock"))
        .await
        .expect("replica's socket");
    let ask_state = MirrorRequest {
        kind: Some(mirror_request::Kind::ReplicaState(
            everserve_wire::proto::everserve::mirror::v1::ReplicaStateQuery {
                engine: "e".into(),
                rank: 2,
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                node: String::new(),
            },
        )),
    };
    let mut state = None;
    for _ in 0..200 {
        if let Some(mirror_reply::Kind::ReplicaState(s)) = ask(&mut local, &ask_state).await.kind
            && s.held
            && s.caught_up
        {
            state = Some(s);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let s = state.expect("the replica never read caught up");
    assert_eq!((s.epoch, s.step), (1, 11));
    // a caught-up mark on a link not bound to the rank is refused
    let stray = MirrorRequest {
        kind: Some(mirror_request::Kind::CaughtUp(
            everserve_wire::proto::everserve::mirror::v1::CaughtUp {
                engine: "e".into(),
                rank: 2,
                epoch: 1,
                step: 11,
            },
        )),
    };
    let refused = ask(&mut local, &stray).await;
    assert!(refused.error.contains("not bound"), "{refused:?}");
    stop(r).await;
    rep.cancel.cancel();
}

#[tokio::test]
async fn a_target_named_after_the_commits_is_given_the_rank_as_held() {
    // the spare node links after the engine has been serving: without this the replica lacks
    // every row from before it was named, and a restore from it is a hole
    let mut r = committed_alone().await;
    let rep = replica().await;
    expect(&rep, "node-src").await;
    replicate_to(&mut r, &rep.peer).await;
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 1").await;
    let rr = restore_from(&rep, 0).await;
    assert_eq!(
        (rr.step, rr.request_ids.clone(), rr.committed_tokens.clone()),
        (11, vec!["a".to_owned()], vec![41])
    );
    let w0: Vec<(i64, i64)> = rr.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(w0, vec![(3, 1), (9, 1), (12, 1)]);
    assert!(
        table_rows(&rr.tables[0].path, &[9, 12])
            .iter()
            .all(|row| row.iter().all(|&x| x == 5))
    );
    assert!(
        table_rows(&rr.tables[0].path, &[3])
            .iter()
            .all(|row| row.iter().all(|&x| x == 1))
    );
    rep.cancel.cancel();
    stop(r).await;
}

#[tokio::test]
async fn a_rank_refused_as_held_under_its_move_fence_is_sent_again_once_its_holder_moves_on() {
    // a spare takes rank 2 at epoch 2 holding rows restored at epoch 1, and its sibling, fenced
    // at epoch 2, refuses them; a carrier that counted the replica covered would never send them
    // again. The refusal leaves it uncovered and the first commit at the new epoch brings every
    // row
    let mut r = committed_alone().await; // rows at epoch 1, steps 10 and 11
    let rep = replica().await;
    expect_at(&rep, "node-src", 2).await;
    replicate_to(&mut r, &rep.peer).await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(
        restore_from(&rep, 0).await.tables.len(),
        0,
        "the replica took rows older than its fence"
    );
    // the carrier read the fence and held the snapshot back, rather than send rows the replica
    // refuses after the whole transfer
    let text = r.telemetry.render();
    assert!(
        text.contains("everserve_store_snapshots_total{result=\"held\"} 1"),
        "{text}"
    );
    assert!(!text.contains("result=\"refused\""), "{text}");
    let d = r.rank_dir.clone();
    assert!(
        ask(
            &mut r.client,
            &commit(2, 12, stage(&d, "s3", 0, &[20], &[8]), &[("a", 42)])
        )
        .await
        .error
        .is_empty()
    );
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 1").await;
    wait_metric(
        &r.telemetry,
        "everserve_store_snapshots_total{result=\"sent\"} 1",
    )
    .await;
    let rr = restore_from(&rep, 0).await;
    assert_eq!(rr.step, 12);
    let w0: Vec<(i64, i64)> = rr.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(
        w0,
        vec![(3, 1), (9, 1), (12, 1), (20, 1)],
        "every row the holder had, the refused ones included"
    );
    rep.cancel.cancel();
    stop(r).await;
}

#[test]
fn a_replicated_row_that_does_not_match_its_checksum_refuses_the_step() {
    // decision 0030: a row damaged anywhere between the source's write and this apply is caught
    // here; nothing of the step is written, and a step with no checksums is refused the same way
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let good = vec![7u8; ROW];
    let table = |data: Vec<u8>, crc32c: Vec<u32>| {
        everserve_wire::proto::everserve::mirror::v1::ReplicatedTable {
            table: 0,
            dtype: "bfloat16".into(),
            shape: vec![2, 4],
            slots: vec![2],
            data,
            crc32c,
        }
    };
    let step = |t| everserve_wire::proto::everserve::mirror::v1::Replicated {
        engine: "e".into(),
        rank: 1,
        epoch: 3,
        step: 5,
        tables: vec![t],
        request_ids: vec!["q".into()],
        committed_tokens: vec![5],
    };
    let mut damaged = good.clone();
    damaged[3] ^= 0x10;
    let err = st
        .apply_replicated(&step(table(damaged, vec![crc32c::crc32c(&good)])))
        .unwrap_err();
    assert!(
        matches!(err, StorageError::Checksum { slot: 2, .. }),
        "{err}"
    );
    let err = st
        .apply_replicated(&step(table(good.clone(), vec![])))
        .unwrap_err();
    assert!(matches!(err, StorageError::Unchecked { .. }), "{err}");
    assert!(!st.state("e", 1).0, "a refused step wrote nothing");
    st.apply_replicated(&step(table(good.clone(), vec![crc32c::crc32c(&good)])))
        .expect("a matching row is taken");
    let held = st.rows("e", 1, 0, &[2]).expect("rows");
    assert_eq!(
        held.rows[0].3,
        crc32c::crc32c(&good),
        "the checksum is kept with the row"
    );
}

#[tokio::test]
async fn a_refused_step_is_healed_by_the_rank_as_held_once_the_holder_moves_on() {
    // decision 0030: a step the replica refuses leaves it holding none of that step's rows; the
    // carrier counts it uncovered and sends the rank as held before its next step
    let mut r = committed_alone().await; // epoch 1, steps 10 and 11, slots 3, 9, 12
    let rep = replica().await;
    expect(&rep, "node-src").await;
    replicate_to(&mut r, &rep.peer).await;
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 1").await;
    // the replica is fenced at epoch 2 (a move): the next step at epoch 1 is refused
    expect_at(&rep, "node-src", 2).await;
    let d = r.rank_dir.clone();
    assert!(
        ask(
            &mut r.client,
            &commit(1, 12, stage(&d, "s3", 0, &[20], &[8]), &[("a", 42)])
        )
        .await
        .error
        .is_empty()
    );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(!slots_of(&restore_from(&rep, 0).await).contains(&20));
    // the holder's next commit, at the move's epoch: the rank as held reaches the replica first
    assert!(
        ask(
            &mut r.client,
            &commit(2, 13, stage(&d, "s4", 0, &[30], &[4]), &[("a", 43)])
        )
        .await
        .error
        .is_empty()
    );
    for _ in 0..200 {
        if slots_of(&restore_from(&rep, 0).await).contains(&20) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        slots_of(&restore_from(&rep, 0).await),
        vec![3, 9, 12, 20, 30],
        "the refused step's row arrived with the rank as held"
    );
    rep.cancel.cancel();
    stop(r).await;
}

/// The slots written in a restore's first table, in order.
fn slots_of(rr: &everserve_wire::proto::everserve::mirror::v1::RestoreReply) -> Vec<i64> {
    rr.tables
        .first()
        .map(|t| {
            t.written
                .iter()
                .flat_map(|w| w.start..w.start + w.count)
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_replica_that_restarts_empty_is_given_the_rank_again() {
    let hold = tempfile::tempdir().expect("tmp"); // the address outlives both replicas
    let peer = hold.path().join("peer.sock");
    let rep = replica_at(&peer).await;
    expect(&rep, "node-src").await;
    let mut r = committed_alone().await;
    replicate_to(&mut r, &peer).await;
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 1").await;
    // the replica goes down and comes back with nothing, on the same address
    rep.cancel.cancel();
    drop(rep);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let rep2 = replica_at(&peer).await;
    expect(&rep2, "node-src").await;
    // the next commit: the carrier reconnects, finds the replica at (0, 0), sends the rank as
    // held at step 12 and skips nothing it needs
    let d = r.rank_dir.clone();
    assert!(
        ask(
            &mut r.client,
            &commit(1, 12, stage(&d, "s3", 1, &[4], &[6]), &[("a", 42)])
        )
        .await
        .error
        .is_empty()
    );
    wait_metric(&rep2.telemetry, "everserve_store_replica_applied_total 1").await;
    let rr = restore_from(&rep2, 0).await;
    assert_eq!((rr.step, rr.committed_tokens.clone()), (12, vec![42]));
    let w0: Vec<(i64, i64)> = rr.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(
        w0,
        vec![(3, 1), (9, 1), (12, 1)],
        "the rows from before the restart are back"
    );
    let w1: Vec<(i64, i64)> = rr.tables[1]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(
        w1,
        vec![(3, 2), (9, 1)],
        "slots 3 and 9 from before the restart, 4 from after"
    );
    assert!(
        table_rows(&rr.tables[1].path, &[4])
            .iter()
            .all(|row| row.iter().all(|&x| x == 6))
    );
    rep2.cancel.cancel();
    stop(r).await;
}

#[tokio::test]
async fn an_open_from_another_protocol_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 8, 1 << 20),
        &telemetry,
    );
    let open = MirrorRequest {
        kind: Some(mirror_request::Kind::Open(Open {
            engine: "e".into(),
            rank: 1,
            protocol: everserve_wire::PROTOCOL_VERSION + 1,
            protocol_oldest: everserve_wire::PROTOCOL_VERSION + 1,
        })),
    };
    let replies = service.answer(open).await;
    assert_eq!(replies.len(), 1);
    assert!(
        replies[0].error.contains("protocol") && replies[0].error.contains("refused"),
        "{:?}",
        replies[0]
    );
}

#[tokio::test]
async fn a_replica_that_never_answers_is_a_failed_link_after_the_reply_timeout() {
    // the carrier reconnects with backoff instead of waiting forever; the step is kept and sent
    // again on the next link, so nothing is lost
    let mut r = start().await;
    let d = r.rank_dir.clone();
    let peer = silent_peer(r.dir.path());
    r.service
        .set_replication_reply_timeout(std::time::Duration::from_millis(200));
    replicate_to(&mut r, &peer).await;
    assert!(
        ask(
            &mut r.client,
            &commit(1, 10, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)])
        )
        .await
        .error
        .is_empty()
    );
    // two reconnects' worth: the first link's state query times out, then the next
    wait_metric_absent_for(&r.telemetry, "everserve_store_replicated_total{", 700).await;
    let text = r.telemetry.render();
    assert!(
        !text.contains("everserve_store_replication_dropped_total{"),
        "a step was dropped on a reply timeout:\n{text}"
    );
    let failures: u64 = text
        .lines()
        .filter(|l| l.starts_with("everserve_store_replication_link_failures_total{"))
        .filter_map(|l| l.rsplit(' ').next()?.parse::<u64>().ok())
        .sum();
    assert!(
        failures >= 1,
        "the silent replica's link failed and is reconnecting:\n{text}"
    );
    stop(r).await;
}

#[test]
fn an_engine_name_that_is_not_one_path_component_is_refused_before_any_path_is_made() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let long = "x".repeat(256);
    for bad in [
        "",
        ".",
        "..",
        "../x",
        "a/b",
        "/abs",
        "e\0",
        "sp ace",
        long.as_str(),
    ] {
        let err = st.open(bad, 0).unwrap_err();
        assert!(
            matches!(err, StorageError::EngineName { .. }),
            "{bad:?}: {err}"
        );
        assert_eq!(err.reason(), "engine_name");
    }
    assert!(
        !dir.path().join("x").exists() && !dir.path().join("abs").exists(),
        "nothing was made outside the data directory"
    );
    st.open("llama-70b.v2_a", 0).expect("a plain name opens");
}

#[test]
fn a_row_shape_with_no_size_or_an_overflowing_one_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    st.open("e", 0).expect("open");
    for shape in [vec![u32::MAX, u32::MAX, u32::MAX], vec![4, 0], Vec::new()] {
        let err = st.prepare("e", 0, 8, &[0], "bfloat16", &shape).unwrap_err();
        assert!(
            matches!(err, StorageError::RowSize { .. }),
            "{shape:?}: {err}"
        );
    }
    assert!(st.layers("e", 0).is_empty(), "no table was made");
}

#[test]
fn a_rank_holds_rows_once_one_is_written_and_not_before() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, 1 << 20);
    let d = st.open("e", 0).expect("open");
    assert_eq!(st.state("e", 0), (false, 0, 0, false));
    st.commit("e", 0, 1, 1, Some(&stage(&d, "a", 0, &[3], &[7])), &[])
        .expect("commit");
    let (held, epoch, step, _) = st.state("e", 0);
    assert!(held);
    assert_eq!((epoch, step), (1, 1));
    // the same slot written again is one slot held, not two
    st.commit("e", 0, 1, 2, Some(&stage(&d, "b", 0, &[3], &[8])), &[])
        .expect("commit");
    assert!(st.state("e", 0).0);
    assert_eq!(st.layers("e", 0)[0].1.written_slots(), vec![3]);
}

/// Wait `ms` and assert `line` never appeared.
async fn wait_metric_absent_for(t: &Telemetry, line: &str, ms: u64) {
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    assert!(
        !t.render().contains(line),
        "{line:?} appeared:\n{}",
        t.render()
    );
}

#[tokio::test]
async fn a_source_the_coordinator_did_not_name_is_refused_and_a_named_one_taken() {
    // decision 0026, the replica's side: rows for a rank come only from the node the
    // coordinator named, whatever certificate the source holds
    let rep = replica().await;
    expect(&rep, "node-other").await;
    let mut r = committed_alone().await;
    replicate_to(&mut r, &rep.peer).await; // the carrier names itself node-src
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        !rep.telemetry
            .render()
            .contains("everserve_store_replica_applied_total 1"),
        "rows from a source the coordinator did not name were taken:\n{}",
        rep.telemetry.render()
    );
    // the coordinator moves the expectation to node-src: the next link binds and the rank as
    // held arrives
    expect(&rep, "node-src").await;
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 1").await;
    let rr = restore_from(&rep, 0).await;
    assert_eq!((rr.step, rr.committed_tokens.clone()), (11, vec![41]));
    rep.cancel.cancel();
    stop(r).await;
}

#[tokio::test]
async fn rows_on_a_link_not_bound_to_their_rank_are_refused() {
    use everserve_store::service::LinkState;

    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 8, 1 << 20),
        &telemetry,
    );
    let rows = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicated(
            everserve_wire::proto::everserve::mirror::v1::Replicated {
                engine: "e".into(),
                rank: 2,
                epoch: 0,
                step: 1,
                ..Default::default()
            },
        )),
    };
    // no first request bound the link
    let replies = service
        .answer_on(rows.clone(), &mut LinkState::local())
        .await;
    assert!(replies[0].error.contains("not bound"), "{:?}", replies[0]);
    // a link bound to rank 2 by an expected source takes rank 2 and refuses rank 3
    service
        .answer(MirrorRequest {
            kind: Some(mirror_request::Kind::Expect(
                everserve_wire::proto::everserve::mirror::v1::Expect {
                    engine: "e".into(),
                    rank: 2,
                    from_node: "node-src".into(),
                    epoch: 0,
                },
            )),
        })
        .await;
    let mut link = LinkState::from(everserve_wire::transport::Peer::Local); // a peer over a Unix socket
    let state = MirrorRequest {
        kind: Some(mirror_request::Kind::ReplicaState(
            everserve_wire::proto::everserve::mirror::v1::ReplicaStateQuery {
                engine: "e".into(),
                rank: 2,
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                node: "node-src".into(),
            },
        )),
    };
    assert!(
        service.answer_on(state, &mut link).await[0]
            .error
            .is_empty()
    );
    assert!(service.answer_on(rows, &mut link).await[0].error.is_empty());
    let other = MirrorRequest {
        kind: Some(mirror_request::Kind::Replicated(
            everserve_wire::proto::everserve::mirror::v1::Replicated {
                engine: "e".into(),
                rank: 3,
                ..Default::default()
            },
        )),
    };
    assert!(
        service.answer_on(other, &mut link).await[0]
            .error
            .contains("not bound")
    );
}

#[tokio::test]
async fn a_peer_store_cannot_say_whose_rows_this_store_takes() {
    // decision 0026: who replicates what is the coordinator's word, through this node's daemon.
    // A peer that could send the daemon's requests could make this store take rows from a node
    // of its choosing; its link carries the replication protocol and nothing else
    use everserve_store::service::LinkState;

    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 8, 1 << 20),
        &telemetry,
    );
    let mut link = LinkState::from(everserve_wire::transport::Peer::Local); // a peer over a Unix socket
    let daemons = [
        mirror_request::Kind::Expect(everserve_wire::proto::everserve::mirror::v1::Expect {
            engine: "e".into(),
            rank: 2,
            from_node: "node-evil".into(),
            epoch: 0,
        }),
        mirror_request::Kind::Replicate(everserve_wire::proto::everserve::mirror::v1::Replicate {
            engine: "e".into(),
            rank: 2,
            targets: vec!["unix:/nowhere".into()],
            node: "node-evil".into(),
        }),
    ];
    for kind in daemons {
        let refused = service
            .answer_on(MirrorRequest { kind: Some(kind) }, &mut link)
            .await;
        assert!(
            refused[0].error.contains("not by a peer store"),
            "{refused:?}"
        );
    }
    // the expectation it sent was never taken: the node it named binds nothing
    let bind = MirrorRequest {
        kind: Some(mirror_request::Kind::ReplicaState(
            everserve_wire::proto::everserve::mirror::v1::ReplicaStateQuery {
                engine: "e".into(),
                rank: 2,
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                node: "node-evil".into(),
            },
        )),
    };
    let bound = service.answer_on(bind, &mut link).await;
    assert!(bound[0].error.contains("not expected here"), "{bound:?}");
}

#[tokio::test]
async fn a_replicated_step_older_than_the_ranks_move_is_refused_after_the_fence() {
    use everserve_store::service::LinkState;

    let dir = tempfile::tempdir().expect("tmp");
    let telemetry = Telemetry::new();
    let service = Service::new(
        Storage::new(dir.path().join("data"), 8, 1 << 20),
        &telemetry,
    );
    let expect = |from: &str, epoch: u64| MirrorRequest {
        kind: Some(mirror_request::Kind::Expect(
            everserve_wire::proto::everserve::mirror::v1::Expect {
                engine: "e".into(),
                rank: 2,
                from_node: from.into(),
                epoch,
            },
        )),
    };
    let rows = |epoch: u64, step: u64| MirrorRequest {
        kind: Some(mirror_request::Kind::Replicated(
            everserve_wire::proto::everserve::mirror::v1::Replicated {
                engine: "e".into(),
                rank: 2,
                epoch,
                step,
                ..Default::default()
            },
        )),
    };
    let bind = MirrorRequest {
        kind: Some(mirror_request::Kind::ReplicaState(
            everserve_wire::proto::everserve::mirror::v1::ReplicaStateQuery {
                engine: "e".into(),
                rank: 2,
                protocol: everserve_wire::PROTOCOL_VERSION,
                protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
                node: "node-src".into(),
            },
        )),
    };
    // the replica takes rank 2 from node-src at epoch 0
    service.answer(expect("node-src", 0)).await;
    let mut link = LinkState::from(everserve_wire::transport::Peer::Local); // a peer over a Unix socket
    assert!(
        service.answer_on(bind.clone(), &mut link).await[0]
            .error
            .is_empty()
    );
    assert!(
        service.answer_on(rows(0, 5), &mut link).await[0]
            .error
            .is_empty()
    );
    // the rank moved at epoch 1: the coordinator's expectation fences it there, even as a clear
    service.answer(expect("", 1)).await;
    let late = service.answer_on(rows(0, 6), &mut link).await;
    assert!(
        late[0]
            .error
            .contains("older than the rank's move at epoch 1"),
        "{late:?}"
    );
    // the new holder's rows at epoch 1 are taken once it is expected
    service.answer(expect("node-src", 1)).await;
    let mut link = LinkState::from(everserve_wire::transport::Peer::Local); // a peer over a Unix socket
    assert!(service.answer_on(bind, &mut link).await[0].error.is_empty());
    assert!(
        service.answer_on(rows(1, 1), &mut link).await[0]
            .error
            .is_empty()
    );
}

#[tokio::test]
async fn the_rows_query_reports_each_slots_written_step_and_bytes() {
    let mut r = committed_alone().await;
    let q = MirrorRequest {
        kind: Some(mirror_request::Kind::Rows(
            everserve_wire::proto::everserve::mirror::v1::RowsQuery {
                engine: "e".into(),
                rank: 2,
                table: 0,
                slots: vec![3, 9, 12, 5],
            },
        )),
    };
    let reply = ask(&mut r.client, &q).await;
    let Some(mirror_reply::Kind::Rows(rows)) = reply.kind else {
        panic!("{reply:?}")
    };
    assert_eq!((rows.epoch, rows.step), (1, 11));
    let got: Vec<(i64, u64)> = rows.rows.iter().map(|x| (x.slot, x.written_step)).collect();
    assert_eq!(got, vec![(3, 10), (9, 11), (12, 11), (5, 0)]);
    assert!(
        rows.rows[1].data.iter().all(|&b| b == 5),
        "slot 9 holds step 11's rows"
    );
    assert!(
        rows.rows[0].data.iter().all(|&b| b == 1),
        "slot 3 holds step 10's rows"
    );
    stop(r).await;
}

#[tokio::test]
async fn the_first_commit_at_step_zero_reaches_a_replica_named_before_it() {
    // at epoch 0 as the engine starts, with the target named before the prefill commits: a resync
    // that finds nothing held on either side and takes (0, 0) as covered skips the prefill, which
    // is step 0, as already there
    let rep = replica().await;
    expect(&rep, "node-src").await;
    let mut r = start().await;
    let d = r.rank_dir.clone();
    replicate_to(&mut r, &rep.peer).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await; // the carrier links first
    assert!(
        ask(
            &mut r.client,
            &commit(0, 0, stage_two(&d, "s0", &[3, 9], 1, 2), &[("a", 40)])
        )
        .await
        .error
        .is_empty()
    );
    assert!(
        ask(
            &mut r.client,
            &commit(0, 1, stage(&d, "s1", 0, &[12], &[5]), &[("a", 41)])
        )
        .await
        .error
        .is_empty()
    );
    wait_metric(&rep.telemetry, "everserve_store_replica_applied_total 2").await;
    let rr = restore_from(&rep, 0).await;
    let w0: Vec<(i64, i64)> = rr.tables[0]
        .written
        .iter()
        .map(|w| (w.start, w.count))
        .collect();
    assert_eq!(
        w0,
        vec![(3, 1), (9, 1), (12, 1)],
        "step 0's rows are on the replica"
    );
    assert!(
        table_rows(&rr.tables[0].path, &[3])
            .iter()
            .all(|row| row.iter().all(|&x| x == 1))
    );
    rep.cancel.cancel();
    stop(r).await;
}

/// Open rank 2 of `e` on `client`; return the staging prefix the store gave the connection.
async fn open_prefix(client: &mut UnixStream) -> String {
    let open = MirrorRequest {
        kind: Some(mirror_request::Kind::Open(Open {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "e".into(),
            rank: 2,
        })),
    };
    let opened = ask(client, &open).await;
    let Some(mirror_reply::Kind::Opened(o)) = opened.kind else {
        panic!("{opened:?}")
    };
    o.stage_prefix
}

#[tokio::test]
async fn a_connection_that_ends_takes_its_staging_files_and_no_other() {
    let mut r = start().await;
    let kept = open_prefix(&mut r.client).await;
    let mut other = UnixStream::connect(&r.sock).await.expect("connect");
    let gone = open_prefix(&mut other).await;
    assert_ne!(kept, gone, "each connection has its own prefix");
    for name in [
        format!("{kept}1.rows"),
        format!("{gone}1.rows"),
        format!("{gone}2.rows"),
    ] {
        std::fs::write(r.rank_dir.join(name), [0u8; ROW]).expect("stage file");
    }
    drop(other); // a worker killed mid-step: its connection ends, its files were never deleted
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while r.rank_dir.join(format!("{gone}1.rows")).exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "the ended connection's files stayed"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(!r.rank_dir.join(format!("{gone}2.rows")).exists());
    assert!(
        r.rank_dir.join(format!("{kept}1.rows")).exists(),
        "a live connection's staging file stays"
    );
    stop(r).await;
}

#[test]
fn a_links_end_removes_its_prefix_only() {
    let dir = tempfile::tempdir().expect("tmp");
    let mut st = Storage::new(dir.path().join("data"), 8, ROW * 8);
    let d = st.open("e", 0).expect("open");
    let names = [
        "stage-1-1.rows",
        "stage-1-2.rows",
        "stage-11-1.rows",
        "stage-2-1.rows",
        "table0.rows",
    ];
    for name in names {
        std::fs::write(d.join(name), [0u8; ROW]).expect("file");
    }
    assert_eq!(st.end_link("e", 0, "stage-1-"), 2);
    let mut left: Vec<_> = std::fs::read_dir(&d)
        .expect("read")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, ["stage-11-1.rows", "stage-2-1.rows", "table0.rows"]);
}

#[test]
fn the_startup_sweep_removes_every_staging_file_and_keeps_the_tables() {
    let dir = tempfile::tempdir().expect("tmp");
    let data = dir.path().join("data");
    let st = Storage::new(data.clone(), 8, ROW * 8);
    assert_eq!(st.sweep_stages().expect("an absent data dir"), 0);
    for (rank, name) in [
        ("rank0", "stage-3-1.rows"),
        ("rank0", "stage-41234-7.rows"), // a pid-named file from an earlier build
        ("rank1", "stage-9-2.rows"),
        ("rank1", "table0.rows"),
    ] {
        let d = data.join("e").join(rank);
        std::fs::create_dir_all(&d).expect("dir");
        std::fs::write(d.join(name), [0u8; ROW]).expect("file");
    }
    assert_eq!(st.sweep_stages().expect("sweep"), 3);
    assert!(data.join("e/rank1/table0.rows").exists(), "tables stay");
    assert_eq!(
        std::fs::read_dir(data.join("e/rank0"))
            .expect("read")
            .count(),
        0
    );
}

#[test]
fn a_snapshot_read_while_commits_go_on_ends_equal_once_the_later_steps_apply() {
    use everserve_wire::proto::everserve::mirror::v1::Replicated;

    // the snapshot is read a run at a time with commits between; a run can hold a row a later
    // commit wrote, and replaying the steps after the snapshot's label makes the replica exact
    let dir = tempfile::tempdir().expect("tmp");
    let mut src = Storage::new(dir.path().join("src"), 64, 1 << 20);
    let d = src.open("e", 0).expect("open");
    src.commit(
        "e",
        0,
        1,
        1,
        Some(&stage_two(&d, "a", &[1, 2, 3], 1, 2)),
        &[("r".into(), 3)],
    )
    .expect("step 1");
    let plan = src.snapshot_plan("e", 0).expect("plan");
    assert_eq!((plan.epoch, plan.step), (1, 1));
    let (first, next) = src.snapshot_run("e", 0, 0, 0, 2).expect("run");
    assert_eq!((first.slots.clone(), next), (vec![1, 2], Some(3)));
    // a commit lands mid-snapshot: slot 2 of table 0 again (already read), slot 3 of table 1
    // (not yet read), and a new slot 4 in both
    let later = stage_two(&d, "b", &[2, 3, 4], 7, 8);
    src.commit("e", 0, 1, 2, Some(&later), &[("r".into(), 5)])
        .expect("step 2");
    let (rest, done) = src.snapshot_run("e", 0, 0, 3, 64).expect("run");
    assert!(done.is_none());
    let (table1, _) = src.snapshot_run("e", 0, 1, 0, 64).expect("run");
    let head = |tables| Replicated {
        engine: "e".into(),
        rank: 0,
        epoch: plan.epoch,
        step: plan.step,
        tables,
        request_ids: Vec::new(),
        committed_tokens: Vec::new(),
    };
    let mut rep = Storage::new(dir.path().join("rep"), 64, 1 << 20);
    rep.apply_replicated(&head(vec![first])).expect("part 1");
    rep.apply_replicated(&Replicated {
        request_ids: plan.request_ids.clone(),
        committed_tokens: plan.committed_tokens.clone(),
        ..head(vec![rest, table1])
    })
    .expect("part 2");
    // the step after the label, as the carrier's queue replays it
    let step2 = src
        .replicated("e", 0, 1, 2, Some(&later), &[("r".into(), 5)])
        .expect("step 2 as replicated");
    rep.apply_replicated(&step2).expect("step 2");
    for table in [0, 1] {
        assert_eq!(
            rep.rows("e", 0, table, &[1, 2, 3, 4]),
            src.rows("e", 0, table, &[1, 2, 3, 4]),
            "table {table}"
        );
    }
}
