// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A rank's compile cache moved between nodes by their stores (decision 0043): a spare's node
//! fetches the set its rank's manifest lists from a holder, past one that does not answer, and it
//! lands whole with its manifest last; a holder serves only the pair asked and only what its
//! manifest lists; a file altered on the holder is never served whole and nothing of it lands; a
//! fetch past the budget writes nothing; and only the node's own socket may ask for a fetch.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use everserve_store::cache::CompileCaches;
use everserve_store::service::Service;
use everserve_store::storage::Storage;
use everserve_telemetry::Telemetry;
use everserve_wire::frame;
use everserve_wire::proto::everserve::mirror::v1::{
    CacheFetch, CacheManifestQuery, CacheRead, MirrorReply, MirrorRequest, mirror_reply,
    mirror_request,
};
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

const KEY: &str = "586b266d";
const DIGEST: &str = "d1";

/// A store keeping compile caches under its own cache root, at most `budget` bytes: its node's
/// socket, its peer socket, and the root.
struct Node {
    sock: PathBuf,
    peer: PathBuf,
    cache: PathBuf,
    cancel: CancellationToken,
    _dir: tempfile::TempDir,
}

async fn node(budget: u64) -> Node {
    let dir = tempfile::tempdir().expect("tmp");
    let cache = dir.path().join("vllm-cache");
    std::fs::create_dir_all(&cache).expect("cache root");
    let service = Service::with_tls(
        Storage::new(dir.path().join("data"), 64, 1 << 20),
        &Telemetry::new(),
        None,
        "store",
        Some(CompileCaches::new(&cache, budget, None, "store")),
    );
    let cancel = CancellationToken::new();
    let peer = dir.path().join("peer.sock");
    let listener = everserve_wire::transport::Listener::bind(
        &everserve_wire::transport::Address::Unix(peer.clone()),
        None,
    )
    .await
    .expect("bind");
    tokio::spawn(Arc::clone(&service).serve_peers(listener, cancel.clone()));
    let sock = dir.path().join("m.sock");
    tokio::spawn(service.serve(sock.clone(), cancel.clone()));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    Node {
        sock,
        peer,
        cache,
        cancel,
        _dir: dir,
    }
}

fn key_dir(cache: &Path) -> PathBuf {
    cache
        .join("torch_compile_cache")
        .join("torch_aot_compile")
        .join(KEY)
}

fn sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Rank 1's set under `cache` as a serving worker leaves it: `files` and their manifest.
fn serve_set(cache: &Path, files: &[(&str, Vec<u8>)]) {
    let dir = key_dir(cache);
    let mut listed = Vec::new();
    for (path, bytes) in files {
        let p = dir.join(path);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("dirs");
        std::fs::write(&p, bytes).expect("file");
        listed
            .push(serde_json::json!({"path": path, "size": bytes.len(), "sha256": sha256(bytes)}));
    }
    let manifest =
        serde_json::json!({"key": KEY, "rank": 1, "dp": 0, "digest": DIGEST, "files": listed});
    std::fs::write(dir.join("rank_1_0.manifest.json"), manifest.to_string()).expect("manifest");
}

/// The set: the rank's graph, larger than one chunk, and a kernel its graph names.
fn set() -> Vec<(&'static str, Vec<u8>)> {
    let graph: Vec<u8> = (0..(5u32 << 20)).map(|i| (i % 251) as u8).collect();
    vec![
        ("rank_1_0/model", graph),
        (
            "inductor_cache/ab/cab12.py",
            b"def kernel(): pass\n".to_vec(),
        ),
    ]
}

async fn ask(sock: &Path, kind: mirror_request::Kind) -> MirrorReply {
    let mut s = UnixStream::connect(sock).await.expect("connect");
    frame::send(&mut s, &MirrorRequest { kind: Some(kind) })
        .await
        .expect("send");
    frame::recv(&mut s).await.expect("recv")
}

fn fetch(holders: &[&Path]) -> mirror_request::Kind {
    mirror_request::Kind::CacheFetch(CacheFetch {
        key: KEY.into(),
        rank: 1,
        dp: 0,
        digest: DIGEST.into(),
        holders: holders.iter().map(|p| p.display().to_string()).collect(),
    })
}

fn done(r: &MirrorReply) -> bool {
    r.error.is_empty() && matches!(r.kind, Some(mirror_reply::Kind::Done(_)))
}

#[tokio::test]
async fn a_spares_node_fetches_its_ranks_set_past_a_dead_holder_and_it_lands_whole() {
    let holder = node(1 << 30).await;
    let spare = node(1 << 30).await;
    serve_set(&holder.cache, &set());
    let dead = holder.peer.with_file_name("gone.sock");

    let r = ask(&spare.sock, fetch(&[&dead, &holder.peer])).await;
    assert!(done(&r), "{r:?}");
    for (path, bytes) in set() {
        let landed = std::fs::read(key_dir(&spare.cache).join(path)).expect("landed");
        assert_eq!(landed, bytes, "{path}");
    }
    let manifest = |c: &Path| {
        serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(key_dir(c).join("rank_1_0.manifest.json")).expect("manifest"),
        )
        .expect("json")
    };
    assert_eq!(manifest(&spare.cache), manifest(&holder.cache));
    let leftovers: Vec<_> = walk(&key_dir(&spare.cache))
        .into_iter()
        .filter(|p| p.to_string_lossy().contains(".fetching"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    // the node group's to read and write, whatever the process's umask: the spare's vLLM reads
    // the set and writes beside it (decision 0043)
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| std::fs::metadata(p).expect("metadata").permissions().mode();
        let key = key_dir(&spare.cache);
        for f in [
            "rank_1_0/model",
            "inductor_cache/ab/cab12.py",
            "rank_1_0.manifest.json",
        ] {
            assert_eq!(mode(&key.join(f)) & 0o777, 0o660, "{f}");
        }
        for d in ["", "rank_1_0", "inductor_cache", "inductor_cache/ab"] {
            assert_eq!(mode(&key.join(d)) & 0o7777, 0o2770, "{d:?}");
        }
    }
    // held now: asked again with no holder, it is in place already
    let r = ask(&spare.sock, fetch(&[])).await;
    assert!(done(&r), "{r:?}");
    // and the spare's node is a holder for the next
    let third = node(1 << 30).await;
    let r = ask(&third.sock, fetch(&[&spare.peer])).await;
    assert!(done(&r), "{r:?}");
    for n in [holder, spare, third] {
        n.cancel.cancel();
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).expect("dir").flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

#[tokio::test]
async fn a_holder_serves_only_the_pair_asked_and_only_what_its_manifest_lists() {
    let holder = node(1 << 30).await;
    serve_set(&holder.cache, &set());
    let query = |digest: &str, key: &str| {
        mirror_request::Kind::CacheManifest(CacheManifestQuery {
            key: key.into(),
            rank: 1,
            dp: 0,
            digest: digest.into(),
        })
    };
    let r = ask(&holder.peer, query(DIGEST, KEY)).await;
    let Some(mirror_reply::Kind::CacheManifest(m)) = r.kind else {
        panic!("{r:?}")
    };
    assert_eq!(m.files.len(), 2);
    let r = ask(&holder.peer, query("other", KEY)).await;
    assert!(r.error.contains("not other"), "another digest: {r:?}");
    let r = ask(&holder.peer, query(DIGEST, "../..")).await;
    assert!(
        r.error.contains("not a path"),
        "a key leaving the root: {r:?}"
    );
    let read = |path: &str| {
        mirror_request::Kind::CacheRead(CacheRead {
            key: KEY.into(),
            rank: 1,
            dp: 0,
            digest: DIGEST.into(),
            path: path.into(),
            offset: 0,
        })
    };
    let r = ask(&holder.peer, read("rank_1_0.manifest.json")).await;
    assert!(r.error.contains("not in the manifest"), "{r:?}");
    let r = ask(&holder.peer, read("inductor_cache/ab/cab12.py")).await;
    let Some(mirror_reply::Kind::CacheChunk(c)) = r.kind else {
        panic!("{r:?}")
    };
    assert_eq!(c.data, b"def kernel(): pass\n");

    // a manifest whose path leaves the key's directory is no manifest
    serve_set(&holder.cache, &[("../escape", b"x".to_vec())]);
    let r = ask(&holder.peer, query(DIGEST, KEY)).await;
    assert!(r.error.contains("not a path"), "{r:?}");
    holder.cancel.cancel();
}

#[tokio::test]
async fn a_file_altered_on_the_holder_is_never_served_whole_and_nothing_of_it_lands() {
    let holder = node(1 << 30).await;
    let spare = node(1 << 30).await;
    serve_set(&holder.cache, &set());
    std::fs::write(
        key_dir(&holder.cache).join("inductor_cache/ab/cab12.py"),
        b"def kernel(): return 1\n",
    )
    .expect("alter");
    let r = ask(
        &holder.peer,
        mirror_request::Kind::CacheRead(CacheRead {
            key: KEY.into(),
            rank: 1,
            dp: 0,
            digest: DIGEST.into(),
            path: "inductor_cache/ab/cab12.py".into(),
            offset: 0,
        }),
    )
    .await;
    assert!(
        r.error.contains("sha256"),
        "the holder checks before it serves: {r:?}"
    );

    let r = ask(&spare.sock, fetch(&[&holder.peer])).await;
    assert!(!done(&r) && r.error.contains("sha256"), "{r:?}");
    assert!(
        !key_dir(&spare.cache)
            .join("rank_1_0.manifest.json")
            .exists(),
        "no manifest: the set is not in place"
    );
    assert!(
        !key_dir(&spare.cache)
            .join("inductor_cache/ab/cab12.py")
            .exists()
    );
    holder.cancel.cancel();
    spare.cancel.cancel();
}

#[tokio::test]
async fn a_fetch_past_the_budget_writes_nothing() {
    let holder = node(1 << 30).await;
    let spare = node(1 << 20).await;
    serve_set(&holder.cache, &set());
    let r = ask(&spare.sock, fetch(&[&holder.peer])).await;
    assert!(r.error.contains("over 1048576"), "{r:?}");
    assert!(!key_dir(&spare.cache).exists(), "nothing written");
    holder.cancel.cancel();
    spare.cancel.cancel();
}

#[tokio::test]
async fn only_the_nodes_own_socket_asks_for_a_fetch() {
    let holder = node(1 << 30).await;
    let other = node(1 << 30).await;
    serve_set(&holder.cache, &set());
    let r = ask(&other.peer, fetch(&[&holder.peer])).await;
    assert!(r.error.contains("node's socket"), "{r:?}");
    assert!(!key_dir(&other.cache).exists());
    holder.cancel.cancel();
    other.cancel.cancel();
}

#[tokio::test]
async fn bytes_a_holder_serves_that_are_not_what_it_listed_never_land() {
    // a holder that lists one file and serves other bytes for it: the fetching store's own
    // check, not the holder's, must refuse them
    use everserve_wire::proto::everserve::mirror::v1::{CacheChunk, CacheFile, CacheManifest};
    let spare = node(1 << 30).await;
    let dir = tempfile::tempdir().expect("tmp");
    let peer = dir.path().join("liar.sock");
    let listener = tokio::net::UnixListener::bind(&peer).expect("bind");
    tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.expect("accept");
        loop {
            let Ok(req) = frame::recv::<_, MirrorRequest>(&mut s).await else {
                return;
            };
            let kind = match req.kind {
                Some(mirror_request::Kind::CacheManifest(_)) => {
                    mirror_reply::Kind::CacheManifest(CacheManifest {
                        key: KEY.into(),
                        rank: 1,
                        dp: 0,
                        digest: DIGEST.into(),
                        files: vec![CacheFile {
                            path: "rank_1_0/model".into(),
                            size: 4,
                            sha256: sha256(b"good"),
                        }],
                    })
                }
                _ => mirror_reply::Kind::CacheChunk(CacheChunk {
                    data: b"evil".to_vec(),
                }),
            };
            let reply = MirrorReply {
                kind: Some(kind),
                error: String::new(),
            };
            if frame::send(&mut s, &reply).await.is_err() {
                return;
            }
        }
    });
    let r = ask(&spare.sock, fetch(&[&peer])).await;
    assert!(!done(&r) && r.error.contains("sha256"), "{r:?}");
    assert!(!key_dir(&spare.cache).join("rank_1_0/model").exists());
    assert!(
        !key_dir(&spare.cache)
            .join("rank_1_0.manifest.json")
            .exists()
    );
    spare.cancel.cancel();
}

#[test]
fn an_unfinished_fetch_left_by_an_earlier_process_is_swept_at_start() {
    let dir = tempfile::tempdir().expect("tmp");
    let cache = dir.path().join("vllm-cache");
    let pair = cache
        .join("torch_compile_cache")
        .join("torch_aot_compile")
        .join("k")
        .join("rank_0_0");
    std::fs::create_dir_all(&pair).expect("pair");
    std::fs::write(pair.join(".model.fetching"), b"half").expect("partial");
    std::fs::write(pair.join("manifest.json.fetching"), b"{").expect("partial manifest");
    std::fs::write(pair.join("model"), b"whole").expect("a finished file");
    let _caches = CompileCaches::new(&cache, 1 << 20, None, "store");
    assert!(
        !pair.join(".model.fetching").exists(),
        "the partial file went"
    );
    assert!(
        !pair.join("manifest.json.fetching").exists(),
        "the partial manifest went"
    );
    assert_eq!(std::fs::read(pair.join("model")).expect("kept"), b"whole");
}
