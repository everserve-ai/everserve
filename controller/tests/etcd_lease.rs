// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The etcd lease against a real etcd (decision 0036): every test starts its own single-member
//! cluster from the `etcd` binary (on `PATH`, or at `EVERSERVE_TEST_ETCD`) and fails without
//! one; a lease test that skips proves nothing.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use everserve_controller::lease::{EtcdLease, EtcdLeaseWatch, Hold, Lease, LeaseError, LeaseWatch};
use everserve_controller::proto::everserve::coord::v1::LeaseRecord;

mod common;

use common::etcd::Etcd;
use common::{Relay, Spawned, bounded, exited};

/// The etcd lease and its watch at `key`, on a test's etcd.
trait EtcdLeases {
    async fn lease(&self, key: &str, ttl_s: u64) -> EtcdLease;
    async fn watch(&self, key: &str) -> EtcdLeaseWatch;
}

impl EtcdLeases for Etcd {
    async fn lease(&self, key: &str, ttl_s: u64) -> EtcdLease {
        EtcdLease::connect(&self.endpoints(), key, Duration::from_secs(ttl_s), None)
            .await
            .expect("lease")
    }

    async fn watch(&self, key: &str) -> EtcdLeaseWatch {
        EtcdLeaseWatch::connect(&self.endpoints(), key, Duration::from_secs(2), None)
            .await
            .expect("watch")
    }
}

fn record(holder: &str) -> LeaseRecord {
    LeaseRecord {
        holder: holder.into(),
        address: format!("unix:/run/{holder}.sock"),
        term: 99, // the lease writes its own term, and a reader takes etcd's
        ..LeaseRecord::default()
    }
}

/// When `hold` ends, by the clock of the moment its token is cancelled.
fn ended_at(hold: &dyn Hold) -> tokio::task::JoinHandle<Instant> {
    let ended = hold.ended().clone();
    tokio::spawn(async move {
        ended.cancelled().await;
        Instant::now()
    })
}

#[tokio::test]
async fn the_etcd_lease_has_one_hold_at_a_time_each_at_a_higher_term() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let key = "/everserve/coordinator";
        let lease = etcd.lease(key, 10).await;
        let watch = etcd.watch(key).await;
        let a = lease.try_acquire().await.expect("a").expect("free");
        assert!(
            lease.try_acquire().await.expect("b").is_none(),
            "a second hold is refused"
        );
        a.publish(record("a")).await.expect("publish");
        let read = watch.read().await.expect("read").expect("a record");
        assert_eq!(
            (read.holder.as_str(), read.term),
            ("a", a.term()),
            "the record carries the key's create revision as its term"
        );
        let a_term = a.term();
        a.release().await.expect("release");
        assert_eq!(
            watch.read().await.expect("read"),
            None,
            "released: no holder"
        );
        let b = lease
            .try_acquire()
            .await
            .expect("b")
            .expect("released, it can be taken");
        assert!(b.term() > a_term, "{} after {a_term}", b.term());
    }))
    .await;
}

#[tokio::test]
async fn a_hold_cut_off_from_etcd_ends_before_a_contender_can_take_the_lease() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let key = "/everserve/partition";
        let relay = Relay::start(&etcd.endpoint).await;
        let cut_off = EtcdLease::connect(
            std::slice::from_ref(&relay.endpoint),
            key,
            Duration::from_secs(2),
            None,
        )
        .await
        .expect("lease");
        let contender = etcd.lease(key, 2).await;
        let asked = Instant::now();
        let a = cut_off.try_acquire().await.expect("a").expect("free");
        let a_ended = ended_at(a.as_ref());
        // a partition at once: every renewal hangs, unanswered, so the grant is the last answer
        // and the hold's deadline is three quarters of the 2 s TTL after it was asked for; the
        // contender reaches etcd directly
        relay.cut();
        let b_took = loop {
            if let Some(b) = contender.try_acquire().await.expect("b") {
                assert!(b.term() > a.term());
                break Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        let a_ended = a_ended.await.expect("ended");
        assert!(
            a_ended < b_took,
            "the hold ended {:?} after the contender took the lease",
            a_ended.saturating_duration_since(b_took)
        );
        let held = a_ended - asked;
        assert!(
            held >= Duration::from_millis(1500) && held < Duration::from_millis(1750),
            "the hold ends three quarters of the TTL after its last answered renewal was sent, \
             not before and not at the TTL: it held {held:?}"
        );
    }))
    .await;
}

#[tokio::test]
async fn a_hold_ends_when_its_renewal_finds_the_key_taken_from_under_it() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let key = "/everserve/renewal";
        let lease = etcd.lease(key, 2).await;
        let a = lease.try_acquire().await.expect("a").expect("free");
        // renewed past its first deadline (1.5 s), the hold stands
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!a.ended().is_cancelled(), "renewal keeps the hold");
        // the key is deleted under a live etcd lease: free for any contender, though the etcd
        // lease itself still answers keep-alives
        etcd.client().await.delete(key, None).await.expect("delete");
        tokio::time::timeout(Duration::from_secs(2), a.ended().cancelled())
            .await
            .expect("a renewal checks the key, not only the etcd lease");
        assert!(lease.try_acquire().await.expect("b").is_some());
    }))
    .await;
}

#[tokio::test]
async fn a_hold_cannot_write_over_the_record_of_the_hold_after_it() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let key = "/everserve/guard";
        // a TTL of 10 s renews every 2.5 s: this runs between a's renewals
        let lease = etcd.lease(key, 10).await;
        let a = lease.try_acquire().await.expect("a").expect("free");
        etcd.client().await.delete(key, None).await.expect("delete");
        let b = lease
            .try_acquire()
            .await
            .expect("b")
            .expect("deleted, it can be taken");
        b.publish(record("b")).await.expect("publish");
        assert!(
            !a.ended().is_cancelled(),
            "a's renewal has not yet run: only the guard stops it"
        );
        assert!(
            matches!(a.publish(record("a")).await, Err(LeaseError::Lost(_))),
            "the write is guarded by the key's create revision"
        );
        assert!(a.ended().is_cancelled(), "the store's word ends the hold");
        let read = etcd
            .watch(key)
            .await
            .read()
            .await
            .expect("read")
            .expect("b's");
        assert_eq!((read.holder.as_str(), read.term), ("b", b.term()));
    }))
    .await;
}

/// A coordinator process holding `key` on `etcd`, listening on `sock`.
fn coordinator(etcd: &Etcd, key: &str, sock: &std::path::Path) -> Spawned {
    let policy =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/policy-two-node.json");
    Command::new(env!("CARGO_BIN_EXE_everserve-controller"))
        .arg("coord")
        .arg("--policy")
        .arg(policy)
        .arg("--listen")
        .arg(format!("unix:{}", sock.display()))
        .arg("--lease")
        .arg(format!("etcd://{}{key}", etcd.endpoint))
        .args(["--lease-ttl-secs", "2", "--graceful-shutdown-secs", "1"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Spawned)
        .expect("spawn the coordinator")
}

/// The record at `key` once its holder is `sock`'s coordinator.
async fn held_by(watch: &EtcdLeaseWatch, sock: &std::path::Path) -> LeaseRecord {
    let address = format!("unix:{}", sock.display());
    loop {
        if let Ok(Some(r)) = watch.read().await
            && r.address == address
        {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn a_coordinator_that_loses_its_lease_exits_and_a_standby_takes_over() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let dir = tempfile::tempdir().expect("tmp");
        let key = "/everserve/coordinator";
        let watch = etcd.watch(key).await;
        let (a_sock, b_sock, c_sock) = (
            dir.path().join("a.sock"),
            dir.path().join("b.sock"),
            dir.path().join("c.sock"),
        );
        let mut a = coordinator(&etcd, key, &a_sock);
        let a_term = held_by(&watch, &a_sock).await.term;
        let b = coordinator(&etcd, key, &b_sock);
        // a's key is deleted under its live etcd lease: b takes it, and a, at its next renewal,
        // finds the key is not its own and exits rather than act beside b
        etcd.client().await.delete(key, None).await.expect("delete");
        let b_term = held_by(&watch, &b_sock).await.term;
        assert!(b_term > a_term);
        assert_eq!(
            exited(&mut a).await,
            Some(75),
            "a lost its lease: EXIT_LEASE_LOST"
        );

        // b dies without a word: its lease expires, and a standby takes over at a higher term
        let c = coordinator(&etcd, key, &c_sock);
        drop(b);
        let c_term = held_by(&watch, &c_sock).await.term;
        assert!(c_term > b_term);
        drop(c);
    }))
    .await;
}

#[tokio::test]
async fn a_coordinator_cut_off_from_etcd_stops_by_its_own_clock() {
    bounded(Box::pin(async {
        let etcd = Etcd::start().await;
        let dir = tempfile::tempdir().expect("tmp");
        let key = "/everserve/coordinator";
        let watch = etcd.watch(key).await;
        let sock = dir.path().join("a.sock");
        let mut a = coordinator(&etcd, key, &sock);
        held_by(&watch, &sock).await;
        // etcd stops answering, mid-hold: renewals hang rather than fail, and only the hold's
        // own deadline (three quarters of the 2 s TTL after the last answered renewal) ends it
        let pid = etcd.child.id().to_string();
        let stopped = Command::new("kill")
            .args(["-STOP", &pid])
            .status()
            .expect("kill");
        assert!(stopped.success());
        let at = Instant::now();
        let code = tokio::time::timeout(Duration::from_secs(8), exited(&mut a))
            .await
            .expect("the coordinator stopped while etcd was silent");
        assert_eq!(code, Some(75), "EXIT_LEASE_LOST");
        // the hold ends by 1.5 s; the graceful period (1 s) and the process's exit follow
        assert!(at.elapsed() < Duration::from_secs(4), "{:?}", at.elapsed());
        let _ = Command::new("kill").args(["-CONT", &pid]).status();
    }))
    .await;
}
