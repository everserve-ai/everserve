// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The file lease (decision 0036): one hold at a time, each at a higher term, its record read by
//! a watch.

use everserve_controller::lease::{FileLease, FileLeaseWatch, Lease, LeaseWatch};
use everserve_controller::proto::everserve::coord::v1::LeaseRecord;

#[tokio::test]
async fn the_file_lease_has_one_hold_at_a_time() {
    let dir = tempfile::tempdir().expect("tmp");
    let lease = FileLease::new(dir.path().join("coordinator.lease"));
    let a = lease.try_acquire().await.expect("a").expect("free");
    assert!(
        lease.try_acquire().await.expect("b").is_none(),
        "a second hold is refused"
    );
    assert!(
        !a.ended().is_cancelled(),
        "a file hold lasts as long as it is kept"
    );
    a.release().await.expect("release");
    let b = lease
        .try_acquire()
        .await
        .expect("b")
        .expect("released, it can be taken");
    drop(b);
    assert!(
        lease.try_acquire().await.expect("c").is_some(),
        "a dropped hold lets the lease go with its file"
    );
    assert!(lease.describe().contains("coordinator.lease"));
}

#[tokio::test]
async fn every_hold_takes_the_next_term_and_a_reopen_continues_from_it() {
    // decision 0036: the term is a fencing token, raised by every holder and never reused
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("coord.lease");
    let a = FileLease::new(&path)
        .try_acquire()
        .await
        .expect("a")
        .expect("free");
    assert_eq!(a.term(), 1);
    a.release().await.expect("release");
    let b = FileLease::new(&path)
        .try_acquire()
        .await
        .expect("b")
        .expect("free");
    assert_eq!(b.term(), 2, "another process's lease takes the next term");
    b.release().await.expect("release");
    let c = FileLease::new(&path)
        .try_acquire()
        .await
        .expect("c")
        .expect("free");
    assert_eq!(c.term(), 3);
}

#[tokio::test]
async fn a_watch_reads_the_holds_record_with_its_term() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("coord.lease");
    let watch = FileLeaseWatch::new(&path);
    assert_eq!(
        watch.read().await.expect("read"),
        None,
        "nothing published yet"
    );
    let hold = FileLease::new(&path)
        .try_acquire()
        .await
        .expect("a")
        .expect("free");
    hold.publish(LeaseRecord {
        holder: "host-a:1".into(),
        address: "unix:/run/coord.sock".into(),
        term: 99, // the hold writes its own term
        ..LeaseRecord::default()
    })
    .await
    .expect("publish");
    let read = watch.read().await.expect("read").expect("a record");
    assert_eq!(
        (read.holder.as_str(), read.term, read.address.as_str()),
        ("host-a:1", 1, "unix:/run/coord.sock")
    );
}
