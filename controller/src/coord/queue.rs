// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The controllers' work queue (decision 0036): keys to reconcile, each queued once however often
//! it is asked before its reconcile runs, with the queue's depth and each key's wait as metrics.
//! The engine controller keys it by engine and the node controller by node, each its own queue.

use std::collections::BTreeMap;
use std::time::Instant;

use everserve_telemetry::prometheus::{Histogram, IntGauge};
use tokio::sync::{Mutex, mpsc};

use crate::metrics::CoordMetrics;

/// A work queue keyed by `K`: a key queued twice before the worker takes it runs once.
pub(super) struct WorkQueue<K> {
    /// Keys queued and not yet taken by the worker, with when each was queued.
    queued: Mutex<BTreeMap<K, Instant>>,
    tx: mpsc::UnboundedSender<K>,
    rx: Mutex<Option<mpsc::UnboundedReceiver<K>>>,
    /// The queue's depth and its keys' wait, as metrics.
    depth: IntGauge,
    wait: Histogram,
}

impl<K: Ord + Clone> WorkQueue<K> {
    /// A queue whose metrics carry `kind` as their label: `engine` or `node`.
    pub(super) fn new(metrics: &CoordMetrics, kind: &str) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            queued: Mutex::new(BTreeMap::new()),
            tx,
            rx: Mutex::new(Some(rx)),
            depth: metrics.control.queue_depth.with_label_values(&[kind]),
            wait: metrics
                .control
                .queue_wait_seconds
                .with_label_values(&[kind]),
        }
    }

    /// Queue `key`, once however often it is asked before the worker takes it.
    pub(super) async fn enqueue(&self, key: K) {
        let mut queued = self.queued.lock().await;
        if !queued.contains_key(&key) {
            queued.insert(key.clone(), Instant::now());
            self.depth
                .set(i64::try_from(queued.len()).unwrap_or(i64::MAX));
            let _ = self.tx.send(key);
        }
    }

    /// The worker took `key` off the queue: how long it waited.
    pub(super) async fn take(&self, key: &K) {
        let mut queued = self.queued.lock().await;
        if let Some(at) = queued.remove(key) {
            self.wait.observe(at.elapsed().as_secs_f64());
        }
        self.depth
            .set(i64::try_from(queued.len()).unwrap_or(i64::MAX));
    }

    /// The queue's receiving end, to the first worker that asks; None to every later one, so the
    /// worker is started once.
    pub(super) async fn receiver(&self) -> Option<mpsc::UnboundedReceiver<K>> {
        self.rx.lock().await.take()
    }
}
