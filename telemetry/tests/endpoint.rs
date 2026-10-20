// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The endpoint serves registered metrics and the health flag, closes a connection that sends
//! nothing, and stops on cancel.

use everserve_telemetry::Telemetry;
use everserve_telemetry::prometheus::{IntCounter, Opts};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

async fn http_get(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
    let mut s = tokio::net::TcpStream::connect(addr).await.expect("connect");
    s.write_all(format!("GET {path} HTTP/1.0\r\nHost: x\r\n\r\n").as_bytes())
        .await
        .expect("write");
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.expect("read");
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status");
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
    (status, body)
}

#[tokio::test]
async fn metrics_and_health_are_served_until_cancelled() {
    let t = Telemetry::new();
    let changes = IntCounter::with_opts(Opts::new("everserve_changes_total", "membership changes"))
        .expect("counter");
    t.registry()
        .register(Box::new(changes.clone()))
        .expect("register");
    changes.inc_by(3);
    let cancel = CancellationToken::new();
    let addr = t
        .clone()
        .start(
            "127.0.0.1:0".parse().unwrap(),
            everserve_telemetry::HEADER_READ,
            cancel.clone(),
        )
        .await
        .expect("bound");

    let (status, body) = http_get(addr, "/metrics").await;
    assert_eq!(status, 200);
    assert!(body.contains("everserve_changes_total 3"), "{body}");
    let (status, body) = http_get(addr, "/healthz").await;
    assert_eq!((status, body.as_str()), (200, "ok"));
    t.set_healthy(false);
    let (status, _) = http_get(addr, "/healthz").await;
    assert_eq!(status, 503);

    cancel.cancel();
    let closed = async {
        while tokio::net::TcpStream::connect(addr).await.is_ok() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), closed)
        .await
        .expect("the listener closes once cancelled");
}

#[tokio::test]
async fn a_taken_port_fails_the_start() {
    let taken = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let started = Telemetry::new()
        .start(
            taken.local_addr().expect("addr"),
            everserve_telemetry::HEADER_READ,
            CancellationToken::new(),
        )
        .await;
    assert!(started.is_err(), "started on a taken port");
}

#[tokio::test]
async fn a_connection_that_sends_nothing_is_closed() {
    // a caller holding connections open by sending nothing would hold the process's file
    // descriptors, and with them its operator API and its links
    let cancel = CancellationToken::new();
    let addr = Telemetry::new()
        .start(
            "127.0.0.1:0".parse().unwrap(),
            std::time::Duration::from_millis(300),
            cancel.clone(),
        )
        .await
        .expect("bound");
    let mut silent = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let mut got = Vec::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        silent.read_to_end(&mut got),
    )
    .await
    .expect("the endpoint closed the connection in time")
    .expect("read");
    // a request still answered after
    let (status, _) = http_get(addr, "/healthz").await;
    assert_eq!(status, 200);
    cancel.cancel();
}
