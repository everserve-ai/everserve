// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The shared Kubernetes API access: every request carries the token its file holds then, since
//! the kubelet rotates a projected service account token within the hour.

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::http::HeaderMap;
use axum::routing::get;
use everserve_controller::kubernetes::Access;

#[tokio::test]
async fn every_request_carries_the_token_its_file_holds_then() {
    let dir = tempfile::tempdir().expect("tmp");
    let token_file = dir.path().join("token");
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let app = Router::new().route(
        "/api/v1/nodes/n",
        get({
            let seen = Arc::clone(&seen);
            move |headers: HeaderMap| async move {
                let bearer = headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_owned();
                seen.lock().expect("lock").push(bearer);
                "{}"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await });
    let access = Access {
        endpoint: format!("http://{addr}"),
        token_file: token_file.clone(),
        ca_file: dir.path().join("ca.crt"),
    };
    let client = access.client().expect("client");

    std::fs::write(&token_file, "first\n").expect("token");
    let a = client
        .get(&access.url("/api/v1/nodes/n"), &[])
        .await
        .expect("get");
    // the kubelet rotates the token in place
    std::fs::write(&token_file, "second\n").expect("token");
    let b = client
        .get(&access.url("/api/v1/nodes/n"), &[])
        .await
        .expect("get");
    assert_eq!((a.status, b.status), (200, 200));
    assert_eq!(
        *seen.lock().expect("lock"),
        vec!["Bearer first".to_owned(), "Bearer second".to_owned()],
        "the second request carries the rotated token"
    );
}
