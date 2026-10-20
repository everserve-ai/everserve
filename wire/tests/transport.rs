// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The two transports carry the same frames: a Unix socket, and TCP under mutual TLS where a peer
//! without a certificate from the deployment's CA is refused.

use std::path::Path;

use everserve_wire::frame;
use everserve_wire::proto::everserve::signals::v1::{Ack, Member, Register, Role, Signal, signal};
use everserve_wire::transport::{Address, Listener, Peer, Tls, connect};

fn register() -> Signal {
    Signal {
        kind: Some(signal::Kind::Register(Register {
            protocol: everserve_wire::PROTOCOL_VERSION,
            protocol_oldest: everserve_wire::PROTOCOL_OLDEST,
            engine: "e".into(),
            rank: 3,
            member: Some(Member {
                node: "n".into(),
                device: 1,
            }),
            role: Role::Worker as i32,
            slot: None,
            compile_cache: None,
            any_rank: false,
            launcher_pid: 0,
            identity: None,
            stage: Vec::new(),
            host: String::new(),
        })),
    }
}

/// A CA, and a leaf certificate for `name` signed by it, as PEM files in `dir`; returns the
/// paths (cert, key, ca).
fn material(
    dir: &Path,
    ca: &rcgen::CertifiedKey,
    name: &str,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let mut params = rcgen::CertificateParams::new(vec![name.to_owned()]).expect("params");
    params.distinguished_name = rcgen::DistinguishedName::new();
    let key = rcgen::KeyPair::generate().expect("key");
    let cert = params
        .signed_by(&key, &ca.cert, &ca.key_pair)
        .expect("sign");
    let cert_path = dir.join(format!("{name}.crt"));
    let key_path = dir.join(format!("{name}.key"));
    let ca_path = dir.join("ca.crt");
    std::fs::write(&cert_path, cert.pem()).expect("cert");
    std::fs::write(&key_path, key.serialize_pem()).expect("key");
    std::fs::write(&ca_path, ca.cert.pem()).expect("ca");
    (cert_path, key_path, ca_path)
}

fn authority(name: &str) -> rcgen::CertifiedKey {
    let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).expect("params");
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, name);
    let key_pair = rcgen::KeyPair::generate().expect("key");
    let cert = params.self_signed(&key_pair).expect("ca");
    rcgen::CertifiedKey { cert, key_pair }
}

async fn round_trip(server: Listener, client_tls: Option<Tls>) {
    let address = server.local_address().expect("address");
    let serve = tokio::spawn(async move {
        let mut s = server.accept().await.expect("accept").stream;
        let got: Signal = frame::recv(&mut s).await.expect("recv");
        assert_eq!(got, register());
        frame::send(
            &mut s,
            &Ack {
                rank: 3,
                epoch: 7,
                error: String::new(),
                received_at_us: 0,
            },
        )
        .await
        .expect("send");
    });
    let mut c = connect(&address, client_tls.as_ref(), "coordinator")
        .await
        .expect("connect");
    frame::send(&mut c, &register()).await.expect("send");
    let ack: Ack = frame::recv(&mut c).await.expect("recv");
    assert_eq!((ack.rank, ack.epoch), (3, 7));
    serve.await.expect("join");
}

#[tokio::test]
async fn a_unix_socket_carries_the_frames() {
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("t.sock"));
    let server = Listener::bind(&address, None).await.expect("bind");
    round_trip(server, None).await;
}

#[tokio::test]
async fn a_listener_dropped_after_another_bound_its_path_leaves_the_others_socket() {
    // a coordinator or a daemon started again binds the path before the old one lets go: the old
    // one's drop must not remove the socket the new one listens on
    let dir = tempfile::tempdir().expect("tmp");
    let address = Address::Unix(dir.path().join("coord.sock"));
    let old = Listener::bind(&address, None).await.expect("old");
    let new = Listener::bind(&address, None).await.expect("new");
    drop(old);
    let accepted = tokio::spawn(async move { new.accept().await.map(|_| ()) });
    connect(&address, None, "coordinator")
        .await
        .expect("the new listener's socket is still there");
    accepted.await.expect("join").expect("accepted");
}

#[tokio::test]
async fn tcp_carries_the_frames_under_mutual_tls_and_refuses_a_stranger() {
    let dir = tempfile::tempdir().expect("tmp");
    let ca = authority("everserve test ca");
    let (sc, sk, sca) = material(dir.path(), &ca, "coordinator");
    let (cc, ck, cca) = material(dir.path(), &ca, "node-a");
    let server_tls = Tls::load(&sc, &sk, &sca).expect("server tls");
    let client_tls = Tls::load(&cc, &ck, &cca).expect("client tls");
    let address = Address::parse("127.0.0.1:0").expect("address");
    let server = Listener::bind(&address, Some(&server_tls))
        .await
        .expect("bind");
    round_trip(server, Some(client_tls)).await;
}

#[tokio::test]
async fn a_peer_from_another_authority_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let ca = authority("everserve test ca");
    let (sc, sk, sca) = material(dir.path(), &ca, "coordinator");
    let server_tls = Tls::load(&sc, &sk, &sca).expect("server tls");
    let other_dir = dir.path().join("other");
    std::fs::create_dir_all(&other_dir).expect("dir");
    let other = authority("someone else");
    let (xc, xk, _) = material(&other_dir, &other, "intruder");
    // the intruder trusts our CA (so it will talk to us) but its own cert is not from our CA
    let intruder_tls = Tls::load(&xc, &xk, &sca).expect("intruder tls");
    let address = Address::parse("127.0.0.1:0").expect("address");
    let server = Listener::bind(&address, Some(&server_tls))
        .await
        .expect("bind");
    let bound = server.local_address().expect("address");
    let serve = tokio::spawn(async move {
        let accepted = server.accept().await;
        assert!(
            accepted.is_err(),
            "the handshake must fail for a client the CA did not sign"
        );
    });
    let attempt = connect(&bound, Some(&intruder_tls), "coordinator").await;
    if let Ok(mut s) = attempt {
        // some stacks surface the rejection on the first read rather than at connect
        let r: Result<Ack, _> = frame::recv(&mut s).await;
        assert!(r.is_err());
    }
    serve.await.expect("join");
}

#[tokio::test]
async fn tcp_without_tls_is_refused() {
    let address = Address::parse("127.0.0.1:0").expect("address");
    assert!(Listener::bind(&address, None).await.is_err());
    assert!(connect(&address, None, "x").await.is_err());
}

#[test]
fn addresses_parse_both_ways() {
    assert_eq!(
        Address::parse("unix:/tmp/a.sock").unwrap(),
        Address::Unix("/tmp/a.sock".into())
    );
    assert_eq!(
        Address::parse("/tmp/a.sock").unwrap(),
        Address::Unix("/tmp/a.sock".into())
    );
    assert!(matches!(
        Address::parse("10.0.0.5:9110").unwrap(),
        Address::Tcp(_)
    ));
    assert!(Address::parse("not an address").is_err());
}

#[tokio::test]
async fn an_accepted_tls_peer_is_named_by_its_certificate_and_a_unix_peer_by_itself() {
    let dir = tempfile::tempdir().expect("tmp");
    let ca = authority("everserve test ca");
    let (sc, sk, sca) = material(dir.path(), &ca, "coordinator");
    let (cc, ck, cca) = material(dir.path(), &ca, "node-b");
    let server = Tls::load(&sc, &sk, &sca).expect("server tls");
    let client = Tls::load(&cc, &ck, &cca).expect("client tls");
    let listener = Listener::bind(&Address::parse("127.0.0.1:0").unwrap(), Some(&server))
        .await
        .expect("bind");
    let address = listener.local_address().expect("address");
    let accept = tokio::spawn(async move { listener.accept().await.expect("accept").peer });
    let _c = connect(&address, Some(&client), "coordinator")
        .await
        .expect("connect");
    let peer = accept.await.expect("join");
    assert!(matches!(peer, Peer::Certified(_)));
    assert!(peer.is_named("node-b"), "the certificate is for node-b");
    assert!(!peer.is_named("node-c"), "and for nobody else");
    assert!(!peer.is_named("not a dns name!"));

    let sock = dir.path().join("u.sock");
    let listener = Listener::bind(&Address::Unix(sock.clone()), None)
        .await
        .expect("bind");
    let accept = tokio::spawn(async move { listener.accept().await.expect("accept").peer });
    let _c = connect(&Address::Unix(sock), None, "")
        .await
        .expect("connect");
    let peer = accept.await.expect("join");
    assert!(matches!(peer, Peer::Local));
    assert!(peer.is_named("anything"), "a local peer is whoever it says");
}

/// A leaf certificate for `name` from `ca`, with a key of its own, written over `cert` and `key`.
fn renew(cert: &Path, key: &Path, ca: &rcgen::CertifiedKey, name: &str) {
    let mut params = rcgen::CertificateParams::new(vec![name.to_owned()]).expect("params");
    params.distinguished_name = rcgen::DistinguishedName::new();
    let pair = rcgen::KeyPair::generate().expect("key");
    let signed = params
        .signed_by(&pair, &ca.cert, &ca.key_pair)
        .expect("sign");
    std::fs::write(cert, signed.pem()).expect("cert");
    std::fs::write(key, pair.serialize_pem()).expect("key");
}

/// A server on `listener` that takes every connection it can, until dropped.
fn serve_all(listener: Listener) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let _ = listener.accept().await;
        }
    })
}

#[tokio::test]
async fn a_listener_already_bound_presents_the_certificate_renewed_on_disk() {
    let dir = tempfile::tempdir().expect("tmp");
    let ca = authority("everserve test ca");
    let (sc, sk, sca) = material(dir.path(), &ca, "node-a");
    let (cc, ck, cca) = material(dir.path(), &ca, "node-b");
    let server_tls = Tls::load(&sc, &sk, &sca).expect("server tls");
    let client = Tls::load(&cc, &ck, &cca).expect("client tls");
    let server = Listener::bind(
        &Address::parse("127.0.0.1:0").expect("address"),
        Some(&server_tls),
    )
    .await
    .expect("bind");
    let bound = server.local_address().expect("address");
    let serving = serve_all(server);
    assert!(connect(&bound, Some(&client), "node-a").await.is_ok());

    renew(&sc, &sk, &ca, "node-a-renewed");
    assert!(
        connect(&bound, Some(&client), "node-a").await.is_ok(),
        "a renewal on disk is not in use until it is read"
    );
    assert!(server_tls.reload().expect("reload"), "the files changed");
    assert!(
        connect(&bound, Some(&client), "node-a-renewed")
            .await
            .is_ok(),
        "the next handshake presents the renewed certificate"
    );
    assert!(connect(&bound, Some(&client), "node-a").await.is_err());
    assert!(
        !server_tls.reload().expect("reload"),
        "the files did not change again"
    );
    serving.abort();
}

#[tokio::test]
async fn a_key_from_another_renewal_is_refused_and_the_pair_in_use_stays() {
    let dir = tempfile::tempdir().expect("tmp");
    let ca = authority("everserve test ca");
    let (sc, sk, sca) = material(dir.path(), &ca, "node-a");
    let (cc, ck, cca) = material(dir.path(), &ca, "node-b");
    let server_tls = Tls::load(&sc, &sk, &sca).expect("server tls");
    let client = Tls::load(&cc, &ck, &cca).expect("client tls");
    let server = Listener::bind(
        &Address::parse("127.0.0.1:0").expect("address"),
        Some(&server_tls),
    )
    .await
    .expect("bind");
    let bound = server.local_address().expect("address");
    let serving = serve_all(server);

    // the new certificate written, its key not yet: read between the two writes
    let old_key = std::fs::read(&sk).expect("key");
    renew(&sc, &sk, &ca, "node-a-renewed");
    std::fs::write(&sk, &old_key).expect("old key");
    assert!(
        server_tls.reload().is_err(),
        "a certificate and another's key load as nothing"
    );
    assert!(
        connect(&bound, Some(&client), "node-a").await.is_ok(),
        "the pair in use stays"
    );
    assert!(
        Tls::load(&sc, &sk, &sca).is_err(),
        "nor does a process start on them"
    );
    serving.abort();
}
