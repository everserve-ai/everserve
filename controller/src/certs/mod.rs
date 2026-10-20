// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! A deployment's TLS material, for an operator without a CA: one CA, a certificate for the
//! coordinator carrying the name daemons verify, and one per node carrying the node's name and
//! the name stores verify on a peer store, so a node's daemon and store share one identity.
//!
//! The CA in the output directory is reused when it exists, so nodes are added under it later:
//! the CA's key signs the new certificates under the same issuer name, and `ca.crt` is left as
//! it is, so every peer's copy of it stays valid. A certificate or key that already exists is
//! kept unless `--overwrite` says otherwise, the rule `CockroachDB`'s `cert` command follows, so a
//! rerun never replaces a file a running process holds. Keys are written readable by the owner
//! only. A deployment with its own CA (Kubernetes secrets, the operator's PKI) does not use
//! this; the processes take any PEM that chains.

use std::path::{Path, PathBuf};

use crate::config::CertsArgs;

/// Why the material could not be written.
#[derive(Debug, thiserror::Error)]
pub enum CertsError {
    /// A file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// A certificate or key could not be made or parsed.
    #[error("{what}: {reason}")]
    Material {
        /// What was being made.
        what: String,
        /// Why not.
        reason: String,
    },
    /// No node was named.
    #[error("name at least one node with --node or operator with --operator")]
    NoNodes,
}

/// What was written.
#[derive(Debug, Default)]
pub struct Written {
    /// Every file written, in order.
    pub files: Vec<PathBuf>,
    /// Files that existed and were kept (no `--overwrite`).
    pub kept: Vec<PathBuf>,
    /// The CA already existed and signed the new certificates.
    pub ca_reused: bool,
}

/// Write the material `args` describe.
///
/// # Errors
///
/// [`CertsError`] when nothing was named, a file fails, or a certificate cannot be made.
pub fn write(args: &CertsArgs) -> Result<Written, CertsError> {
    if args.nodes.is_empty() && args.operators.is_empty() {
        return Err(CertsError::NoNodes);
    }
    // a node's certificate must never be valid for an operator's name (decision 0039)
    if let Some(node) = args
        .nodes
        .iter()
        .find(|n| n.ends_with(crate::api::OPERATOR_DOMAIN))
    {
        return Err(CertsError::Material {
            what: node.clone(),
            reason: format!(
                "a node may not be named under {}",
                crate::api::OPERATOR_DOMAIN
            ),
        });
    }
    std::fs::create_dir_all(&args.out).map_err(|source| CertsError::Io {
        path: args.out.clone(),
        source,
    })?;
    let mut written = Written::default();
    let (ca, ca_reused) = authority(&args.out, args.days, &mut written)?;
    written.ca_reused = ca_reused;
    leaf(
        &args.out,
        "coordinator",
        std::slice::from_ref(&args.coordinator_name),
        args.days,
        &ca,
        args.overwrite,
        &mut written,
    )?;
    for node in &args.nodes {
        leaf(
            &args.out,
            node,
            &node_names(node, &args.store_name),
            args.days,
            &ca,
            args.overwrite,
            &mut written,
        )?;
    }
    // an operator's or a viewer's certificate for the operator API (decision 0039): valid only
    // for its name in the operators' domain
    for operator in &args.operators {
        leaf(
            &args.out,
            &format!("operator-{operator}"),
            &[crate::api::operator_name(operator)],
            args.days,
            &ca,
            args.overwrite,
            &mut written,
        )?;
    }
    Ok(written)
}

/// The CA's subject; the issuer name every certificate it signs carries.
pub const CA_NAME: &str = "Everserve deployment CA";

/// The CA whose key is `key_pem` and whose subject is `name`, rebuilt to sign with (decision
/// 0037: the coordinator issues node certificates on Kubernetes). A chain verifies by the
/// issuer's name and key, so `name` must be the subject of the `ca.crt` the peers hold:
/// [`CA_NAME`] for a CA this command made.
///
/// # Errors
///
/// The key does not parse.
pub fn authority_from(key_pem: &str, name: &str) -> Result<rcgen::CertifiedKey, CertsError> {
    let mut params =
        rcgen::CertificateParams::new(Vec::<String>::new()).map_err(|e| material("ca", e))?;
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, name);
    let key_pair = rcgen::KeyPair::from_pem(key_pem).map_err(|e| material("ca.key", e))?;
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| material("ca", e))?;
    Ok(rcgen::CertifiedKey { cert, key_pair })
}

/// The names a node's certificate carries (decision 0026): the node's, which the coordinator
/// believes a daemon's hello for, and the one stores verify on a peer store, so a node's daemon
/// and store share one identity.
#[must_use]
pub fn node_names(node: &str, store_name: &str) -> Vec<String> {
    vec![node.to_owned(), store_name.to_owned()]
}

/// A node's public key as its daemon sends it to be certified: an ECDSA P-256 point,
/// uncompressed (65 bytes, the first 0x04).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeKey(Vec<u8>);

impl NodeKey {
    /// The key whose point is `point`.
    ///
    /// # Errors
    ///
    /// `point` is not an uncompressed P-256 point.
    pub fn new(point: Vec<u8>) -> Result<Self, CertsError> {
        if point.len() != 65 || point[0] != 0x04 {
            return Err(material(
                "node key",
                "not an uncompressed ECDSA P-256 point (65 bytes, the first 0x04)",
            ));
        }
        Ok(Self(point))
    }
}

impl rcgen::PublicKeyData for NodeKey {
    fn der_bytes(&self) -> &[u8] {
        &self.0
    }

    fn algorithm(&self) -> &rcgen::SignatureAlgorithm {
        &rcgen::PKCS_ECDSA_P256_SHA256
    }
}

/// `node`'s certificate, for `key`, signed by `ca`, valid from five minutes ago (clock skew) for
/// `lifetime`: its PEM and when it expires.
///
/// # Errors
///
/// The certificate could not be made.
pub fn sign_node(
    ca: &rcgen::CertifiedKey,
    node: &str,
    store_name: &str,
    key: &NodeKey,
    lifetime: std::time::Duration,
) -> Result<(String, time::OffsetDateTime), CertsError> {
    let mut params = rcgen::CertificateParams::new(node_names(node, store_name))
        .map_err(|e| material(node, e))?;
    params.distinguished_name = rcgen::DistinguishedName::new();
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, node);
    let now = time::OffsetDateTime::now_utc();
    params.not_before = now - time::Duration::minutes(5);
    let not_after = now + lifetime;
    params.not_after = not_after;
    let cert = params
        .signed_by(key, &ca.cert, &ca.key_pair)
        .map_err(|e| material(node, e))?;
    Ok((cert.pem(), not_after))
}

/// The CA in `dir`: its key read and its issuer rebuilt when both files exist, made otherwise.
///
/// A reused CA is rebuilt from its key and fixed name rather than parsed from `ca.crt` (that
/// needs an X.509 parser the tree does not carry). A chain verifies by the issuer's name and key,
/// both of which the rebuilt issuer shares with the `ca.crt` the peers hold.
fn authority(
    dir: &Path,
    days: u32,
    written: &mut Written,
) -> Result<(rcgen::CertifiedKey, bool), CertsError> {
    let cert_path = dir.join("ca.crt");
    let key_path = dir.join("ca.key");
    let reuse = cert_path.exists() && key_path.exists();
    let mut params =
        rcgen::CertificateParams::new(Vec::<String>::new()).map_err(|e| material("ca", e))?;
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, CA_NAME);
    set_validity(&mut params, days, time::OffsetDateTime::now_utc());
    let key_pair = if reuse {
        rcgen::KeyPair::from_pem(&read(&key_path)?).map_err(|e| material("ca.key", e))?
    } else {
        rcgen::KeyPair::generate().map_err(|e| material("ca", e))?
    };
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| material("ca", e))?;
    if !reuse {
        write_pem(&cert_path, &cert.pem(), false, written)?;
        write_pem(&key_path, &key_pair.serialize_pem(), true, written)?;
    }
    Ok((rcgen::CertifiedKey { cert, key_pair }, reuse))
}

/// A certificate for `name` with `sans`, signed by `ca`, as `<name>.crt` and `<name>.key`; an
/// existing pair is kept unless `overwrite`.
fn leaf(
    dir: &Path,
    name: &str,
    sans: &[String],
    days: u32,
    ca: &rcgen::CertifiedKey,
    overwrite: bool,
    written: &mut Written,
) -> Result<(), CertsError> {
    let cert_path = dir.join(format!("{name}.crt"));
    let key_path = dir.join(format!("{name}.key"));
    if !overwrite && (cert_path.exists() || key_path.exists()) {
        written.kept.push(cert_path);
        written.kept.push(key_path);
        return Ok(());
    }
    let mut params = rcgen::CertificateParams::new(sans.to_vec()).map_err(|e| material(name, e))?;
    params.distinguished_name = rcgen::DistinguishedName::new();
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, name);
    set_validity(&mut params, days, time::OffsetDateTime::now_utc());
    let key = rcgen::KeyPair::generate().map_err(|e| material(name, e))?;
    let cert = params
        .signed_by(&key, &ca.cert, &ca.key_pair)
        .map_err(|e| material(name, e))?;
    write_pem(&cert_path, &cert.pem(), false, written)?;
    write_pem(&key_path, &key.serialize_pem(), true, written)
}

/// Valid from a day before `now` (a peer's clock may run behind) to `days` days after it, to the
/// second: a whole-date bound at midnight makes a certificate issued at 23:59 for one day expire
/// a minute later.
fn set_validity(params: &mut rcgen::CertificateParams, days: u32, now: time::OffsetDateTime) {
    params.not_before = now - time::Duration::days(1);
    params.not_after = now + time::Duration::days(i64::from(days));
}

fn material(what: &str, e: impl std::fmt::Display) -> CertsError {
    CertsError::Material {
        what: what.to_owned(),
        reason: e.to_string(),
    }
}

fn read(path: &Path) -> Result<String, CertsError> {
    std::fs::read_to_string(path).map_err(|source| CertsError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write `pem` to `path`; a key is readable by its owner only.
fn write_pem(
    path: &Path,
    pem: &str,
    private: bool,
    written: &mut Written,
) -> Result<(), CertsError> {
    put_pem(path, pem, private).map_err(|source| CertsError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    written.files.push(path.to_path_buf());
    Ok(())
}

/// Replace the file at `path` with `pem` in one step: written beside it, then renamed over it,
/// so a process reading it ([`everserve_wire::transport::Tls::reload`]) sees the old file or
/// the new one, never part of either. A key is readable by its owner only.
///
/// # Errors
///
/// The write or the rename.
pub fn replace_pem(path: &Path, pem: &str, private: bool) -> std::io::Result<()> {
    let mut beside = path.as_os_str().to_owned();
    beside.push(".new");
    let beside = std::path::PathBuf::from(beside);
    put_pem(&beside, pem, private)?;
    std::fs::rename(&beside, path)
}

fn put_pem(path: &Path, pem: &str, private: bool) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if private { 0o600 } else { 0o644 });
    }
    let mut file = options.open(path)?;
    std::io::Write::write_all(&mut file, pem.as_bytes())?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::set_validity;

    #[test]
    fn a_certificate_lives_its_days_whatever_the_hour_it_was_made() {
        // made a second before midnight for one day: valid 23 hours later, as at any other hour
        let now = time::macros::datetime!(2026-10-14 23:59:59 UTC);
        let mut params = rcgen::CertificateParams::default();
        set_validity(&mut params, 1, now);
        assert!(
            params.not_after > now + time::Duration::hours(23),
            "{:?}",
            params.not_after
        );
        assert_eq!(params.not_after, now + time::Duration::days(1));
        assert!(params.not_before <= now - time::Duration::hours(23));
    }
}
