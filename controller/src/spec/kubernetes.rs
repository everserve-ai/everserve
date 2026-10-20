// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The spec store on Kubernetes (decision 0036, amended 2026-10-13): `EverserveState` objects in
//! the coordinator's namespace, one root and one per engine, every write a replace guarded by the
//! `resourceVersion` this coordinator last read or wrote.
//!
//! The root holds the newest term, the specs (nodes, pools) and the names of the engines' objects;
//! an engine's object holds its status. Each holds its part as a `SpecSnapshot` in protobuf JSON,
//! readable with `kubectl get everservestates -o yaml`, beside the term that last took or wrote
//! it. A coordinator takes the store ([`SpecStore::take`]) by replacing every object with its
//! term, guarded by the version it read, and acts on what it read. Every later write is guarded by
//! the version it holds, so a conflict means another writer changed the object since this
//! coordinator took it, which only a newer holder of the lease does: the write is refused as
//! [`SpecError::Superseded`], and the coordinator stops acting.
//!
//! A new engine's object is registered on the root first, by a guarded replace a deposed holder
//! can no longer make, and created after; a taker creates a registered object that does not exist
//! yet, so the deposed holder's create of it fails. No object is written that a taker could miss.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::kubernetes::Access;
use crate::kubernetes::objects::{Objects, Put};
use crate::lease::BoxFuture;
use crate::proto::everserve::spec::v1::{ResourceSpec, ResourceStatus, SpecSnapshot};

use super::{SpecError, SpecStore, apply, apply_status, fence};

/// The objects' API group and version.
pub const API: &str = "everserve.ai/v1alpha1";
/// Their resource.
pub const PLURAL: &str = "everservestates";
/// Their kind.
pub const KIND: &str = "EverserveState";
/// The label naming the store an object is part of.
pub const STORE_LABEL: &str = "everserve.ai/store";
/// The annotation naming an engine object's engine.
pub const ENGINE_ANNOTATION: &str = "everserve.ai/engine";
/// The most an object's body may be: etcd keeps an object of at most 1.5 MiB, and the API server
/// adds its own fields.
pub const MOST_OBJECT_BYTES: usize = 1024 * 1024;
/// How many times a take reads an object again after another write landed between its read and
/// its replace: a deposed holder's writes before it noticed, which end.
const TAKE_TRIES: usize = 16;

/// One object as this coordinator last read or wrote it.
#[derive(Debug)]
struct Shard {
    /// The object, `resourceVersion` and all.
    object: Value,
    /// What it holds.
    snapshot: SpecSnapshot,
    /// The engines' objects it registers; the root's alone.
    engines: Vec<String>,
}

/// The spec store on Kubernetes.
pub struct KubernetesStore {
    objects: Objects,
    namespace: String,
    name: String,
    root: Mutex<Option<Shard>>,
    engines: Mutex<BTreeMap<String, Arc<Mutex<Shard>>>>,
}

impl std::fmt::Debug for KubernetesStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KubernetesStore")
            .field("namespace", &self.namespace)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl KubernetesStore {
    /// The store named `name` in `namespace`, reached by `access`, each request bounded by
    /// `timeout`.
    ///
    /// # Errors
    ///
    /// The access's CA cannot be read.
    pub fn new(
        access: &Access,
        namespace: &str,
        name: &str,
        timeout: Duration,
    ) -> Result<Self, SpecError> {
        let objects = Objects::new(access, API, PLURAL, namespace, timeout).map_err(|reason| {
            SpecError::Refused {
                key: format!("{namespace}/{name}"),
                reason,
            }
        })?;
        Ok(Self {
            objects,
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            root: Mutex::new(None),
            engines: Mutex::new(BTreeMap::new()),
        })
    }

    /// The object holding `engine`'s status: its name in hex, so any engine name makes an
    /// object name.
    fn engine_object(&self, engine: &str) -> String {
        format!(
            "{}-engine-{}",
            self.name,
            crate::issuer::hex(engine.as_bytes())
        )
    }

    /// The engine whose status the object `name` holds: [`KubernetesStore::engine_object`]
    /// read back.
    fn engine_named(&self, name: &str) -> Option<String> {
        let hex = name.strip_prefix(&format!("{}-engine-", self.name))?;
        let bytes = crate::issuer::unhex(hex)?;
        String::from_utf8(bytes).ok()
    }

    /// The object `name` holding `snapshot` under `term`, guarded by `version` when it has one.
    fn object(
        &self,
        name: &str,
        engine: Option<&str>,
        snapshot: &SpecSnapshot,
        engines: &[String],
        version: Option<&str>,
    ) -> Result<Value, String> {
        let state = serde_json::to_value(snapshot).map_err(|e| e.to_string())?;
        let mut object = json!({
            "apiVersion": API,
            "kind": KIND,
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": { STORE_LABEL: self.name },
            },
            "spec": { "term": snapshot.term.to_string(), "state": state },
        });
        if let Some(engine) = engine {
            object["metadata"]["annotations"] = json!({ ENGINE_ANNOTATION: engine });
        } else {
            object["spec"]["engines"] = json!(engines);
        }
        if let Some(version) = version {
            object["metadata"]["resourceVersion"] = json!(version);
        }
        let bytes = object.to_string().len();
        if bytes > MOST_OBJECT_BYTES {
            return Err(format!(
                "{name} would be {bytes} bytes, over the {MOST_OBJECT_BYTES} an object may be"
            ));
        }
        Ok(object)
    }

    fn refused(&self, key: &str, reason: impl std::fmt::Display) -> SpecError {
        SpecError::Refused {
            key: key.to_owned(),
            reason: format!("{} {}/{}: {reason}", PLURAL, self.namespace, self.name),
        }
    }

    /// Take the object `name` under `term`: read it, refuse a newer term on it, and replace it
    /// with `term`, guarded by the version read, reading again when another write landed between;
    /// None when it does not exist.
    async fn stamp(
        &self,
        name: &str,
        engine: Option<&str>,
        term: u64,
    ) -> Result<Option<Shard>, SpecError> {
        for _ in 0..TAKE_TRIES {
            let Some(found) = self
                .objects
                .get(name)
                .await
                .map_err(|e| self.refused(name, e))?
            else {
                return Ok(None);
            };
            let (mut snapshot, engines) = read(&found).map_err(|e| self.refused(name, e))?;
            fence(&snapshot, name, term)?;
            snapshot.term = term;
            let object = self
                .object(name, engine, &snapshot, &engines, version(&found))
                .map_err(|e| self.refused(name, e))?;
            match self
                .objects
                .replace(&object)
                .await
                .map_err(|e| self.refused(name, e))?
            {
                Put::Written(object) => {
                    return Ok(Some(Shard {
                        object,
                        snapshot,
                        engines,
                    }));
                }
                // written between this read and this replace: read it again
                Put::Conflict => {}
                Put::Gone => return Ok(None),
            }
        }
        Err(self.refused(
            name,
            format!("written by another {TAKE_TRIES} times while this coordinator took it"),
        ))
    }

    /// Create the object `name` holding `snapshot` and, for the root, `engines`; None when one
    /// of its name exists.
    async fn create(
        &self,
        name: &str,
        engine: Option<&str>,
        snapshot: SpecSnapshot,
        engines: Vec<String>,
    ) -> Result<Option<Shard>, SpecError> {
        let object = self
            .object(name, engine, &snapshot, &engines, None)
            .map_err(|e| self.refused(name, e))?;
        match self
            .objects
            .create(&object)
            .await
            .map_err(|e| self.refused(name, e))?
        {
            Put::Written(object) => Ok(Some(Shard {
                object,
                snapshot,
                engines,
            })),
            Put::Conflict | Put::Gone => Ok(None),
        }
    }

    /// Take the object `name` under `term`, making it if it does not exist.
    async fn take_object(
        &self,
        name: &str,
        engine: Option<&str>,
        term: u64,
    ) -> Result<Shard, SpecError> {
        for _ in 0..TAKE_TRIES {
            if let Some(shard) = self.stamp(name, engine, term).await? {
                return Ok(shard);
            }
            let empty = SpecSnapshot {
                term,
                ..SpecSnapshot::default()
            };
            if let Some(shard) = self.create(name, engine, empty, Vec::new()).await? {
                return Ok(shard);
            }
        }
        Err(self.refused(name, "made and removed while this coordinator took it"))
    }

    /// Replace `shard`'s object with `snapshot` (and, for the root, `engines`), guarded by the
    /// version it holds; a conflict is a newer holder's write.
    async fn write(
        &self,
        key: &str,
        shard: &mut Shard,
        engine: Option<&str>,
        snapshot: SpecSnapshot,
        engines: Vec<String>,
    ) -> Result<(), SpecError> {
        let name = shard.object["metadata"]["name"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let object = self
            .object(&name, engine, &snapshot, &engines, version(&shard.object))
            .map_err(|e| self.refused(key, e))?;
        match self
            .objects
            .replace(&object)
            .await
            .map_err(|e| self.refused(key, e))?
        {
            Put::Written(object) => {
                *shard = Shard {
                    object,
                    snapshot,
                    engines,
                };
                Ok(())
            }
            Put::Conflict | Put::Gone => Err(SpecError::Superseded {
                key: key.to_owned(),
            }),
        }
    }

    /// The object holding `engine`'s status, registered on the root and made under `term` if
    /// this coordinator has none.
    async fn engine_shard(
        &self,
        key: &str,
        engine: &str,
        term: u64,
    ) -> Result<Arc<Mutex<Shard>>, SpecError> {
        if let Some(shard) = self.engines.lock().await.get(engine) {
            return Ok(Arc::clone(shard));
        }
        let name = self.engine_object(engine);
        {
            let mut root = self.root.lock().await;
            let root = root
                .as_mut()
                .ok_or_else(|| self.refused(key, "not taken"))?;
            fence(&root.snapshot, key, term)?;
            // registered first, by a guarded write a deposed holder can no longer make
            if !root.engines.contains(&name) {
                let mut engines = root.engines.clone();
                engines.push(name.clone());
                let snapshot = root.snapshot.clone();
                self.write(key, root, None, snapshot, engines).await?;
            }
        }
        let snapshot = SpecSnapshot {
            term,
            ..SpecSnapshot::default()
        };
        // a holder that took the store since made it: this coordinator was deposed
        let shard = self
            .create(&name, Some(engine), snapshot, Vec::new())
            .await?
            .ok_or_else(|| SpecError::Superseded {
                key: key.to_owned(),
            })?;
        let shard = Arc::new(Mutex::new(shard));
        self.engines
            .lock()
            .await
            .insert(engine.to_owned(), Arc::clone(&shard));
        Ok(shard)
    }
}

/// An object's `resourceVersion`.
fn version(object: &Value) -> Option<&str> {
    object["metadata"]["resourceVersion"].as_str()
}

/// What an object holds: its snapshot, the term on it, and the engines' objects it registers.
fn read(object: &Value) -> Result<(SpecSnapshot, Vec<String>), String> {
    let mut snapshot: SpecSnapshot = serde_json::from_value(object["spec"]["state"].clone())
        .map_err(|e| format!("not a spec snapshot: {e}"))?;
    snapshot.term = object["spec"]["term"]
        .as_str()
        .and_then(|t| t.parse().ok())
        .ok_or("no term")?;
    let engines = object["spec"]["engines"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|e| e.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    Ok((snapshot, engines))
}

/// The engine a status key names.
fn engine_of(key: &str) -> Option<&str> {
    key.strip_prefix("engine/")
}

impl SpecStore for KubernetesStore {
    fn take(&self, term: u64) -> BoxFuture<'_, Result<SpecSnapshot, SpecError>> {
        Box::pin(async move {
            let root = self.take_object(&self.name, None, term).await?;
            let mut taken = SpecSnapshot {
                term,
                resources: root.snapshot.resources.clone(),
                ..SpecSnapshot::default()
            };
            let mut engines = BTreeMap::new();
            for name in &root.engines {
                let engine = self
                    .engine_named(name)
                    .ok_or_else(|| self.refused(name, "not an engine's object name"))?;
                // registered and not yet made, a deposed holder's: made here, so its create fails
                let shard = self.take_object(name, Some(&engine), term).await?;
                taken.statuses.extend(shard.snapshot.statuses.clone());
                engines.insert(engine, Arc::new(Mutex::new(shard)));
            }
            *self.root.lock().await = Some(root);
            *self.engines.lock().await = engines;
            Ok(taken)
        })
    }

    fn put(
        &self,
        key: String,
        spec: ResourceSpec,
        read: u64,
        term: u64,
    ) -> BoxFuture<'_, Result<ResourceSpec, SpecError>> {
        Box::pin(async move {
            let mut root = self.root.lock().await;
            let root = root
                .as_mut()
                .ok_or_else(|| self.refused(&key, "not taken"))?;
            let mut snapshot = root.snapshot.clone();
            let written = apply(&mut snapshot, &key, spec, read, term)?;
            let engines = root.engines.clone();
            self.write(&key, root, None, snapshot, engines).await?;
            Ok(written)
        })
    }

    fn put_status(
        &self,
        key: String,
        status: ResourceStatus,
        term: u64,
    ) -> BoxFuture<'_, Result<(), SpecError>> {
        Box::pin(async move {
            let engine = engine_of(&key)
                .ok_or_else(|| self.refused(&key, "a status is an engine's"))?
                .to_owned();
            let shard = self.engine_shard(&key, &engine, term).await?;
            let mut shard = shard.lock().await;
            let mut snapshot = shard.snapshot.clone();
            apply_status(&mut snapshot, &key, status, term)?;
            self.write(&key, &mut shard, Some(&engine), snapshot, Vec::new())
                .await
        })
    }

    fn describe(&self) -> String {
        format!("kubernetes {PLURAL} {}/{}", self.namespace, self.name)
    }
}
