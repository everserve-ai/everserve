// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator API's routes (decision 0042), each declared once: its path, its verb, what it
//! takes and what it answers. The router and the published document (`docs/api/openapi.json`)
//! are both made from [`routes`], so the document cannot leave a route or a field out.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{FromRequest, Path, Query, Request, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::mpsc;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use super::Identity;
use super::error::{ApiError, Reason};
use super::feed::{Feed, Kind, Stale, parse};
use super::objects::{
    Engine, EventType, Launcher, List, ListMetadata, Node, NodeRelease, NodeSpec, Pool, PoolSpec,
    StacksRequest, WatchEvent, Write,
};
use crate::coord::{NodeSpecError, NodeWant, PoolSpecError, StacksAt, StacksError, StacksTaken};
use crate::proto::everserve::spec::v1::PoolSpec as StoredPool;
use crate::spec::SpecError;
use crate::status::Event;
use crate::topology::Device;

/// How many lines a watch holds for a client that reads slowly before the watch ends.
const WATCH_BACKLOG: usize = 256;

/// The API's description, beside what the routes declare.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Everserve operator API",
        description = "Engines, nodes and pools, each a spec and a status. Every request is \
                       authenticated and authorized (decision 0039); the objects, their \
                       versions, watches and errors are decision 0042's."
    ),
    components(schemas(ApiError, Reason, WatchEvent, EventType)),
    modifiers(&Credentials),
    security(("bearer" = []), ("clientCertificate" = []))
)]
struct Document;

/// The two credentials a caller may present (decision 0039): a bearer token the cluster's API
/// server reviews, or a client certificate chained to the deployment's CA.
struct Credentials;

impl utoipa::Modify for Credentials {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        use utoipa::openapi::security::{Http, HttpAuthScheme, SecurityScheme};
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "bearer",
            SecurityScheme::Http(
                Http::builder()
                    .scheme(HttpAuthScheme::Bearer)
                    .description(Some(
                        "A Kubernetes token; the cluster's RBAC grants its verbs on the \
                         everserve.ai group's resources",
                    ))
                    .build(),
            ),
        );
        components.add_security_scheme(
            "clientCertificate",
            SecurityScheme::MutualTls {
                description: Some(
                    "A certificate from `everserve-controller certs --operator <name>`, its name \
                     one the coordinator's settings make an operator or a viewer"
                        .to_owned(),
                ),
                deprecated: None,
                extensions: None,
            },
        );
        openapi.info.license = Some(
            utoipa::openapi::LicenseBuilder::new()
                .name("Apache-2.0")
                .url(Some("https://www.apache.org/licenses/LICENSE-2.0"))
                .build(),
        );
    }
}

/// Every route, declared once.
fn routes() -> OpenApiRouter<Arc<Feed>> {
    OpenApiRouter::with_openapi(Document::openapi())
        .routes(routes!(list_engines))
        .routes(routes!(get_engine))
        .routes(routes!(engine_events))
        .routes(routes!(engine_stacks))
        .routes(routes!(list_nodes))
        .routes(routes!(get_node, put_node))
        .routes(routes!(node_events))
        .routes(routes!(node_release))
        .routes(routes!(list_pools))
        .routes(routes!(get_pool, put_pool))
        .routes(routes!(get_launcher, put_launcher))
        .routes(routes!(pool_events))
        .routes(routes!(topology))
}

/// The routes over `feed`'s objects; a path no route serves is answered as the API's error.
pub fn router(feed: Arc<Feed>) -> axum::Router {
    let (router, _) = routes().with_state(feed).split_for_parts();
    router.fallback(|| async { ApiError::new(Reason::NotFound, "no such route") })
}

/// The API's description, as `docs/api/openapi.json` publishes it.
#[must_use]
pub fn document() -> utoipa::openapi::OpenApi {
    routes().into_openapi()
}

/// A JSON request body, its rejection answered as the API's error.
pub struct JsonBody<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for JsonBody<T> {
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<T>::from_request(req, state)
            .await
            .map(|Json(t)| Self(t))
            .map_err(|r| ApiError::new(Reason::Invalid, r.body_text()))
    }
}

/// A list's query: `?watch=true&resourceVersion=N` streams the changes after N;
/// `?limit=N&continue=T` reads the list a page at a time.
#[derive(Debug, Default, Deserialize, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
#[into_params(parameter_in = Query)]
pub struct ListQuery {
    /// Stream changes, one JSON line each, rather than list.
    #[serde(default)]
    pub watch: bool,
    /// For a watch: the version to stream from. Absent, every object is sent as `ADDED` first.
    #[serde(default)]
    pub resource_version: Option<String>,
    /// For a list: at most this many objects, with a `continue` token when there are more.
    #[serde(default)]
    pub limit: Option<usize>,
    /// For a list: the token the previous page gave, to read the next one.
    #[serde(default, rename = "continue")]
    #[param(rename = "continue")]
    pub continue_token: Option<String>,
}

/// A list's query, its rejection answered as the API's error.
pub struct ListParams(pub ListQuery);

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for ListParams {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        Query::<ListQuery>::from_request_parts(parts, state)
            .await
            .map(|Query(q)| Self(q))
            .map_err(|r| ApiError::new(Reason::Invalid, r.body_text()))
    }
}

/// Every object of `kind`, or a watch of them.
async fn list_or_watch(feed: Arc<Feed>, kind: Kind, q: ListQuery) -> Response {
    let paged = q.limit.is_some() || q.continue_token.is_some();
    if q.watch {
        if paged {
            return ApiError::new(
                Reason::Invalid,
                "a watch is not paged: drop limit and continue",
            )
            .into_response();
        }
        return watch(feed, kind, q.resource_version).await;
    }
    if !paged {
        let (version, items) = feed.list(kind).await;
        return Json(List::<Value> {
            metadata: ListMetadata {
                resource_version: version,
                continue_token: None,
            },
            items,
        })
        .into_response();
    }
    match feed
        .page(
            kind,
            q.limit.unwrap_or(usize::MAX),
            q.continue_token.as_deref(),
        )
        .await
    {
        Ok(page) => Json(List::<Value> {
            metadata: ListMetadata {
                resource_version: page.version,
                continue_token: page.next,
            },
            items: page.items,
        })
        .into_response(),
        Err(e @ Stale::Gone(_)) => ApiError::new(Reason::Gone, e.to_string()).into_response(),
        Err(e @ Stale::Invalid(_)) => ApiError::new(Reason::Invalid, e.to_string()).into_response(),
    }
}

/// `name`'s object of `kind`, or 404.
async fn one(feed: &Feed, kind: Kind, name: &str) -> Result<Json<Value>, ApiError> {
    feed.get(kind, name)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::new(Reason::NotFound, format!("{name}: no such object")))
}

/// List the engines, or watch them.
#[utoipa::path(get, path = "/v1/engines", tag = "engines", params(ListQuery),
    responses(
        (status = 200, description = "Every engine, or with `watch`, its changes as JSON lines",
         content((List<Engine> = "application/json"), (WatchEvent = "application/x-ndjson"))),
        (status = 410, description = "The watch's version is too old: list again", body = ApiError),
    ))]
async fn list_engines(State(feed): State<Arc<Feed>>, ListParams(q): ListParams) -> Response {
    list_or_watch(feed, Kind::Engines, q).await
}

/// Read an engine.
#[utoipa::path(get, path = "/v1/engines/{engine}", tag = "engines",
    params(("engine" = String, Path, description = "The engine's name in the policy")),
    responses((status = 200, body = Engine), (status = 404, body = ApiError)))]
async fn get_engine(
    State(feed): State<Arc<Feed>>,
    Path(engine): Path<String>,
) -> Result<Json<Value>, ApiError> {
    one(&feed, Kind::Engines, &engine).await
}

/// An engine's events, oldest first.
#[utoipa::path(get, path = "/v1/engines/{engine}/events", tag = "engines",
    params(("engine" = String, Path, description = "The engine's name in the policy")),
    responses((status = 200, body = Vec<Event>), (status = 404, body = ApiError)))]
async fn engine_events(
    State(feed): State<Arc<Feed>>,
    Path(engine): Path<String>,
) -> Result<Json<Vec<Event>>, ApiError> {
    feed.coordinator()
        .engine_events(&engine)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::new(Reason::NotFound, format!("{engine}: no such engine")))
}

/// Ask a process of an engine for its stacks: a serving rank (or the spare preloaded for it or
/// taking it), a spare standing for any position by its node and device, or the head.
#[utoipa::path(post, path = "/v1/engines/{engine}/stacks", tag = "engines",
    params(("engine" = String, Path, description = "The engine's name in the policy")),
    request_body = StacksRequest,
    responses(
        (status = 202, body = StacksTaken),
        (status = 404, body = ApiError),
        (status = 422, body = ApiError),
        (status = 502, description = "The process's node did not answer", body = ApiError),
    ))]
async fn engine_stacks(
    State(feed): State<Arc<Feed>>,
    Path(engine): Path<String>,
    JsonBody(req): JsonBody<StacksRequest>,
) -> Result<(axum::http::StatusCode, Json<StacksTaken>), ApiError> {
    let at = match (req.head, req.rank, req.node, req.device) {
        (false, Some(r), None, None) => StacksAt::Rank(r),
        (false, None, Some(node), Some(device)) => StacksAt::Spare { node, device },
        (true, None, None, None) => StacksAt::Head,
        _ => {
            return Err(ApiError::new(
                Reason::Invalid,
                "a stacks request names a rank, a node and a device, or the head",
            ));
        }
    };
    match feed.coordinator().stacks(&engine, at).await {
        Ok(t) => Ok((axum::http::StatusCode::ACCEPTED, Json(t))),
        Err(e @ StacksError::NotFound(_)) => Err(ApiError::new(Reason::NotFound, e.to_string())),
        Err(e @ StacksError::Node { .. }) => Err(ApiError::upstream(e.to_string())),
    }
}

/// List the nodes, or watch them: every node whose daemon is linked, that has a spec, or that
/// holds a member of an engine.
#[utoipa::path(get, path = "/v1/nodes", tag = "nodes", params(ListQuery),
    responses(
        (status = 200, description = "Every node, or with `watch`, its changes as JSON lines",
         content((List<Node> = "application/json"), (WatchEvent = "application/x-ndjson"))),
        (status = 410, description = "The watch's version is too old: list again", body = ApiError),
    ))]
async fn list_nodes(State(feed): State<Arc<Feed>>, ListParams(q): ListParams) -> Response {
    list_or_watch(feed, Kind::Nodes, q).await
}

/// Read a node.
#[utoipa::path(get, path = "/v1/nodes/{node}", tag = "nodes",
    params(("node" = String, Path, description = "The node's name")),
    responses((status = 200, body = Node), (status = 404, body = ApiError)))]
async fn get_node(
    State(feed): State<Arc<Feed>>,
    Path(node): Path<String>,
) -> Result<Json<Value>, ApiError> {
    one(&feed, Kind::Nodes, &node).await
}

/// Write a node's spec: cordon it, drain it by a deadline, or withdraw either. What came of it
/// is the node's status: its Draining condition says whether the drain is `Scheduled` or
/// `Waiting` its turn.
#[utoipa::path(put, path = "/v1/nodes/{node}", tag = "nodes",
    params(("node" = String, Path, description = "The node's name")),
    request_body = Write<NodeSpec>,
    responses(
        (status = 200, description = "The node after the write", body = Node),
        (status = 409, description = "`Conflict`: the node changed since the version written \
            against; `DrainUnderway`: the write moves or clears a drain whose notices went out",
         body = ApiError),
        (status = 422, body = ApiError),
        (status = 503, body = ApiError),
    ))]
async fn put_node(
    State(feed): State<Arc<Feed>>,
    axum::Extension(who): axum::Extension<Identity>,
    Path(node): Path<String>,
    JsonBody(write): JsonBody<Write<NodeSpec>>,
) -> Result<Json<Value>, ApiError> {
    let current = feed.get(Kind::Nodes, &node).await;
    let expected = precondition(current.as_ref(), write.metadata.resource_version.as_deref())?;
    let deadline = match write.spec.drain {
        Some(d) if d.deadline_ms <= 0 => {
            return Err(ApiError::new(
                Reason::Invalid,
                "a drain's deadlineMs is a time, Unix milliseconds",
            ));
        }
        Some(d) => d.deadline_ms,
        None => 0,
    };
    let want = NodeWant {
        unschedulable: write.spec.unschedulable,
        drain_by_ms: deadline,
        reason: write.spec.reason,
    };
    feed.coordinator()
        .put_node_spec(&node, &who.writer(), want, expected)
        .await
        .map_err(|e| node_error(&e))?;
    one(&feed, Kind::Nodes, &node).await
}

/// A node's events, oldest first.
#[utoipa::path(get, path = "/v1/nodes/{node}/events", tag = "nodes",
    params(("node" = String, Path, description = "The node's name")),
    responses((status = 200, body = Vec<Event>)))]
async fn node_events(State(feed): State<Arc<Feed>>, Path(node): Path<String>) -> Json<Vec<Event>> {
    Json(feed.coordinator().node_events(&node).await)
}

/// The longest a release question waits, below the request deadline every route answers within.
const RELEASE_WAIT_MOST_MS: u64 = 25_000;

/// A release question's query: `?wait_ms=N` answers once the node may be given back, or after N
/// milliseconds (at most 25 s), whichever comes first.
#[derive(Debug, Default, Deserialize, utoipa::IntoParams)]
#[serde(rename_all = "camelCase")]
#[into_params(parameter_in = Query)]
pub struct ReleaseQuery {
    /// How long to wait for the node to become releasable, in milliseconds; absent, answer now.
    #[serde(default, rename = "wait_ms")]
    #[param(rename = "wait_ms")]
    pub wait_ms: Option<u64>,
}

/// Whether a node may be given back now, and why not; with `wait_ms`, once it may
/// or when the wait runs out.
#[utoipa::path(get, path = "/v1/nodes/{node}/release", tag = "nodes",
    params(("node" = String, Path, description = "The node's name"), ReleaseQuery),
    responses((status = 200, body = NodeRelease)))]
async fn node_release(
    State(feed): State<Arc<Feed>>,
    Path(node): Path<String>,
    Query(q): Query<ReleaseQuery>,
) -> Json<NodeRelease> {
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_millis(q.wait_ms.unwrap_or(0).min(RELEASE_WAIT_MOST_MS));
    loop {
        let why = feed.coordinator().release_blockers(&node).await;
        if why.is_empty() || tokio::time::Instant::now() >= deadline {
            return Json(NodeRelease {
                node,
                releasable: why.is_empty(),
                why,
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// List the pools, or watch them.
#[utoipa::path(get, path = "/v1/pools", tag = "pools", params(ListQuery),
    responses(
        (status = 200, description = "Every pool, or with `watch`, its changes as JSON lines",
         content((List<Pool> = "application/json"), (WatchEvent = "application/x-ndjson"))),
        (status = 410, description = "The watch's version is too old: list again", body = ApiError),
    ))]
async fn list_pools(State(feed): State<Arc<Feed>>, ListParams(q): ListParams) -> Response {
    list_or_watch(feed, Kind::Pools, q).await
}

/// Read a pool.
#[utoipa::path(get, path = "/v1/pools/{pool}", tag = "pools",
    params(("pool" = String, Path, description = "The pool's name")),
    responses((status = 200, body = Pool), (status = 404, body = ApiError)))]
async fn get_pool(
    State(feed): State<Arc<Feed>>,
    Path(pool): Path<String>,
) -> Result<Json<Value>, ApiError> {
    one(&feed, Kind::Pools, &pool).await
}

/// Write a pool's spec; its launcher is kept as it is.
#[utoipa::path(put, path = "/v1/pools/{pool}", tag = "pools",
    params(("pool" = String, Path, description = "The pool's name")),
    request_body = Write<PoolSpec>,
    responses(
        (status = 200, description = "The pool after the write", body = Pool),
        (status = 409, description = "The pool changed since the version written against",
         body = ApiError),
        (status = 422, body = ApiError),
        (status = 503, body = ApiError),
    ))]
async fn put_pool(
    State(feed): State<Arc<Feed>>,
    axum::Extension(who): axum::Extension<Identity>,
    Path(pool): Path<String>,
    JsonBody(write): JsonBody<Write<PoolSpec>>,
) -> Result<Json<Value>, ApiError> {
    let current = feed.get(Kind::Pools, &pool).await;
    let precondition = precondition(current.as_ref(), write.metadata.resource_version.as_deref())?;
    let held = feed.coordinator().pool_spec(&pool).await;
    // the launcher is kept: written over the generation it was read at, or refused
    let (generation, launcher) = held.map_or((0, StoredPool::default()), |(g, s)| (g, s));
    let spec = StoredPool {
        name: pool.clone(),
        engines: write.spec.engines,
        wants: write.spec.wants,
        device_class: write.spec.device_class,
        domain: write.spec.domain,
        reason: write.spec.reason,
        command: launcher.command,
        env: launcher.env,
    };
    write_pool(&feed, &who, spec, Some(precondition.unwrap_or(generation))).await?;
    one(&feed, Kind::Pools, &pool).await
}

/// Read how a pool's spares are launched.
#[utoipa::path(get, path = "/v1/pools/{pool}/launcher", tag = "pools",
    params(("pool" = String, Path, description = "The pool's name")),
    responses((status = 200, body = Launcher), (status = 404, body = ApiError)))]
async fn get_launcher(
    State(feed): State<Arc<Feed>>,
    Path(pool): Path<String>,
) -> Result<Json<Launcher>, ApiError> {
    let (_, spec) = feed
        .coordinator()
        .pool_spec(&pool)
        .await
        .ok_or_else(|| ApiError::new(Reason::NotFound, format!("{pool}: no such pool")))?;
    Ok(Json(Launcher {
        command: spec.command,
        env: spec.env.into_iter().collect(),
    }))
}

/// Write how a node's daemon launches one of the pool's spares: a command every node of the
/// pool runs, written under a grant of its own (`update pools/launcher`).
#[utoipa::path(put, path = "/v1/pools/{pool}/launcher", tag = "pools",
    params(("pool" = String, Path, description = "The pool's name")),
    request_body = Launcher,
    responses(
        (status = 200, description = "The launcher after the write", body = Launcher),
        (status = 404, body = ApiError),
        (status = 409, description = "The pool's spec was written meanwhile: write again",
         body = ApiError),
        (status = 503, body = ApiError),
    ))]
async fn put_launcher(
    State(feed): State<Arc<Feed>>,
    axum::Extension(who): axum::Extension<Identity>,
    Path(pool): Path<String>,
    JsonBody(launcher): JsonBody<Launcher>,
) -> Result<Json<Launcher>, ApiError> {
    let (generation, held) = feed
        .coordinator()
        .pool_spec(&pool)
        .await
        .ok_or_else(|| ApiError::new(Reason::NotFound, format!("{pool}: no such pool")))?;
    let spec = StoredPool {
        command: launcher.command.clone(),
        env: launcher.env.clone().into_iter().collect(),
        ..held
    };
    write_pool(&feed, &who, spec, Some(generation)).await?;
    Ok(Json(launcher))
}

/// A pool's events, oldest first.
#[utoipa::path(get, path = "/v1/pools/{pool}/events", tag = "pools",
    params(("pool" = String, Path, description = "The pool's name")),
    responses((status = 200, body = Vec<Event>)))]
async fn pool_events(State(feed): State<Arc<Feed>>, Path(pool): Path<String>) -> Json<Vec<Event>> {
    Json(feed.coordinator().pool_events(&pool).await)
}

/// Every device the daemons reported.
#[utoipa::path(get, path = "/v1/topology", tag = "nodes",
    responses((status = 200, body = Vec<Device>)))]
async fn topology(State(feed): State<Arc<Feed>>) -> Json<Vec<Device>> {
    Json(
        feed.coordinator()
            .topology()
            .await
            .devices()
            .cloned()
            .collect(),
    )
}

/// The spec generation a write carrying `version` is made against: None without one; 409 when
/// the object is not at that version.
fn precondition(current: Option<&Value>, version: Option<&str>) -> Result<Option<u64>, ApiError> {
    let Some(version) = version else {
        return Ok(None);
    };
    let held = current.and_then(|c| c["metadata"]["resourceVersion"].as_str());
    if held != Some(version) {
        return Err(ApiError::new(
            Reason::Conflict,
            format!(
                "written against resourceVersion {version}; the object is at {}",
                held.unwrap_or("none")
            ),
        ));
    }
    Ok(current.and_then(|c| c["metadata"]["generation"].as_u64()))
}

async fn write_pool(
    feed: &Feed,
    who: &Identity,
    spec: StoredPool,
    expected: Option<u64>,
) -> Result<(), ApiError> {
    feed.coordinator()
        .write_pool_spec(&who.writer(), spec, expected)
        .await
        .map(drop)
        .map_err(|e| match e {
            PoolSpecError::UnknownEngine(_) | PoolSpecError::Shape(_) => {
                ApiError::new(Reason::Invalid, e.to_string())
            }
            PoolSpecError::Stale { .. } => ApiError::new(Reason::Conflict, e.to_string()),
            PoolSpecError::Store(s) => store_error(&s),
        })
}

fn node_error(e: &NodeSpecError) -> ApiError {
    match e {
        NodeSpecError::DrainUnderway(_) => ApiError::new(Reason::DrainUnderway, e.to_string()),
        NodeSpecError::Stale { .. } => ApiError::new(Reason::Conflict, e.to_string()),
        NodeSpecError::NoMembers(_) | NodeSpecError::Invalid(_) => {
            ApiError::new(Reason::Invalid, e.to_string())
        }
        NodeSpecError::Store(s) => store_error(s),
    }
}

/// A refused write is a conflict the caller resolves by reading again; one fenced or deposed is
/// another coordinator's to take; a failed store is ours.
fn store_error(e: &SpecError) -> ApiError {
    match e {
        SpecError::Conflict { .. } => ApiError::new(Reason::Conflict, e.to_string()),
        SpecError::Fenced { .. } | SpecError::Deposed { .. } => {
            ApiError::new(Reason::Unavailable, e.to_string())
        }
        _ => ApiError::new(Reason::Internal, e.to_string()),
    }
}

/// A watch of `kind` from `from`: the changes after it, one JSON line each, a bookmark every
/// [`super::feed::Pacing::bookmark`], until the client goes, the coordinator's term ends, or it stops. Without
/// a version, every object is sent as `ADDED` first.
async fn watch(feed: Arc<Feed>, kind: Kind, from: Option<String>) -> Response {
    let (first, start) = match from {
        None => {
            let (version, items) = feed.list(kind).await;
            let added = items
                .into_iter()
                .map(|object| WatchEvent {
                    kind: EventType::Added,
                    object,
                })
                .collect();
            (added, version)
        }
        Some(version) => {
            feed.refresh().await;
            match feed.since(kind, &version).await {
                Ok((events, last)) => (events, feed.version(last).await),
                Err(e @ Stale::Gone(_)) => {
                    return ApiError::new(Reason::Gone, e.to_string()).into_response();
                }
                Err(e @ Stale::Invalid(_)) => {
                    return ApiError::new(Reason::Invalid, e.to_string()).into_response();
                }
            }
        }
    };
    let Some((term, mut at)) = parse(&start) else {
        return ApiError::new(Reason::Internal, "the feed gave no version").into_response();
    };
    let (tx, rx) = mpsc::channel::<Bytes>(WATCH_BACKLOG);
    let stopped = feed.stopped().clone();
    tokio::spawn(async move {
        let mut changes = feed.changes();
        let mut bookmark = tokio::time::interval(feed.pacing().bookmark);
        bookmark.tick().await;
        let mut pending = first;
        loop {
            for event in pending.drain(..) {
                if tx.try_send(line(&event)).is_err() {
                    // gone, or reading too slowly to keep up: it lists again
                    return;
                }
            }
            tokio::select! {
                () = stopped.cancelled() => return,
                () = tx.closed() => return,
                _ = bookmark.tick() => {
                    let version = feed.version(at).await;
                    pending.push(WatchEvent {
                        kind: EventType::Bookmark,
                        object: serde_json::json!({"metadata": {"resourceVersion": version}}),
                    });
                }
                moved = changes.changed() => {
                    if moved.is_err() {
                        return;
                    }
                    let (now, _) = *changes.borrow_and_update();
                    if now != term {
                        return;
                    }
                    let version = feed.version(at).await;
                    match feed.since(kind, &version).await {
                        Ok((events, last)) => {
                            pending = events;
                            at = last;
                        }
                        Err(_) => return,
                    }
                }
            }
        }
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(Lines(rx)))
        .unwrap_or_else(|e| ApiError::new(Reason::Internal, e.to_string()).into_response())
}

/// `event` as one line of a watch.
fn line(event: &WatchEvent) -> Bytes {
    let mut out = serde_json::to_vec(event).unwrap_or_default();
    out.push(b'\n');
    Bytes::from(out)
}

/// A watch's lines, as the response's body streams them.
struct Lines(mpsc::Receiver<Bytes>);

impl futures_core::Stream for Lines {
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|b| b.map(Ok))
    }
}
