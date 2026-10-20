# everserve-wire

The generated schema types (`proto::everserve::{signals,mirror,policy,discovery,spec,coord}::v1`,
one module per package in `proto/`) and what every Everserve process speaks over them:

- `frame`: the frame protocol, a 4-byte big-endian length, then one protobuf message, with a cap
  on read and write and a reader that keeps a half-read frame across a cancelled read. Identical
  to `core/everserve/wire.py`.
- `transport`: the two transports, a Unix socket on a node and TCP under mutual TLS across nodes,
  behind one stream type, with a connect and handshake timeout, `TCP_NODELAY`, jittered
  reconnect backoff and an orderly TLS close.
- `host`: what a node's processes share on the host: the pod contract's directories belong to
  the node's group, mode 2770, and a process makes its files there the group's.
- `proto`: the schema, generated at build time by `build.rs` with `protox` (no system protoc) and
  `pbjson` for the protobuf JSON mapping. Do not edit; edit the `.proto` files.

Depend on this crate; never copy the framing.
