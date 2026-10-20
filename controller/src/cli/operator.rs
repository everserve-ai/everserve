// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! The operator's commands over the operator API: `get`, `watch`, `cordon`, `uncordon`, `drain`
//! and `undrain`. Each exits 0 when done, 1 when the API refused (its reason printed), 2 when the
//! coordinator could not be reached or the credentials not used.

use std::process::ExitCode;

use serde_json::Value;
use tracing::error;

use crate::api::objects::Node;
use crate::clock::now_ms;
use crate::config::{ApiAccessArgs, DrainArgs, GetArgs, NodeWriteArgs, Output, WatchArgs};
use crate::operator::{Operator, OperatorError, table};

use super::runtime;

/// Run `work` against the operator API `api` names, on a runtime of its own.
fn with_operator<F, Fut>(api: &ApiAccessArgs, work: F) -> ExitCode
where
    F: FnOnce(Operator) -> Fut,
    Fut: std::future::Future<Output = Result<(), OperatorError>>,
{
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    let operator = match Operator::connect(api) {
        Ok(o) => o,
        Err(e) => {
            error!("{e}");
            return ExitCode::from(2);
        }
    };
    match rt.block_on(work(operator)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e @ OperatorError::Refused(_)) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
        Err(e) => {
            error!("{e}");
            ExitCode::from(2)
        }
    }
}

pub(super) fn run_get(args: &GetArgs) -> ExitCode {
    with_operator(&args.api, async |op| {
        let read: Value = match &args.name {
            Some(name) => op.get(&format!("/v1/{}/{name}", args.kind.path())).await?,
            // a page at a time: a cell's thousands of nodes are not one answer
            None => serde_json::to_value(op.list(args.kind).await?).unwrap_or_default(),
        };
        match args.output {
            Output::Json => println!(
                "{}",
                serde_json::to_string_pretty(&read).unwrap_or_default()
            ),
            Output::Table => {
                let objects = match &args.name {
                    Some(_) => vec![read],
                    None => read["items"].as_array().cloned().unwrap_or_default(),
                };
                println!("{}", table(args.kind, &objects));
            }
        }
        Ok(())
    })
}

pub(super) fn run_watch(args: &WatchArgs) -> ExitCode {
    with_operator(&args.api, async |op| {
        let mut changes = op.watch(args.kind).await?;
        while let Some(line) = changes
            .line()
            .await
            .map_err(|e| OperatorError::Unreachable(e.to_string()))?
        {
            match args.output {
                Output::Json => println!("{line}"),
                Output::Table => {
                    let event: Value = serde_json::from_str(&line).unwrap_or_default();
                    let object = &event["object"];
                    println!(
                        "{:9} {} {}",
                        event["type"].as_str().unwrap_or("-"),
                        object["metadata"]["name"].as_str().unwrap_or("-"),
                        object["metadata"]["resourceVersion"]
                            .as_str()
                            .unwrap_or("-")
                    );
                }
            }
        }
        Err(OperatorError::Unreachable(
            "the watch ended: the coordinator stopped or its term changed; watch again".into(),
        ))
    })
}

/// One line of what a write left: the node's spec and where its drain stands.
fn written(node: &Node) {
    let draining = node
        .status
        .conditions
        .iter()
        .find(|c| c.kind == "Draining" && c.status == crate::status::Truth::True)
        .map_or(String::new(), |c| {
            format!(", draining: {} {}", c.reason, c.message)
        });
    println!(
        "{}: {}{}{} (generation {})",
        node.metadata.name,
        if node.spec.unschedulable {
            "cordoned"
        } else {
            "schedulable"
        },
        node.spec.drain.map_or(String::new(), |d| format!(
            ", drained by {} ms",
            d.deadline_ms
        )),
        draining,
        node.metadata.generation
    );
}

pub(super) fn run_cordon(args: &NodeWriteArgs, on: bool) -> ExitCode {
    with_operator(&args.api, async |op| {
        written(&op.cordon(&args.node, on, &args.reason).await?);
        Ok(())
    })
}

pub(super) fn run_drain(args: &DrainArgs) -> ExitCode {
    let lead = i64::try_from(args.lead_secs.saturating_mul(1000)).unwrap_or(i64::MAX);
    let deadline = now_ms().saturating_add(lead);
    with_operator(&args.api, async |op| {
        written(&op.drain(&args.node, deadline, &args.reason).await?);
        Ok(())
    })
}

pub(super) fn run_undrain(args: &NodeWriteArgs) -> ExitCode {
    with_operator(&args.api, async |op| {
        written(&op.undrain(&args.node, &args.reason).await?);
        Ok(())
    })
}
