// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! What a resource's status says it means (decision 0036): its conditions, and the log of what
//! the controller decided about it and why.
//!
//! A condition is a type with true, false or unknown, a reason in one word, a message in a
//! sentence, and when its status last changed. A scheduler or a console reads conditions, never
//! infers state from fields. Conditions are derived from observed status each time it changes,
//! never written by an operator; only a change of status moves the transition time, and each
//! such change is also an event.
//!
//! The event log is bounded per resource: the newest [`EVENTS_KEPT`] events, oldest first.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

/// Events kept per resource.
pub const EVENTS_KEPT: usize = 64;

/// A condition's status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub enum Truth {
    /// It holds.
    True,
    /// It does not.
    False,
    /// The coordinator cannot tell yet.
    Unknown,
}

impl From<bool> for Truth {
    fn from(b: bool) -> Self {
        if b { Self::True } else { Self::False }
    }
}

/// Whether a condition's status is good news when true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// True is healthy: Serving, Covered.
    Healthy,
    /// True is trouble: Degraded, Down.
    Trouble,
    /// Neither: Progressing, Draining.
    Neutral,
}

/// One condition as derived from observed status: its type, status, reason and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observed {
    /// The condition's type: `Serving`, `Degraded`.
    pub kind: &'static str,
    /// Whether true is good news.
    pub polarity: Polarity,
    /// Its status.
    pub status: Truth,
    /// Why, in one word: `HeadLost`, `NodeUncovered`.
    pub reason: &'static str,
    /// Why, in a sentence, for people.
    pub message: String,
}

impl Observed {
    /// A condition of `kind` and `polarity`, `status` for `reason`, said as `message`.
    #[must_use]
    pub fn new(
        kind: &'static str,
        polarity: Polarity,
        status: impl Into<Truth>,
        reason: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            polarity,
            status: status.into(),
            reason,
            message: message.into(),
        }
    }
}

/// One condition as status carries it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    /// Its type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Its status.
    pub status: Truth,
    /// Why, in one word.
    pub reason: String,
    /// Why, in a sentence.
    pub message: String,
    /// When its status last changed, Unix milliseconds on the coordinator's clock.
    pub last_transition_ms: i64,
}

/// Whether an event is routine or calls for attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub enum Severity {
    /// Routine: a change committed, a drain finished.
    Normal,
    /// Calls for attention: a change refused, a drain blocked.
    Warning,
}

/// One event: what the controller decided or saw, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    /// When, Unix milliseconds on the coordinator's clock.
    pub at_ms: i64,
    /// Routine, or calls for attention.
    pub severity: Severity,
    /// What, in one word: `ChangeCommitted`, `DrainWaiting`.
    pub reason: String,
    /// What, in a sentence.
    pub message: String,
}

/// A resource's conditions and its event log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Status {
    conditions: BTreeMap<&'static str, Condition>,
    events: VecDeque<Event>,
}

impl Status {
    /// Take `observed` as the conditions now, at `now_ms`. A condition whose status changed gets
    /// `now_ms` as its transition time and an event: a warning when it turned to trouble, normal
    /// otherwise. One whose status held keeps its transition time and takes the new reason and
    /// message without an event.
    pub fn observe(&mut self, observed: Vec<Observed>, now_ms: i64) {
        for o in observed {
            let changed = self
                .conditions
                .get(o.kind)
                .is_none_or(|c| c.status != o.status);
            if !changed {
                if let Some(c) = self.conditions.get_mut(o.kind) {
                    o.reason.clone_into(&mut c.reason);
                    c.message = o.message;
                }
                continue;
            }
            let trouble = match o.polarity {
                Polarity::Healthy => o.status == Truth::False,
                Polarity::Trouble => o.status == Truth::True,
                Polarity::Neutral => false,
            };
            let first = !self.conditions.contains_key(o.kind);
            // a condition first seen in its quiet state is not news
            if !first || trouble || o.polarity == Polarity::Healthy && o.status == Truth::True {
                self.record(
                    now_ms,
                    if trouble {
                        Severity::Warning
                    } else {
                        Severity::Normal
                    },
                    o.reason,
                    format!("{} is {:?}: {}", o.kind, o.status, o.message),
                );
            }
            self.conditions.insert(
                o.kind,
                Condition {
                    kind: o.kind.to_owned(),
                    status: o.status,
                    reason: o.reason.to_owned(),
                    message: o.message,
                    last_transition_ms: now_ms,
                },
            );
        }
    }

    /// Record an event, dropping the oldest beyond [`EVENTS_KEPT`].
    pub fn record(
        &mut self,
        at_ms: i64,
        severity: Severity,
        reason: &str,
        message: impl Into<String>,
    ) {
        if self.events.len() == EVENTS_KEPT {
            self.events.pop_front();
        }
        self.events.push_back(Event {
            at_ms,
            severity,
            reason: reason.to_owned(),
            message: message.into(),
        });
    }

    /// The conditions, by type.
    #[must_use]
    pub fn conditions(&self) -> Vec<Condition> {
        self.conditions.values().cloned().collect()
    }

    /// The condition of `kind`, if it was ever observed.
    #[must_use]
    pub fn condition(&self, kind: &str) -> Option<&Condition> {
        self.conditions.get(kind)
    }

    /// The events kept, oldest first.
    #[must_use]
    pub fn events(&self) -> Vec<Event> {
        self.events.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{EVENTS_KEPT, Observed, Polarity, Severity, Status, Truth};

    fn serving(up: bool, why: &str) -> Observed {
        Observed::new(
            "Serving",
            Polarity::Healthy,
            up,
            if up { "Members" } else { "Down" },
            why,
        )
    }

    #[test]
    fn a_transition_moves_the_time_and_is_an_event_and_a_steady_status_is_neither() {
        let mut s = Status::default();
        s.observe(vec![serving(true, "8 members")], 10);
        s.observe(vec![serving(true, "7 members")], 20);
        let c = s.condition("Serving").expect("serving");
        assert_eq!((c.status, c.last_transition_ms), (Truth::True, 10));
        assert_eq!(c.message, "7 members", "the message follows the status");
        assert_eq!(s.events().len(), 1, "{:?}", s.events());
        s.observe(vec![serving(false, "its head was lost")], 30);
        let c = s.condition("Serving").expect("serving");
        assert_eq!((c.status, c.last_transition_ms), (Truth::False, 30));
        let e = s.events().pop().expect("event");
        assert_eq!(
            (e.severity, e.reason.as_str(), e.at_ms),
            (Severity::Warning, "Down", 30)
        );
    }

    #[test]
    fn a_trouble_condition_first_seen_false_is_not_news() {
        let mut s = Status::default();
        s.observe(
            vec![Observed::new(
                "Degraded",
                Polarity::Trouble,
                false,
                "AsSpecified",
                "",
            )],
            10,
        );
        assert!(s.events().is_empty());
        s.observe(
            vec![Observed::new(
                "Degraded",
                Polarity::Trouble,
                true,
                "Shrunk",
                "rank 3 lost",
            )],
            20,
        );
        assert_eq!(s.events()[0].severity, Severity::Warning);
    }

    #[test]
    fn the_log_keeps_the_newest_events() {
        let mut s = Status::default();
        for i in 0..EVENTS_KEPT + 3 {
            s.record(i64::try_from(i).unwrap(), Severity::Normal, "Tick", "");
        }
        let events = s.events();
        assert_eq!(events.len(), EVENTS_KEPT);
        assert_eq!(events[0].at_ms, 3, "the oldest went first");
    }
}
