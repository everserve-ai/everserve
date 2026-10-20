// SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
// SPDX-License-Identifier: Apache-2.0
//! Capability before movement (decision 0034): whether a process may become a destination, and
//! whether what it runs matches what a position runs. Every refusal names the field.

use crate::proto::everserve::signals::v1::{KvFormat, SoftwareIdentity, StepConfig};

/// Why a process cannot be a destination at all, whatever it would take: it sent no identity
/// (every process of this protocol sends one, so none is a defect, not an older build), or its
/// adapter was built from a tree with uncommitted changes.
pub(super) fn destination_refusal(id: Option<&SoftwareIdentity>) -> Option<String> {
    match id {
        None => Some("it reported no software identity".to_owned()),
        Some(i) if i.adapter_dirty => Some(format!(
            "its adapter at {} was built from a tree with uncommitted changes",
            i.adapter_commit
        )),
        Some(_) => None,
    }
}

/// The first field in which `spare` differs from `position` among those that make two processes
/// of one engine interchangeable whatever position they hold: the engine and adapter commits,
/// NCCL, the checkpoint, the numerics, the parallel widths and the KV's dtype and block size, and
/// under the exact class the CUDA runtime. A compared field that either side left empty refuses
/// too, naming the side: the adapter leaves a field empty when it cannot read it rather than
/// guess, which is safe only if an empty field never matches another (two processes installed
/// from one unstamped wheel both report an empty adapter commit). The quantization method is the
/// one field where empty is a value: no quantization.
pub(super) fn engine_difference(
    spare: &SoftwareIdentity,
    position: &SoftwareIdentity,
    exact: bool,
) -> Option<String> {
    let (s_kv, p_kv) = (
        spare.kv_format.clone().unwrap_or_default(),
        position.kv_format.clone().unwrap_or_default(),
    );
    let block = |k: &KvFormat| {
        if k.block_size == 0 {
            String::new()
        } else {
            k.block_size.to_string()
        }
    };
    // (field, here, on the position, whether empty refuses)
    let mut fields: Vec<(&str, String, String, bool)> = vec![
        (
            "engine",
            spare.engine.clone(),
            position.engine.clone(),
            true,
        ),
        (
            "engine_commit",
            spare.engine_commit.clone(),
            position.engine_commit.clone(),
            true,
        ),
        (
            "adapter_commit",
            spare.adapter_commit.clone(),
            position.adapter_commit.clone(),
            true,
        ),
        (
            "nccl_version",
            spare.nccl_version.clone(),
            position.nccl_version.clone(),
            true,
        ),
        (
            "checkpoint_digest",
            spare.checkpoint_digest.clone(),
            position.checkpoint_digest.clone(),
            true,
        ),
        (
            "quantization",
            spare.quantization.clone(),
            position.quantization.clone(),
            false,
        ),
        (
            "model_dtype",
            spare.model_dtype.clone(),
            position.model_dtype.clone(),
            true,
        ),
        (
            "guarantee",
            spare.guarantee.clone(),
            position.guarantee.clone(),
            true,
        ),
        (
            "layout",
            format!("{:?}", spare.layout),
            format!("{:?}", position.layout),
            false,
        ),
        (
            "kv_format.dtype",
            s_kv.dtype.clone(),
            p_kv.dtype.clone(),
            true,
        ),
        ("kv_format.block_size", block(&s_kv), block(&p_kv), true),
    ];
    if exact {
        fields.push((
            "cuda_version",
            spare.cuda_version.clone(),
            position.cuda_version.clone(),
            true,
        ));
    }
    fields
        .into_iter()
        .find_map(|(field, s, p, required)| {
            if required && s.is_empty() {
                Some(format!("{field} is empty here"))
            } else if required && p.is_empty() {
                Some(format!("{field} is empty on the position"))
            } else {
                (s != p).then(|| format!("{field} is {s:?} here, {p:?} on the position"))
            }
        })
        .or_else(|| step_difference(spare.step.as_ref(), position.step.as_ref()))
}

/// How `here`'s step config differs from `engine`'s, each differing key named with both values;
/// None when they are the same. One left empty matches none, naming its side: a process whose
/// adapter could not resolve what shapes its collectives cannot be shown to join them in step.
pub(super) fn step_difference(
    here: Option<&StepConfig>,
    engine: Option<&StepConfig>,
) -> Option<String> {
    fn given(s: Option<&StepConfig>) -> Option<&StepConfig> {
        s.filter(|s| !s.digest.is_empty())
    }
    let (Some(here), Some(engine)) = (given(here), given(engine)) else {
        return Some(if given(here).is_none() {
            "the step config is empty here".to_owned()
        } else {
            "the step config is empty on the engine".to_owned()
        });
    };
    if here.digest == engine.digest {
        return None;
    }
    let keys: std::collections::BTreeSet<&String> =
        here.fields.keys().chain(engine.fields.keys()).collect();
    let differing: Vec<String> = keys
        .into_iter()
        .filter_map(|k| {
            let (h, e) = (here.fields.get(k), engine.fields.get(k));
            (h != e).then(|| {
                format!(
                    "{k} is {:?} here, {:?} on the engine",
                    h.map_or("", String::as_str),
                    e.map_or("", String::as_str)
                )
            })
        })
        .collect();
    Some(if differing.is_empty() {
        format!(
            "the step config's digest is {} here, {} on the engine",
            here.digest, engine.digest
        )
    } else {
        differing.join("; ")
    })
}

/// [`engine_difference`], and the KV's layout for the position: what a spare preloaded for one
/// position must equal before it is admitted for it. A spare for one rank and a serving worker
/// both register after their caches exist, so either side's layout empty refuses.
pub(super) fn position_difference(
    spare: &SoftwareIdentity,
    position: &SoftwareIdentity,
    exact: bool,
) -> Option<String> {
    engine_difference(spare, position, exact).or_else(|| {
        let s = spare
            .kv_format
            .as_ref()
            .map_or("", |k| k.layout_hash.as_str());
        let p = position
            .kv_format
            .as_ref()
            .map_or("", |k| k.layout_hash.as_str());
        if s.is_empty() {
            Some("kv_format.layout_hash is empty here".to_owned())
        } else if p.is_empty() {
            Some("kv_format.layout_hash is empty on the position".to_owned())
        } else {
            (s != p).then(|| format!("kv_format.layout_hash is {s:?} here, {p:?} on the position"))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{destination_refusal, engine_difference, position_difference, step_difference};
    use crate::proto::everserve::signals::v1::{
        KvFormat, ParallelLayout, SoftwareIdentity, StepConfig,
    };

    fn step(seqs: &str) -> StepConfig {
        StepConfig {
            digest: format!("seqs={seqs}"),
            fields: [
                ("max_num_seqs".to_owned(), seqs.to_owned()),
                ("block_size".to_owned(), "16".to_owned()),
            ]
            .into(),
        }
    }

    fn identity() -> SoftwareIdentity {
        SoftwareIdentity {
            engine: "vllm".into(),
            engine_commit: "6e517b15".into(),
            adapter_commit: "a9c4391".into(),
            cuda_version: "12.8".into(),
            nccl_version: "2.27.3".into(),
            checkpoint_digest: "c0ffee".into(),
            quantization: "compressed-tensors".into(),
            model_dtype: "bfloat16".into(),
            kv_format: Some(KvFormat {
                dtype: "bfloat16".into(),
                block_size: 16,
                layout_hash: "stage-0".into(),
            }),
            layout: Some(ParallelLayout {
                tensor: 1,
                pipeline: 2,
                data: 1,
                expert: 1,
                context: 1,
            }),
            step: Some(step("256")),
            guarantee: "exact".into(),
            ..SoftwareIdentity::default()
        }
    }

    #[test]
    fn a_spare_started_for_another_guarantee_than_its_positions_is_named() {
        let p = identity();
        let continuous = SoftwareIdentity {
            guarantee: "continuous".into(),
            ..identity()
        };
        assert!(
            engine_difference(&continuous, &p, false)
                .is_some_and(|w| w.contains(r#"guarantee is "continuous" here, "exact""#)),
            "{:?}",
            engine_difference(&continuous, &p, false)
        );
        let none = SoftwareIdentity {
            guarantee: String::new(),
            ..identity()
        };
        assert!(
            engine_difference(&none, &p, false).is_some_and(|w| w.contains("guarantee is empty"))
        );
    }

    #[test]
    fn a_step_config_that_differs_names_each_differing_field_and_an_empty_one_matches_none() {
        // a stage built with the class defaults asks 128 sequences of a collective its head asks
        // 256 of, and hangs there
        let (head, stage) = (step("256"), step("128"));
        assert_eq!(step_difference(Some(&head), Some(&head)), None);
        assert_eq!(
            step_difference(Some(&stage), Some(&head)).as_deref(),
            Some(r#"max_num_seqs is "128" here, "256" on the engine"#)
        );
        assert!(
            step_difference(None, Some(&head)).is_some_and(|w| w.contains("empty here")),
            "a process reporting none is refused"
        );
        let blank = StepConfig::default();
        assert!(step_difference(Some(&blank), Some(&blank)).is_some());
        // a spare whose step config differs is refused, naming it
        let spare = SoftwareIdentity {
            step: Some(stage),
            ..identity()
        };
        assert!(
            engine_difference(&spare, &identity(), false)
                .is_some_and(|w| w.contains("max_num_seqs"))
        );
    }

    #[test]
    fn a_destination_needs_an_identity_from_a_clean_build() {
        assert!(destination_refusal(None).is_some());
        assert!(destination_refusal(Some(&identity())).is_none());
        let dirty = SoftwareIdentity {
            adapter_dirty: true,
            ..identity()
        };
        assert!(destination_refusal(Some(&dirty)).is_some_and(|why| why.contains("uncommitted")));
    }

    #[test]
    fn each_compared_field_is_named_and_reported_ones_pass() {
        let p = identity();
        assert_eq!(engine_difference(&p, &p, true), None);
        let other = SoftwareIdentity {
            checkpoint_digest: "beef".into(),
            ..identity()
        };
        assert!(
            engine_difference(&other, &p, true).is_some_and(|w| w.starts_with("checkpoint_digest"))
        );
        // reported only: torch, the driver, the revision, the model's served id
        let reported = SoftwareIdentity {
            torch_version: "2.9".into(),
            driver_version: "570".into(),
            model_revision: "f50dbad2".into(),
            model: "other-name".into(),
            ..identity()
        };
        assert_eq!(engine_difference(&reported, &p, true), None);
        // CUDA is compared under the exact class only
        let cuda = SoftwareIdentity {
            cuda_version: "12.9".into(),
            ..identity()
        };
        assert!(engine_difference(&cuda, &p, true).is_some_and(|w| w.starts_with("cuda_version")));
        assert_eq!(engine_difference(&cuda, &p, false), None);
    }

    #[test]
    fn a_position_also_compares_the_kv_layout_an_engine_wide_check_does_not() {
        let p = identity();
        let mut other_stage = identity();
        if let Some(kv) = other_stage.kv_format.as_mut() {
            kv.layout_hash = "stage-1".into();
        }
        assert_eq!(engine_difference(&other_stage, &p, true), None);
        assert!(
            position_difference(&other_stage, &p, true)
                .is_some_and(|w| w.starts_with("kv_format.layout_hash"))
        );
    }

    #[test]
    fn an_empty_compared_field_never_matches_another_and_names_its_side() {
        // two processes installed from one unstamped wheel both report no adapter commit
        let unstamped = SoftwareIdentity {
            adapter_commit: String::new(),
            ..identity()
        };
        assert_eq!(
            engine_difference(&unstamped, &unstamped, true).as_deref(),
            Some("adapter_commit is empty here")
        );
        assert_eq!(
            engine_difference(&identity(), &unstamped, true).as_deref(),
            Some("adapter_commit is empty on the position")
        );
        let no_block = SoftwareIdentity {
            kv_format: Some(KvFormat {
                block_size: 0,
                ..identity().kv_format.unwrap_or_default()
            }),
            ..identity()
        };
        assert!(
            engine_difference(&no_block, &no_block, false)
                .is_some_and(|w| w.starts_with("kv_format.block_size is empty"))
        );
        // no quantization is a value both sides can hold
        let unquantized = SoftwareIdentity {
            quantization: String::new(),
            ..identity()
        };
        assert_eq!(engine_difference(&unquantized, &unquantized, true), None);
        // a CUDA left empty refuses under the exact class only
        let no_cuda = SoftwareIdentity {
            cuda_version: String::new(),
            ..identity()
        };
        assert!(engine_difference(&no_cuda, &no_cuda, true).is_some());
        assert_eq!(engine_difference(&no_cuda, &no_cuda, false), None);
        // a spare for one rank with no KV layout refuses even against another with none
        let mut no_layout = identity();
        if let Some(kv) = no_layout.kv_format.as_mut() {
            kv.layout_hash = String::new();
        }
        assert_eq!(
            position_difference(&no_layout, &no_layout, true).as_deref(),
            Some("kv_format.layout_hash is empty here")
        );
    }
}
