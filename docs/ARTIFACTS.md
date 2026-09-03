# External artifacts

Large datasets and model weights are deliberately not committed. The formal configurations are evidence-locked: paths, sample counts and selected SHA-256 values describe the exact artifacts used for the paper. Changing an artifact creates a new experimental lineage and requires updating the corresponding configuration rather than silently bypassing validation.

## MovieLens-1M

Required local inputs:

```text
data/ml-1m/raw_data/                                  MovieLens-1M raw files
pretrained_models/t5-base/                            complete T5-base directory
checkpoint/ml-1m-base-original-0.0005/model.pt        trained recommendation Original
outputs/ru1/i02s42v1/configs/base_t5.yaml             model/data protocol
outputs/ru1/i02s42v1/data/train.json                  60,000 training prompts
outputs/ru1/i02s42v1/data/retain.json                 retained training prompts
outputs/ru1/i02s42v1/data/forget.json                 deletion prompts
outputs/ru1/i02s42v1/data/development.json            20,000 Development prompts
data/ml-1m/proc_data/data/test/test_10_simple.json    held-out FinalTest prompts
```

The Original checkpoint used in the audited experiments is approximately 892 MB and therefore must be distributed through a release asset, model hub or institutional archive rather than ordinary Git. The text-to-text preprocessing lineage follows [E2URec](https://github.com/justarter/E2URec). The upstream repository contains the MovieLens preprocessing notebooks and prompt conversion scripts; it does not remove the need to obtain MovieLens under its own terms.

The seed-41/42/43 experiments use the same frozen data and recommendation Original, while changing the fixed LoRA-A seed and stochastic run seed through the supplied configs.

The main three-seed FinalTest evaluator validates
`data/ml-1m/proc_data/data/test/test_10_simple.json` against a deterministic
replay from `ratings.dat` before inference. The E2URec supplemental evaluator
also requires the completed primary FinalTest directory under:

```text
outputs/bota_short_multiseed_finaltest_v3/evaluations/<primary-finaltest-run-name>/
```

Pass `<primary-finaltest-run-name>` with
`-PrimaryFinalTestRunName`; the reported experiment used
`bota_short_multiseed_finaltest_ml1m_seed41_43_v3_recovery1`.

## GoodReads comics

Required inputs begin with the public GoodReads comics interactions under:

```text
data/goodreads/comics/
pretrained_models/t5-base/
```

The release launchers create or expect:

```text
outputs/bota_goodreads_v1/prepared/goodreads_comics_seed42_v1/
outputs/bota_goodreads_v1/recommendation_originals/goodreads_recommendation_original_seed42_v2/model/
```

GoodReads configs contain prepared-data and Original-model hashes from the paper lineage. A fresh preprocessing run should be treated as a new lineage if those hashes differ.

## What may be hosted separately

For an archival release, the following are suitable for Zenodo, an institutional repository or a model hub subject to the original licenses:

1. the trained recommendation Original checkpoint;
2. non-sensitive processed prompt files when redistribution is permitted;
3. the frozen benchmark registry;
4. final aggregate result tables and manifests;
5. a small example impact bank containing synthetic users only.

Do not publish raw user identifiers, request-indexed production banks, prompts containing private user content, access tokens, absolute machine paths or resumable failed-run directories.
