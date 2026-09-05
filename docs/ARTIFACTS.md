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

## Amazon Reviews 2023: Movies and TV

Download the official Movies and TV review and metadata files from the
[Amazon Reviews 2023 project](https://amazon-reviews-2023.github.io/) and place
them at the exact paths below:

```text
data/Amazon_Reviews_2023_Movies_and_TV/raw/review_categories/Movies_and_TV.jsonl
data/Amazon_Reviews_2023_Movies_and_TV/raw/meta_categories/meta_Movies_and_TV.jsonl.gz
pretrained_models/t5-base/
outputs/ru1/i02s42v1/configs/base_t5.yaml
```

The metadata archive used in the paper has SHA-256
`0afbb248ddb00c012c3e2c1e68505b7cd28b02a2c83fb4421ff0c71ad5a48b8d`.
The unified launcher creates the following lineage:

```text
outputs/bota_amazon_movies_tv_v1/prepared/amazon_movies_tv_titles_seed42_v2/
outputs/bota_amazon_movies_tv_v3/originals/amazon_movies_tv_titles_original_p5_seed42_v3/
outputs/bota_amazon_movies_tv_v4/prepared/amazon_movies_tv_newusers_seed42_v4/
outputs/amz_v4/k2k4/
outputs/amz_v4/k2_all_evaluation/amz_new_k2_all_eval_s42_v4/
```

The first 256 eligible users form the historical recommendation cohort. The
next 768 users form a disjoint adaptation cohort. Raw user and item identifiers
are not written to the prepared artifacts. This experiment is Development-only;
it neither creates nor accesses a FinalTest split.

## What may be hosted separately

For an archival release, the following are suitable for Zenodo, an institutional repository or a model hub subject to the original licenses:

1. the trained recommendation Original checkpoint;
2. non-sensitive processed prompt files when redistribution is permitted;
3. the frozen benchmark registry;
4. final aggregate result tables and manifests;
5. a small example impact bank containing synthetic users only.

Do not publish raw user identifiers, request-indexed production banks, prompts containing private user content, access tokens, absolute machine paths or resumable failed-run directories.
