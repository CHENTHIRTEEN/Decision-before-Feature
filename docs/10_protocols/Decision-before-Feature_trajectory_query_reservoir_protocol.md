# Decision-before-Feature trajectory query reservoir protocol

> Formal revision (2026-08-11): the trajectory contract no longer tries to reconstruct a complete evaluated-point archive after the run. Instead, each optimizer run maintains a zero-extra-FE online reservoir of already evaluated points, frozen per seed and per problem, so that trajectory-based query baselines can be computed without independent re-sampling. The reservoir is a representative subset, not a complete archive.

## 1. Revision objective

The previous trajectory contract retained only:

- emitted native-update population checkpoints,
- lightweight late-stage scalar history,
- no full history of all evaluated points.

That was sufficient for behavior analysis and selection reference construction, but insufficient for fair post hoc comparison against trajectory-based query baselines that require the evaluated-point prefix set at state \(t\).

This revision adds an online `TrajectoryQueryReservoir` that records already evaluated points at evaluation time, without spending extra FE and without depending on post hoc reconstruction.

## 2. Scientific scope

The reservoir is intended for:

- trajectory-based query features,
- reservoir-based approximate ELA baselines,
- per-run algorithm selection diagnostics,
- zero-extra-FE query accounting.

It is not intended to replace a complete evaluated archive when the research question requires exact reconstruction of all historical points.

## 3. Frozen protocol

| field | value |
| --- | --- |
| `query_id` | `trajectory_descriptor_cheap_16` |
| `query_protocol` | `trajectory_query_reservoir_v1` |
| `query_source_id` | `descriptor_cheap_invariant` |
| `query_preprocessing_id` | `unit_cube_x__median_iqr_y_v1` |
| `query_feature_columns` | frozen 16-column descriptor whitelist |
| reservoir size | `50D` per run by default |

The reservoir size may be overridden only by an explicit protocol revision. It must not be tuned per problem, per algorithm, or per validation result.

## 4. Reservoir update rule

For each already evaluated point \((x_i, f(x_i))\):

1. The point is passed to the reservoir immediately when the objective value is known.
2. The reservoir uses deterministic semantic seeding derived from:
   - `query_protocol`
   - `problem_id`
   - `family`
   - `dimension`
   - `algorithm`
   - `seed`
   - reservoir size
3. The reservoir performs standard replacement sampling over the observed evaluation stream.
4. The reservoir never triggers additional objective evaluations.

The reservoir therefore stores a representative subset of the evaluated stream and is reproducible for the same seed and problem configuration.

## 5. Output contract

Each reservoir snapshot is written as a separate Parquet row with the following required fields:

```text
split
problem_id
family
function
instance
dimension
algorithm
seed
FE
FE_ratio
FE_total
query_id
query_protocol
query_source_id
query_preprocessing_id
query_feature_columns
trajectory_query_reservoir_size
trajectory_query_seen_count
trajectory_sample_count
trajectory_sample_coverage_ratio
trajectory_query_runtime
feature_status
feature_count
feature_failure
feature_group_status
feature_nonfinite
```

The feature columns themselves are stored in the same row and must exactly match the frozen descriptor whitelist.

## 6. Interpretation constraints

A reservoir snapshot is valid evidence for:

- zero-extra-FE trajectory query computation,
- reservoir-based trajectory descriptors,
- algorithm behavior derived from the already evaluated stream.

It is **not** valid to claim that the reservoir equals the entire evaluated prefix archive. Any analysis requiring exact historical coverage must use a separately stored complete archive.

## 7. Relationship to the existing trajectory contract

The existing `trajectory.parquet` contract remains responsible for:

- native-update checkpoints,
- dynamic sampling metadata,
- window statistics,
- selection and utility joins.

The new reservoir protocol is additive. It does not alter the meaning of:

- `FE`
- `FE_ratio`
- `optimizer_state_mode`
- `sampling_protocol`
- `window_statistics`

## 8. Fair comparison statement

With this revision, the following comparison becomes protocol-valid:

- `Behavior-only Query`
- `Trajectory descriptor Query`
- `Independent descriptor Query`

The first two may share the same already evaluated stream under the reservoir protocol. The independent query remains an external baseline with its own sampling budget and must keep its extra FE accounting explicit.

## 9. Reproducibility and versioning

This protocol is frozen and must be versioned independently from the main trajectory sampling protocol.
Any future change to reservoir size, replacement policy, or feature whitelist requires a new `query_protocol` and a new document revision.
