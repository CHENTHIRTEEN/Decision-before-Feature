from __future__ import annotations

from dataclasses import dataclass


LANDSCAPE_QUERY_PROTOCOL_VERSION = "landscape_query_v3"
QUERY_PREPROCESSING_VERSION = "unit_cube_x__median_iqr_y_v1"
SAMPLE_PROTOCOL_VERSION = "lhs_problem_sample_v2"
MAIN_QUERY_ID = "descriptor_cheap_invariant"


@dataclass(frozen=True)
class SampleDesignSpec:
    sample_design_id: str
    design_code: int
    sample_size_per_dimension: int
    protocol: str = SAMPLE_PROTOCOL_VERSION

    def sample_size(self, dimension: int) -> int:
        if int(dimension) <= 0:
            raise ValueError("dimension must be positive")
        return int(self.sample_size_per_dimension * int(dimension))


@dataclass(frozen=True)
class LandscapeQuerySpec:
    query_id: str
    query_code: int
    sample_design_id: str
    backend: str
    feature_groups: tuple[str, ...]
    feature_columns: tuple[str, ...]
    primary: bool
    protocol: str
    preprocessing_id: str

    @property
    def sample_design(self) -> SampleDesignSpec:
        return get_sample_design_spec(self.sample_design_id)


DESCRIPTOR_CHEAP_COLUMNS = (
    "descriptor_y_min",
    "descriptor_y_max",
    "descriptor_y_mean",
    "descriptor_y_std",
    "descriptor_y_skew",
    "descriptor_y_kurtosis",
    "descriptor_x_mean_pairwise",
    "descriptor_x_std_pairwise",
    "descriptor_x_best_dist_center",
    "descriptor_x_mean_dist_center",
    "descriptor_corr_y_dist_center",
    "descriptor_corr_y_nn_dist",
    "descriptor_linear_r2",
    "descriptor_linear_gradient_norm",
)

PFLACCO_PCA_COLUMNS = (
    "pca.expl_var.cov_x",
    "pca.expl_var.cor_x",
    "pca.expl_var.cov_init",
    "pca.expl_var.cor_init",
    "pca.expl_var_PC1.cov_x",
    "pca.expl_var_PC1.cor_x",
    "pca.expl_var_PC1.cov_init",
    "pca.expl_var_PC1.cor_init",
)

PFLACCO_NBC_COLUMNS = (
    "nbc.nn_nb.sd_ratio",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
)

PFLACCO_DISPERSION_COLUMNS = tuple(
    f"disp.{stat}_{quantile}"
    for stat in ("ratio_mean", "ratio_median", "diff_mean", "diff_median")
    for quantile in ("02", "05", "10", "25")
)

PFLACCO_INFORMATION_CONTENT_COLUMNS = (
    "ic.h_max",
    "ic.eps_s",
    "ic.eps_max",
    "ic.eps_ratio",
    "ic.m0",
)

PFLACCO_DISTRIBUTION_COLUMNS = (
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
)

PFLACCO_LEVEL_COLUMNS = tuple(
    f"ela_level.{stat}_{quantile}"
    for quantile in ("10", "25", "50")
    for stat in ("mmce_lda", "mmce_qda", "lda_qda")
)

PFLACCO_FDC_COLUMNS = (
    "fitness_distance.fd_correlation",
    "fitness_distance.fd_cov",
    "fitness_distance.distance_mean",
    "fitness_distance.distance_std",
    "fitness_distance.fitness_mean",
    "fitness_distance.fitness_std",
)

PFLACCO_GROUP_COLUMNS = {
    "pca": PFLACCO_PCA_COLUMNS,
    "nbc": PFLACCO_NBC_COLUMNS,
    "dispersion": PFLACCO_DISPERSION_COLUMNS,
    "information_content": PFLACCO_INFORMATION_CONTENT_COLUMNS,
    "ela_distribution": PFLACCO_DISTRIBUTION_COLUMNS,
    "ela_level": PFLACCO_LEVEL_COLUMNS,
    "fitness_distance_correlation": PFLACCO_FDC_COLUMNS,
}

PFLACCO_STANDARD_GROUPS = (
    "pca",
    "nbc",
    "dispersion",
    "information_content",
    "ela_distribution",
)
PFLACCO_BROAD_GROUPS = PFLACCO_STANDARD_GROUPS + (
    "ela_level",
    "fitness_distance_correlation",
)


def _columns_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for group in groups for column in PFLACCO_GROUP_COLUMNS[group])


PFLACCO_STANDARD_COLUMNS = _columns_for_groups(PFLACCO_STANDARD_GROUPS)
PFLACCO_BROAD_COLUMNS = _columns_for_groups(PFLACCO_BROAD_GROUPS)

SAMPLE_DESIGN_SPECS = {
    "lhs_50d": SampleDesignSpec(
        sample_design_id="lhs_50d",
        design_code=50,
        sample_size_per_dimension=50,
    ),
    "lhs_100d": SampleDesignSpec(
        sample_design_id="lhs_100d",
        design_code=100,
        sample_size_per_dimension=100,
    ),
    "lhs_5d": SampleDesignSpec(
        sample_design_id="lhs_5d",
        design_code=5,
        sample_size_per_dimension=5,
    ),
    "lhs_10d": SampleDesignSpec(
        sample_design_id="lhs_10d",
        design_code=10,
        sample_size_per_dimension=10,
    ),
    "lhs_20d": SampleDesignSpec(
        sample_design_id="lhs_20d",
        design_code=20,
        sample_size_per_dimension=20,
    ),
}

LANDSCAPE_QUERY_SPECS = {
    MAIN_QUERY_ID: LandscapeQuerySpec(
        query_id=MAIN_QUERY_ID,
        query_code=5016,
        sample_design_id="lhs_50d",
        backend="native_descriptor",
        feature_groups=("descriptor_cheap",),
        feature_columns=DESCRIPTOR_CHEAP_COLUMNS,
        primary=True,
        protocol=f"{LANDSCAPE_QUERY_PROTOCOL_VERSION}:descriptor_cheap_invariant_14_lhs_50d",
        preprocessing_id=QUERY_PREPROCESSING_VERSION,
    ),
    "pflacco_standard_invariant": LandscapeQuerySpec(
        query_id="pflacco_standard_invariant",
        query_code=5037,
        sample_design_id="lhs_50d",
        backend="pflacco_1.2.2",
        feature_groups=PFLACCO_STANDARD_GROUPS,
        feature_columns=PFLACCO_STANDARD_COLUMNS,
        primary=False,
        protocol=f"{LANDSCAPE_QUERY_PROTOCOL_VERSION}:pflacco_1.2.2_standard_invariant_37_lhs_50d",
        preprocessing_id=QUERY_PREPROCESSING_VERSION,
    ),
    "pflacco_broad_invariant": LandscapeQuerySpec(
        query_id="pflacco_broad_invariant",
        query_code=10052,
        sample_design_id="lhs_100d",
        backend="pflacco_1.2.2",
        feature_groups=PFLACCO_BROAD_GROUPS,
        feature_columns=PFLACCO_BROAD_COLUMNS,
        primary=False,
        protocol=f"{LANDSCAPE_QUERY_PROTOCOL_VERSION}:pflacco_1.2.2_broad_invariant_52_lhs_100d",
        preprocessing_id=QUERY_PREPROCESSING_VERSION,
    ),
}


def get_query_spec(query_id: str) -> LandscapeQuerySpec:
    try:
        return LANDSCAPE_QUERY_SPECS[str(query_id)]
    except KeyError as exc:
        raise ValueError(f"unknown landscape query: {query_id!r}") from exc


def get_sample_design_spec(sample_design_id: str) -> SampleDesignSpec:
    try:
        return SAMPLE_DESIGN_SPECS[str(sample_design_id)]
    except KeyError as exc:
        raise ValueError(f"unknown sample design: {sample_design_id!r}") from exc


def validate_frozen_query_specs() -> None:
    cheap = get_query_spec("descriptor_cheap_invariant")
    standard = get_query_spec("pflacco_standard_invariant")
    broad = get_query_spec("pflacco_broad_invariant")
    if cheap.sample_design_id != standard.sample_design_id:
        raise ValueError("descriptor_cheap_invariant and pflacco_standard_invariant must share lhs_50d")
    if cheap.sample_design.sample_size_per_dimension != 50:
        raise ValueError("descriptor_cheap_invariant must use lhs_50d")
    if broad.sample_design.sample_size_per_dimension != 100:
        raise ValueError("pflacco_broad_invariant must use lhs_100d")
    expected_counts = {
        "descriptor_cheap_invariant": 14,
        "pflacco_standard_invariant": 37,
        "pflacco_broad_invariant": 52,
    }
    for query_id, expected in expected_counts.items():
        spec = get_query_spec(query_id)
        if len(spec.feature_columns) != expected or len(set(spec.feature_columns)) != expected:
            raise ValueError(f"{query_id} must define exactly {expected} unique feature columns")
        if spec.preprocessing_id != QUERY_PREPROCESSING_VERSION:
            raise ValueError(f"{query_id} must use {QUERY_PREPROCESSING_VERSION}")
