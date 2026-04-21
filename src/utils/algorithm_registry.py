PYOD_ALGORITHMS = [
    "ECOD",
    "ABOD",
    "FastABOD",
    "COPOD",
    "MAD",
    "SOS",
    "QMCD",
    "KDE",
    "Sampling",
    "GMM",
    "PCA",
    "KPCA",
    "MCD",
    "CD",
    "OCSVM",
    "LMDD",
    "LOF",
    "COF",
    "(Incremental) COF",
    "CBLOF",
    "LOCI",
    "HBOS",
    "kNN",
    "AvgKNN",
    "MedKNN",
    "SOD",
    "ROD",
    "IForest",
    "INNE",
    "DIF",
    "FeatureBagging",
    "LSCP",
    "XGBOD",
    "LODA",
    "SUOD",
    "AutoEncoder",
    "VAE",
    "Beta-VAE",
    "SO_GAAL",
    "MO_GAAL",
    "DeepSVDD",
    "AnoGAN",
    "ALAD",
    "AE1SVM",
    "DevNet",
    "R-Graph",
    "LUNAR",
]

PYGOD_ALGORITHMS = [
    "SCAN",
    "GAE",
    "Radar",
    "ANOMALOUS",
    "ONE",
    "DOMINANT",
    "DONE",
    "AdONE",
    "AnomalyDAE",
    "GAAN",
    "DMGD",
    "OCGNN",
    "CoLA",
    "GUIDE",
    "CONAD",
    "GADNR",
    "CARD",
]

TSB_AD_ALGORITHM_DESCRIPTIONS = {
    "IForest": "Isolation Forest based anomaly detector supported by TSB-AD.",
    "LOF": "Local Outlier Factor based detector supported by TSB-AD.",
    "KNN": "k-nearest-neighbor distance based detector supported by TSB-AD.",
    "KMeansAD": "K-means based anomaly detector for time-series subsequences.",
    "CBLOF": "Cluster-based Local Outlier Factor detector.",
    "COPOD": "Copula-based outlier detector.",
    "HBOS": "Histogram-based outlier score detector.",
    "MCD": "Minimum covariance determinant based detector.",
    "OCSVM": "One-class SVM based detector.",
    "PCA": "Principal component analysis based detector.",
    "RobustPCA": "Robust PCA based detector.",
    "MatrixProfile": "Matrix Profile based anomaly detector.",
    "EIF": "Extended Isolation Forest supported by TSB-AD.",
    "Series2Graph": "Graph-based detector that converts time series into a graph.",
    "SAND": "Streaming anomaly detector with online clustering.",
    "SR": "Spectral residual anomaly detector.",
    "POLY": "Polynomial approximation based point anomaly detector.",
    "KShapeAD": "k-Shape based anomaly detector.",
    "AutoEncoder": "AutoEncoder based neural anomaly detector.",
    "LSTMAD": "LSTM based anomaly detector.",
    "Donut": "Variational autoencoder based time-series anomaly detector.",
    "CNN": "CNN based detector for time-series anomaly detection.",
    "OmniAnomaly": "Stochastic recurrent neural anomaly detector.",
    "USAD": "UnSupervised Anomaly Detection with adversarial autoencoders.",
    "AnomalyTransformer": "Transformer based anomaly detector with association discrepancy.",
    "TranAD": "Transformer based adversarial anomaly detector.",
    "TimesNet": "TimesNet model evaluated inside TSB-AD.",
    "FITS": "Frequency interpolation based detector.",
    "M2N2": "Distribution-shift aware neural anomaly detector.",
    "OFA": "Foundation-model based detector in TSB-AD.",
    "Lag-Llama": "Lag-Llama foundation model based detector.",
    "Chronos": "Chronos foundation model based detector.",
    "TimesFM": "TimesFM foundation model based detector.",
    "MOMENT": "MOMENT foundation model based detector.",
}

TSB_AD_ALGORITHMS = list(TSB_AD_ALGORITHM_DESCRIPTIONS.keys())

TSB_AD_SINGLE_INPUT_ALGORITHMS = [
    "CBLOF",
    "COF",
    "COPOD",
    "Chronos",
    "EIF",
    "FFT",
    "HBOS",
    "IForest",
    "KMeansAD",
    "KMeansAD_U",
    "KNN",
    "KShapeAD",
    "LOF",
    "Lag_Llama",
    "MOMENT_ZS",
    "MatrixProfile",
    "NORMA",
    "PCA",
    "POLY",
    "RobustPCA",
    "SR",
    "Series2Graph",
    "Sub_HBOS",
    "Sub_IForest",
    "Sub_KNN",
    "Sub_LOF",
    "Sub_PCA",
    "TimesFM",
]

TSB_AD_MODEL_SELECTION_CANDIDATES = TSB_AD_SINGLE_INPUT_ALGORITHMS
def normalize_algorithm_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def is_pyod_algorithm(name: str) -> bool:
    normalized = normalize_algorithm_name(name)
    return any(normalize_algorithm_name(candidate) == normalized for candidate in PYOD_ALGORITHMS)


def is_pygod_algorithm(name: str) -> bool:
    normalized = normalize_algorithm_name(name)
    return any(normalize_algorithm_name(candidate) == normalized for candidate in PYGOD_ALGORITHMS)


def _import_model_wrapper():
    try:
        import TSB_AD.model_wrapper as model_wrapper
    except Exception:
        return None
    return model_wrapper


def get_installed_tsb_ad_algorithms() -> list[str]:
    model_wrapper = _import_model_wrapper()
    if model_wrapper is None:
        return []

    discovered = []
    for name in dir(model_wrapper):
        if not name.startswith("run_"):
            continue
        algo = name[len("run_"):]
        if algo:
            discovered.append(algo)
    return sorted(set(discovered))


def get_effective_tsb_ad_algorithms() -> list[str]:
    installed = get_installed_tsb_ad_algorithms()
    return installed if installed else TSB_AD_ALGORITHMS


def is_tsb_ad_algorithm(name: str) -> bool:
    normalized = normalize_algorithm_name(name)
    return any(
        normalize_algorithm_name(candidate) == normalized
        for candidate in get_effective_tsb_ad_algorithms()
    )


def is_installed_tsb_ad_algorithm(name: str) -> bool:
    normalized = normalize_algorithm_name(name)
    installed = get_installed_tsb_ad_algorithms()
    return any(normalize_algorithm_name(candidate) == normalized for candidate in installed)
