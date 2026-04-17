# Stage 3 Deep Expansion Plan

Status: implementation plan
Date: 2026-04-15
Depends on: stage3 initial commit (3c244b3)
Purpose: operationalize every remaining planned value in 3_training, add missing model families, build full tuning/validation engine

---

## Scope

이 문서는 Stage 3 (training layer)를 계획 문서의 최종 목표 수준까지 두껍게 만드는 계획이다.
현재 Stage 3는 28개 축, 143개 값, 45개 operational.
이 계획 완료 시 목표: 28개 축 유지, ~200개 값, ~120개 operational.

세 기둥으로 나뉜다:
1. **모델 확장** — 나머지 planned + 신규 모델 추가
2. **튜닝 엔진** — validation/search/budget/objective 전체 operational화
3. **프레임워크/실행 확장** — refit, outer_window, split, factor, feature builder

---

# Pillar 1: Model Family Expansion

## 1.1 AdaptiveLasso [planned -> operational]

### 알고리즘
Two-stage adaptive lasso:
1. Stage 1: OLS (또는 Ridge) fit -> coefficient estimates beta_init
2. Stage 2: Weighted Lasso with penalty weights w_j = 1/|beta_init_j|^gamma (gamma typically 1 or 2)

### 구현
```python
# execution/models/adaptive_lasso.py
class AdaptiveLassoExecutor:
    def __init__(self, gamma=1.0, init_estimator="ridge"):
        ...

    def fit(self, X_train, y_train):
        # Stage 1: initial estimates
        if self.init_estimator == "ridge":
            init = Ridge(alpha=1.0).fit(X_train, y_train)
        else:
            init = LinearRegression().fit(X_train, y_train)

        # Adaptive weights
        coefs = np.abs(init.coef_)
        weights = 1.0 / (coefs ** self.gamma + 1e-10)

        # Stage 2: weighted Lasso via rescaled X
        X_scaled = X_train / weights
        self.lasso = LassoCV(cv=5).fit(X_scaled, y_train)
        self.coef_ = self.lasso.coef_ / weights
        self.intercept_ = self.lasso.intercept_

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

### HP grid
```python
{
    "gamma": [0.5, 1.0, 2.0],
    "init_estimator": ["ridge", "ols"],
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
}
```

### Importance
- `linear_coefficients`: adaptive lasso coefficients -> sparsity-based importance
- zero coefficients = excluded variables

### 테스트
- 기존 Ridge/Lasso 테스트 패턴 재활용
- sparsity 검증: 일부 coefficient가 0인지 확인
- init_estimator="ols" vs "ridge" 결과 비교

---

## 1.2 SVR_linear [planned -> operational]

### 알고리즘
Support Vector Regression with linear kernel.

### 구현
```python
from sklearn.svm import LinearSVR

class SVRLinearExecutor:
    def fit(self, X_train, y_train):
        self.model = LinearSVR(
            epsilon=self.epsilon,
            C=self.C,
            max_iter=10000,
        ).fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
```

### HP grid
```python
{
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "epsilon": [0.001, 0.01, 0.1, 0.5],
}
```

### Importance
- `linear_coefficients`: LinearSVR has `.coef_` attribute
- permutation importance as fallback

---

## 1.3 SVR_rbf [planned -> operational]

### 알고리즘
Support Vector Regression with RBF (Gaussian) kernel.

### 구현
```python
from sklearn.svm import SVR

class SVRRbfExecutor:
    def fit(self, X_train, y_train):
        self.model = SVR(
            kernel="rbf",
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma_param,
        ).fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
```

### HP grid
```python
{
    "C": [0.1, 1.0, 10.0, 100.0],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
}
```

### Importance
- No native `.coef_` (nonlinear kernel) -> permutation importance only
- 추후 KernelSHAP 연동 가능

### 주의
- RBF SVR은 데이터 크기에 민감 (O(n^2 ~ n^3))
- FRED-MD 패널 (~130 변수, ~700 obs) 수준에서는 문제 없음
- 큰 데이터에서는 경고 필요

---

## 1.4 Linear Boosting Models [신규 추가]

### 1.4.1 ComponentwiseBoosting (L2Boost / componentwise gradient boosting)

경제학 forecasting에서 자주 쓰이는 모델. Bai & Ng (2009), Buchen & Wohlrabe (2011) 등에서 사용.

### 알고리즘
- 각 iteration에서 하나의 변수(component)만 선택
- 선택된 변수로 simple OLS/Ridge -> residual 업데이트
- early stopping으로 implicit regularization

### 구현
```python
class ComponentwiseBoostingExecutor:
    def __init__(self, n_iterations=100, learning_rate=0.1,
                 base_learner="ols", early_stop_rounds=10):
        ...

    def fit(self, X_train, y_train):
        residuals = y_train.copy()
        self.coefs = np.zeros(X_train.shape[1])
        self.intercept_ = np.mean(y_train)
        residuals -= self.intercept_
        self.selected_features = []

        for t in range(self.n_iterations):
            # Find best single-variable fit
            best_j, best_coef, best_sse = None, None, np.inf
            for j in range(X_train.shape[1]):
                xj = X_train[:, j]
                coef = np.dot(xj, residuals) / (np.dot(xj, xj) + 1e-10)
                sse = np.sum((residuals - coef * xj) ** 2)
                if sse < best_sse:
                    best_j, best_coef, best_sse = j, coef, sse

            self.coefs[best_j] += self.learning_rate * best_coef
            residuals -= self.learning_rate * best_coef * X_train[:, best_j]
            self.selected_features.append(best_j)

    def predict(self, X):
        return X @ self.coefs + self.intercept_
```

### HP grid
```python
{
    "n_iterations": [50, 100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "early_stop_rounds": [5, 10, 20, None],
}
```

### Importance
- Selection frequency: 각 변수가 몇 번 선택됐는지
- Final coefficient magnitude
- Cumulative coefficient path

### 1.4.2 BoostingRidge (Friedman-style L2 boosted Ridge)

### 알고리즘
- 각 iteration에서 전체 변수를 사용하는 Ridge fit
- Shrinkage/learning rate로 coefficient 점진적 축적
- GBM의 linear 버전

### 구현
```python
class BoostingRidgeExecutor:
    def __init__(self, n_iterations=100, learning_rate=0.1, ridge_alpha=1.0):
        ...

    def fit(self, X_train, y_train):
        residuals = y_train.copy()
        self.intercept_ = np.mean(y_train)
        residuals -= self.intercept_
        self.coefs = np.zeros(X_train.shape[1])

        for t in range(self.n_iterations):
            ridge = Ridge(alpha=self.ridge_alpha).fit(X_train, residuals)
            self.coefs += self.learning_rate * ridge.coef_
            residuals -= self.learning_rate * ridge.predict(X_train)

    def predict(self, X):
        return X @ self.coefs + self.intercept_
```

### HP grid
```python
{
    "n_iterations": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "ridge_alpha": [0.01, 0.1, 1.0, 10.0],
}
```

### 1.4.3 BoostingLasso (Friedman-style L1 boosted Lasso)

유사 구조, 각 iteration에서 Lasso fit. Sparse update.

```python
class BoostingLassoExecutor:
    # Same pattern as BoostingRidge but uses Lasso per iteration
    ...
```

---

## 1.5 추가 고려 모델

### 1.5.1 PCR (Principal Component Regression)
- PCA -> 선택된 factor 수로 truncate -> OLS on factors
- `dimensionality_reduction_policy=pca` + `feature_builder=factors_plus_AR`와 연동
- HP: n_components

### 1.5.2 PLS (Partial Least Squares)
- sklearn.cross_decomposition.PLSRegression
- supervised dimensionality reduction
- HP: n_components

### 1.5.3 factor_augmented_linear
- PCA factors + target lags -> Ridge/Lasso/OLS
- `feature_builder=factors_plus_AR` operational화와 동시 진행
- Diffusion index forecasting (Stock & Watson, 2002) 구현

### 1.5.4 HuberRegressor
- sklearn.linear_model.HuberRegressor
- 이상치에 robust한 linear regression
- HP: epsilon (outlier threshold), alpha (regularization)

### 1.5.5 QuantileLinear
- sklearn.linear_model.QuantileRegressor (sklearn >= 1.0)
- `forecast_object=quantile`과 연동
- HP: quantile, alpha
- Note: planned for now, requires forecast_object infrastructure

### 1.5.6 CatBoost
- catboost.CatBoostRegressor
- Optional dependency (xgboost/lightgbm과 같은 패턴)
- HP: iterations, learning_rate, depth, l2_leaf_reg

---

## 1.6 모델 registry 확장 목표

현재 15개 -> 목표 ~25개:

| model_family | status 목표 | category |
|---|---|---|
| ar | operational (유지) | benchmark |
| ols | operational (유지) | linear |
| ridge | operational (유지) | linear_ml |
| lasso | operational (유지) | linear_ml |
| elasticnet | operational (유지) | linear_ml |
| bayesianridge | operational (유지) | linear_ml |
| adaptivelasso | **operational** | linear_ml |
| huber | **operational** (신규) | robust_linear |
| svr_linear | **operational** | kernel |
| svr_rbf | **operational** | kernel |
| componentwise_boosting | **operational** (신규) | linear_boosting |
| boosting_ridge | **operational** (신규) | linear_boosting |
| boosting_lasso | **operational** (신규) | linear_boosting |
| pcr | **operational** (신규) | factor_linear |
| pls | **operational** (신규) | factor_linear |
| factor_augmented_linear | **operational** (신규) | factor_linear |
| quantile_linear | **planned** (신규) | quantile |
| randomforest | operational (유지) | tree_ensemble |
| extratrees | operational (유지) | tree_ensemble |
| gbm | operational (유지) | tree_ensemble |
| xgboost | operational (유지) | tree_ensemble |
| lightgbm | operational (유지) | tree_ensemble |
| catboost | **operational** (신규) | tree_ensemble |
| mlp | operational (유지) | neural |

---

# Pillar 2: Tuning Engine

> 현재 상태: search_algorithm, tuning_budget, tuning_objective, validation_size_rule, validation_location 모두 0 operational.
> 모델은 현재 고정 HP 또는 sklearn 내부 CV만 사용.
> 목표: 외부 tuning loop 전체 구축.

## 2.1 Architecture

```
macrocast/
  tuning/
    __init__.py
    types.py          # TuningSpec, TuningResult, TuningTrial, HPDistribution
    engine.py         # run_tuning(), main orchestrator
    search/
      __init__.py
      grid.py         # grid_search
      random.py       # random_search
      bayesian.py     # bayesian_optimization (optuna backend)
      genetic.py      # genetic_algorithm (pure numpy)
    validation/
      __init__.py
      splitter.py     # temporal CV splitters
      scorer.py       # validation scoring
    budget.py         # budget enforcement (max_trials, max_time, early_stop)
    hp_spaces.py      # per-model default HP spaces
```

## 2.2 Core types

```python
@dataclass(frozen=True)
class HPDistribution:
    type: Literal["float", "int", "categorical", "log_float"]
    low: float | None = None
    high: float | None = None
    choices: tuple | None = None
    log: bool = False

    def sample(self, rng: np.random.RandomState):
        if self.type == "float":
            return rng.uniform(self.low, self.high)
        elif self.type == "log_float":
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high)))
        elif self.type == "int":
            return rng.randint(self.low, self.high + 1)
        elif self.type == "categorical":
            return self.choices[rng.randint(len(self.choices))]

@dataclass(frozen=True)
class TuningSpec:
    search_algorithm: str
    tuning_objective: str
    tuning_budget: dict[str, Any]
    hp_space: dict[str, HPDistribution]
    validation_size_rule: str
    validation_size_config: dict[str, Any]
    validation_location: str
    embargo_gap: str
    embargo_gap_size: int
    seed: int | None

@dataclass(frozen=True)
class TuningTrial:
    trial_id: int
    hp_values: dict[str, Any]
    validation_score: float
    fit_time_seconds: float
    status: Literal["completed", "failed", "pruned"]

@dataclass(frozen=True)
class TuningResult:
    best_hp: dict[str, Any]
    best_score: float
    all_trials: tuple[TuningTrial, ...]
    search_algorithm: str
    total_trials: int
    total_time_seconds: float
    convergence_info: dict[str, Any]
```

## 2.3 Search Algorithms

### 2.3.1 grid_search [planned -> operational]

Exhaustive enumeration of all HP combinations on a discrete grid.

```python
def grid_search(
    model_factory: Callable,
    hp_grid: dict[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_splitter: TemporalCVSplitter,
    scorer: Callable,
    budget: TuningBudget,
) -> TuningResult:
    all_combos = list(itertools.product(*hp_grid.values()))
    keys = list(hp_grid.keys())
    trials = []
    for i, combo in enumerate(all_combos):
        if budget.exceeded(trials):
            break
        hp = dict(zip(keys, combo))
        score = _evaluate_hp(model_factory, hp, X_train, y_train, validation_splitter, scorer)
        trial = TuningTrial(trial_id=i, hp_values=hp, validation_score=score, ...)
        trials.append(trial)
        budget.update(score)
    best = min(trials, key=lambda t: t.validation_score)
    return TuningResult(best_hp=best.hp_values, best_score=best.validation_score, ...)
```

### 2.3.2 random_search [planned -> operational]

Random sampling from HP distributions with budget enforcement.

```python
def random_search(
    model_factory: Callable,
    hp_space: dict[str, HPDistribution],
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_splitter: TemporalCVSplitter,
    scorer: Callable,
    budget: TuningBudget,
    random_state: int = 42,
) -> TuningResult:
    rng = np.random.RandomState(random_state)
    trials = []
    trial_id = 0
    while not budget.exceeded(trials):
        hp = {k: dist.sample(rng) for k, dist in hp_space.items()}
        score = _evaluate_hp(model_factory, hp, X_train, y_train, validation_splitter, scorer)
        trials.append(TuningTrial(trial_id=trial_id, hp_values=hp, validation_score=score, ...))
        budget.update(score)
        trial_id += 1
    ...
```

### 2.3.3 bayesian_optimization [registry_only -> operational]

Backend: **optuna** (TPE sampler by default)

```python
def bayesian_optimization(
    model_factory: Callable,
    hp_space: dict[str, HPDistribution],
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_splitter: TemporalCVSplitter,
    scorer: Callable,
    budget: TuningBudget,
    random_state: int = 42,
) -> TuningResult:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hp = {}
        for name, dist in hp_space.items():
            if dist.type == "log_float":
                hp[name] = trial.suggest_float(name, dist.low, dist.high, log=True)
            elif dist.type == "float":
                hp[name] = trial.suggest_float(name, dist.low, dist.high)
            elif dist.type == "int":
                hp[name] = trial.suggest_int(name, dist.low, dist.high)
            elif dist.type == "categorical":
                hp[name] = trial.suggest_categorical(name, list(dist.choices))
        return _evaluate_hp(model_factory, hp, X_train, y_train, validation_splitter, scorer)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=budget.max_trials,
        timeout=budget.max_time_seconds,
    )

    return TuningResult(
        best_hp=study.best_params,
        best_score=study.best_value,
        all_trials=_convert_optuna_trials(study.trials),
        search_algorithm="bayesian_optimization",
        total_trials=len(study.trials),
        ...
    )
```

Why optuna:
- TPE is efficient for mixed-type HP spaces
- Pruning support for early stopping
- Handles conditional parameters
- Well-maintained, widely adopted
- Import-guarded: no mandatory dependency

### 2.3.4 genetic_algorithm [future -> operational]

Pure numpy implementation. No DEAP dependency.

```python
def genetic_algorithm(
    model_factory: Callable,
    hp_space: dict[str, HPDistribution],
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_splitter: TemporalCVSplitter,
    scorer: Callable,
    budget: TuningBudget,
    random_state: int = 42,
    population_size: int = 50,
    n_generations: int = 20,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    tournament_size: int = 3,
    elitism_count: int = 2,
) -> TuningResult:
    rng = np.random.RandomState(random_state)

    # Initialize population
    population = [_random_individual(hp_space, rng) for _ in range(population_size)]
    fitness = np.array([
        _evaluate_hp(model_factory, ind, X_train, y_train, validation_splitter, scorer)
        for ind in population
    ])

    all_trials = []
    generation_log = []

    for gen in range(n_generations):
        if budget.exceeded(all_trials):
            break

        # Elitism: preserve top individuals
        elite_idx = np.argsort(fitness)[:elitism_count]
        elites = [(population[i].copy(), fitness[i]) for i in elite_idx]

        # Tournament selection
        parents = _tournament_select(population, fitness, tournament_size, len(population), rng)

        # Crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if rng.random() < crossover_prob:
                c1, c2 = _crossover(parents[i], parents[i + 1], hp_space, rng)
            else:
                c1, c2 = parents[i].copy(), parents[i + 1].copy()
            offspring.extend([c1, c2])

        # Mutation
        for ind in offspring:
            if rng.random() < mutation_prob:
                _mutate(ind, hp_space, rng)

        # Evaluate offspring
        offspring_fitness = np.array([
            _evaluate_hp(model_factory, ind, X_train, y_train, validation_splitter, scorer)
            for ind in offspring
        ])

        # Replace worst with elites
        population = offspring[:population_size]
        fitness = offspring_fitness[:population_size]
        for j, (elite_ind, elite_fit) in enumerate(elites):
            worst_idx = np.argmax(fitness)
            if elite_fit < fitness[worst_idx]:
                population[worst_idx] = elite_ind
                fitness[worst_idx] = elite_fit

        # Log
        gen_best = np.min(fitness)
        gen_mean = np.mean(fitness)
        generation_log.append({"generation": gen, "best": gen_best, "mean": gen_mean})

        all_trials.extend([
            TuningTrial(trial_id=len(all_trials) + k, hp_values=ind, validation_score=f, ...)
            for k, (ind, f) in enumerate(zip(offspring, offspring_fitness))
        ])

    best_idx = np.argmin(fitness)
    return TuningResult(
        best_hp=population[best_idx],
        best_score=fitness[best_idx],
        all_trials=tuple(all_trials),
        search_algorithm="genetic_algorithm",
        total_trials=len(all_trials),
        convergence_info={"generation_log": generation_log},
        ...
    )
```

### Crossover operators

```python
def _crossover(parent1, parent2, hp_space, rng):
    child1, child2 = {}, {}
    for name, dist in hp_space.items():
        if dist.type == "categorical":
            # Uniform crossover for categorical
            if rng.random() < 0.5:
                child1[name], child2[name] = parent1[name], parent2[name]
            else:
                child1[name], child2[name] = parent2[name], parent1[name]
        else:
            # BLX-alpha crossover for continuous/integer
            alpha = 0.5
            p1, p2 = float(parent1[name]), float(parent2[name])
            d = abs(p1 - p2)
            low = min(p1, p2) - alpha * d
            high = max(p1, p2) + alpha * d
            low = max(low, dist.low)
            high = min(high, dist.high)
            c1 = rng.uniform(low, high)
            c2 = rng.uniform(low, high)
            if dist.type == "int":
                c1, c2 = int(round(c1)), int(round(c2))
            if dist.log:
                c1 = np.clip(c1, dist.low, dist.high)
                c2 = np.clip(c2, dist.low, dist.high)
            child1[name], child2[name] = c1, c2
    return child1, child2
```

### Mutation operators

```python
def _mutate(individual, hp_space, rng, sigma_frac=0.1):
    name = list(hp_space.keys())[rng.randint(len(hp_space))]
    dist = hp_space[name]
    if dist.type == "categorical":
        individual[name] = dist.choices[rng.randint(len(dist.choices))]
    elif dist.type in ("float", "log_float"):
        sigma = (dist.high - dist.low) * sigma_frac
        individual[name] = np.clip(
            individual[name] + rng.normal(0, sigma),
            dist.low, dist.high,
        )
    elif dist.type == "int":
        delta = max(1, int((dist.high - dist.low) * sigma_frac))
        individual[name] = np.clip(
            individual[name] + rng.randint(-delta, delta + 1),
            dist.low, dist.high,
        )
```

### GA-specific configuration
```python
{
    "population_size": [30, 50, 100],
    "n_generations": [10, 20, 50],
    "crossover_prob": [0.5, 0.7, 0.9],
    "mutation_prob": [0.1, 0.2, 0.3],
    "tournament_size": [2, 3, 5],
    "elitism_count": [1, 2, 5],
}
```

---

## 2.4 Validation Engine

### 2.4.1 Temporal CV Splitters

```python
class TemporalCVSplitter:
    """Base class for time-series-aware cross-validation."""
    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

class LastBlockSplitter(TemporalCVSplitter):
    """Single held-out block at end of training window."""
    def __init__(self, validation_size: int, embargo_gap: int = 0):
        self.validation_size = validation_size
        self.embargo_gap = embargo_gap

    def split(self, n_samples):
        val_start = n_samples - self.validation_size
        train_end = val_start - self.embargo_gap
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, n_samples)
        yield train_idx, val_idx

class RollingBlocksSplitter(TemporalCVSplitter):
    """Multiple rolling validation blocks."""
    def __init__(self, n_blocks: int = 3, block_size: int | None = None, embargo_gap: int = 0):
        ...
    def split(self, n_samples):
        # Divide into n_blocks segments, use each as val with prior as train
        ...

class ExpandingValidationSplitter(TemporalCVSplitter):
    """Expanding window validation (walk-forward within training)."""
    def __init__(self, min_train_size: int, step_size: int = 1, embargo_gap: int = 0):
        ...
    def split(self, n_samples):
        for t in range(self.min_train_size, n_samples - self.embargo_gap, self.step_size):
            train_idx = np.arange(0, t)
            val_idx = np.arange(t + self.embargo_gap, min(t + self.embargo_gap + 1, n_samples))
            if len(val_idx) > 0:
                yield train_idx, val_idx

class BlockedKFoldSplitter(TemporalCVSplitter):
    """Blocked k-fold respecting temporal order."""
    def __init__(self, n_splits: int = 5, embargo_gap: int = 0):
        ...
    def split(self, n_samples):
        fold_size = n_samples // self.n_splits
        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = min(val_start + fold_size, n_samples)
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([
                np.arange(0, max(0, val_start - self.embargo_gap)),
                np.arange(min(n_samples, val_end + self.embargo_gap), n_samples),
            ])
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
```

### 2.4.2 Validation Size Rules

```python
def resolve_validation_size(
    rule: str,
    total_train_size: int,
    config: dict[str, Any],
) -> int:
    if rule == "ratio":
        return max(1, int(total_train_size * config["ratio"]))
    elif rule == "fixed_n":
        return min(config["n"], total_train_size // 2)
    elif rule == "fixed_years":
        obs_per_year = config.get("obs_per_year", 12)
        return min(config["years"] * obs_per_year, total_train_size // 2)
    else:
        raise ValueError(f"unknown validation_size_rule: {rule}")
```

### 2.4.3 Scoring Functions

```python
SCORERS = {
    "validation_mse": lambda y, yhat: np.mean((y - yhat) ** 2),
    "validation_rmse": lambda y, yhat: np.sqrt(np.mean((y - yhat) ** 2)),
    "validation_mae": lambda y, yhat: np.mean(np.abs(y - yhat)),
}

def get_scorer(objective: str) -> Callable:
    if objective not in SCORERS:
        raise ValueError(f"unknown tuning_objective: {objective}")
    return SCORERS[objective]
```

---

## 2.5 Budget Enforcement

```python
@dataclass
class TuningBudget:
    max_trials: int | None = None
    max_time_seconds: float | None = None
    early_stop_trials: int | None = None

    def __post_init__(self):
        self._start_time = time.time()
        self._no_improvement_count = 0
        self._best_score = float("inf")
        self._trial_count = 0

    def exceeded(self, trials: list | None = None) -> bool:
        if self.max_trials is not None and self._trial_count >= self.max_trials:
            return True
        if self.max_time_seconds is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self.max_time_seconds:
                return True
        if self.early_stop_trials is not None:
            if self._no_improvement_count >= self.early_stop_trials:
                return True
        return False

    def update(self, score: float):
        self._trial_count += 1
        if score < self._best_score:
            self._best_score = score
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
```

---

## 2.6 Per-Model HP Space Definitions

```python
MODEL_HP_SPACES: dict[str, dict[str, HPDistribution]] = {
    "ridge": {
        "alpha": HPDistribution("log_float", 1e-4, 1e4, log=True),
    },
    "lasso": {
        "alpha": HPDistribution("log_float", 1e-6, 1e2, log=True),
    },
    "elasticnet": {
        "alpha": HPDistribution("log_float", 1e-6, 1e2, log=True),
        "l1_ratio": HPDistribution("float", 0.01, 0.99),
    },
    "adaptivelasso": {
        "gamma": HPDistribution("float", 0.5, 3.0),
        "init_estimator": HPDistribution("categorical", choices=("ridge", "ols")),
    },
    "svr_linear": {
        "C": HPDistribution("log_float", 0.01, 1000, log=True),
        "epsilon": HPDistribution("log_float", 0.001, 1.0, log=True),
    },
    "svr_rbf": {
        "C": HPDistribution("log_float", 0.01, 1000, log=True),
        "epsilon": HPDistribution("log_float", 0.001, 1.0, log=True),
        "gamma": HPDistribution("log_float", 1e-5, 1.0, log=True),
    },
    "componentwise_boosting": {
        "n_iterations": HPDistribution("int", 50, 500),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
    },
    "boosting_ridge": {
        "n_iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
        "ridge_alpha": HPDistribution("log_float", 0.01, 100, log=True),
    },
    "boosting_lasso": {
        "n_iterations": HPDistribution("int", 50, 300),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
        "lasso_alpha": HPDistribution("log_float", 1e-4, 10, log=True),
    },
    "randomforest": {
        "n_estimators": HPDistribution("int", 100, 1000),
        "max_depth": HPDistribution("int", 3, 20),
        "min_samples_split": HPDistribution("int", 2, 20),
        "max_features": HPDistribution("categorical", choices=("sqrt", "log2", 0.5, 0.8, 1.0)),
    },
    "xgboost": {
        "n_estimators": HPDistribution("int", 50, 500),
        "max_depth": HPDistribution("int", 2, 10),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
        "subsample": HPDistribution("float", 0.5, 1.0),
        "colsample_bytree": HPDistribution("float", 0.3, 1.0),
        "reg_alpha": HPDistribution("log_float", 1e-8, 10, log=True),
        "reg_lambda": HPDistribution("log_float", 1e-8, 10, log=True),
    },
    "lightgbm": {
        "n_estimators": HPDistribution("int", 50, 500),
        "num_leaves": HPDistribution("int", 8, 128),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
        "subsample": HPDistribution("float", 0.5, 1.0),
        "colsample_bytree": HPDistribution("float", 0.3, 1.0),
        "reg_alpha": HPDistribution("log_float", 1e-8, 10, log=True),
        "reg_lambda": HPDistribution("log_float", 1e-8, 10, log=True),
    },
    "catboost": {
        "iterations": HPDistribution("int", 100, 1000),
        "learning_rate": HPDistribution("log_float", 0.005, 0.3, log=True),
        "depth": HPDistribution("int", 2, 10),
        "l2_leaf_reg": HPDistribution("log_float", 1e-3, 100, log=True),
    },
    "mlp": {
        "hidden_layer_sizes": HPDistribution("categorical", choices=((64,), (128,), (64, 32), (128, 64), (256, 128))),
        "alpha": HPDistribution("log_float", 1e-6, 1e-1, log=True),
        "learning_rate_init": HPDistribution("log_float", 1e-4, 1e-1, log=True),
    },
    "pcr": {
        "n_components": HPDistribution("int", 1, 20),
    },
    "pls": {
        "n_components": HPDistribution("int", 1, 20),
    },
    "huber": {
        "epsilon": HPDistribution("float", 1.01, 5.0),
        "alpha": HPDistribution("log_float", 1e-6, 1e2, log=True),
    },
}
```

---

# Pillar 3: Framework / Execution Expansion

## 3.1 outer_window

### anchored_rolling [planned -> operational]
- Fixed start, expanding train set up to max_window_size, then rolling
- 기존 expanding/rolling 실행 루프에 조건 추가

```python
if outer_window == "anchored_rolling":
    if current_train_size > max_window_size:
        train_start = current_end - max_window_size
    else:
        train_start = anchor_date  # fixed
```

## 3.2 refit_policy

### refit_every_k_steps [planned -> operational]
- OOS 루프에서 k step마다만 모델 재적합
- k=1이면 refit_every_step와 동일

### fit_once_predict_many [planned -> operational]
- 한 번만 fit하고 전체 OOS에 대해 predict
- 가장 빠르지만 비정상성 대응 불가

## 3.3 split_family expansion

### blocked_kfold [planned -> operational]
- 시계열 순서 존중하는 blocked k-fold
- embargo gap 적용 가능

### expanding_cv [planned -> operational]
- Walk-forward within training: 점진적으로 training set 확장하며 validation

### rolling_cv [planned -> operational]
- Walk-forward with fixed window size

## 3.4 factor_count [all planned -> operational]

### fixed
- n_components를 leaf_config에서 직접 지정

### cv_select
- validation_mse 기준으로 factor 수 1~max_k sweep해서 최적 선택

### BaiNg_rule
- Bai & Ng (2002) information criteria: IC_p1, IC_p2, IC_p3
- Eigenvalue ratio 기반 factor 수 결정
- kmax = min(floor(sqrt(min(N, T))), 20)으로 상한

## 3.5 feature_builder expansion

### factors_plus_AR [planned -> operational]
- PCA factor extraction -> truncated factors + target lags -> model fit
- dimensionality_reduction_policy=pca 연동
- factor_count axis 참조

### factor_pca [planned -> operational]
- Pure PCA factor regression (PCR과 동일)
- target lags 없이 factor만 사용

## 3.6 Early stopping

### validation_patience [planned -> operational]
- 연속 N회 validation score 개선 없으면 boosting/neural iteration 중단
- 주로 gbm, xgboost, lightgbm, mlp에 적용

### loss_plateau [planned -> operational]
- 절대 개선폭이 threshold (예: 1e-6) 이하이면 중단

## 3.7 Convergence handling

### fallback_to_safe_hp [planned -> operational]
- 모델 수렴 실패 시 보수적 HP로 자동 전환
- 예: MLP 발산 시 alpha=0.01, hidden=(64,)로 fallback
- 예: XGBoost OOM 시 max_depth=3, n_estimators=100으로 fallback

---

# Integration with Existing Execution Layer

## Current execution/build.py flow:
1. Load raw data
2. Build features (autoreg or raw_panel)
3. For each OOS date: fit model, predict, collect
4. Compute metrics
5. Write artifacts

## New flow with tuning:
1. Load raw data
2. Build features
3. **If tuning_spec present:**
   a. Build validation splitter from tuning_spec
   b. For each OOS date:
      - Extract training window
      - **Run tuning engine** on training window -> best_hp
      - Fit model with best_hp on full training window
      - Predict
   c. Record tuning_result per OOS date (or per refit cycle)
4. Compute metrics
5. Write artifacts + tuning_result artifact

## Tuning artifact
```json
{
    "tuning_method": "bayesian_optimization",
    "total_trials": 50,
    "best_hp": {"alpha": 0.0123, "l1_ratio": 0.45},
    "best_validation_score": 0.0045,
    "convergence_info": {...},
    "per_oos_date_hp": [
        {"date": "2010-01", "best_hp": {...}, "n_trials": 50},
        ...
    ]
}
```

---

# Implementation Order

## Phase 1: Tuning Engine Foundation (blocking everything else)
1. `macrocast/tuning/types.py`
2. `macrocast/tuning/budget.py`
3. `macrocast/tuning/validation/splitter.py` (4 splitters)
4. `macrocast/tuning/validation/scorer.py` (3 scorers)
5. `macrocast/tuning/search/grid.py`
6. `macrocast/tuning/search/random.py`
7. `macrocast/tuning/engine.py`
8. Wire into execution/build.py

Estimated: ~600-800 LOC new code

## Phase 2: Model Family Expansion
1. AdaptiveLasso
2. SVR_linear + SVR_rbf
3. ComponentwiseBoosting + BoostingRidge + BoostingLasso
4. PCR + PLS
5. HuberRegressor
6. CatBoost
7. factor_augmented_linear

Estimated: ~400-600 LOC new code (each model ~50-80 LOC)

## Phase 3: Advanced Search
1. `macrocast/tuning/search/bayesian.py` (optuna)
2. `macrocast/tuning/search/genetic.py` (pure numpy)
3. `macrocast/tuning/hp_spaces.py` (all model HP spaces)

Estimated: ~500-700 LOC new code

## Phase 4: Framework Expansion
1. anchored_rolling
2. refit_every_k_steps + fit_once_predict_many
3. factor_count (fixed, cv_select, BaiNg)
4. feature_builder (factors_plus_AR, factor_pca)
5. Early stopping + convergence handling

Estimated: ~300-500 LOC new code

## Phase 5: Registry Update + Tests
1. Update all registry statuses
2. Per-model end-to-end tests
3. Tuning engine unit tests
4. GA convergence tests
5. Bayesian vs grid comparison tests
6. Manifest tuning_result recording tests

Estimated: ~500-800 LOC tests

---

# Expected Final State

## Model family
- 12 operational -> **~22 operational** + 3 planned/registry_only
- 새 카테고리: linear_boosting (3), factor_linear (3), robust_linear (1), kernel (2)

## Tuning
- 0 operational -> **5 operational** (grid, random, bayesian, GA, manual_fixed_hp)
- Validation: 0 -> **4 operational** (last_block, rolling_blocks, expanding, blocked_cv)
- Budget: 0 -> **3 operational** (max_trials, max_time, early_stop_trials)
- Objective: 0 -> **3 operational** (mse, rmse, mae)

## Framework
- outer_window: +1 operational (anchored_rolling)
- refit_policy: +2 operational (every_k, fit_once)
- split_family: +3 operational (blocked_kfold, expanding_cv, rolling_cv)
- factor_count: +3 operational (fixed, cv_select, BaiNg)
- feature_builder: +2 operational (factors_plus_AR, factor_pca)
- early_stopping: +2 operational (patience, plateau)
- convergence: +1 operational (fallback_to_safe_hp)

## Total Stage 3 delta
- values: 143 -> ~200
- operational: 45 -> ~120

---

# Dependencies

External (optional, import-guarded):
- `optuna` -- bayesian_optimization
- `catboost` -- CatBoost model
- `xgboost` -- already used
- `lightgbm` -- already used

No new mandatory dependencies. All search algorithms work without optional deps (grid, random, GA are pure numpy).

---

# Estimated Total New Code

| Area | LOC |
|------|-----|
| Tuning engine (types + budget + validation + search + engine) | ~800 |
| Model executors (10 new models) | ~600 |
| Advanced search (bayesian + GA) | ~600 |
| Framework expansion | ~400 |
| HP space definitions | ~200 |
| Tests | ~800 |
| **Total** | **~3,400** |

Current execution/build.py: 1,149 LOC
After expansion: ~2,000-2,500 LOC (split across execution/ and tuning/)
