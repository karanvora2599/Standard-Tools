"""
The manifest records the numerical environment that produced the model.

`git_commit_sha` and `package_version` say what source created a model.
Two machines at that commit can still differ on numpy, scikit-learn, the
BLAS behind the matrix work, whether the native extension was loaded and
whether it was current, and how many threads the kernels were allowed --
and those differences are exactly the ones that show up as coefficients
agreeing to four digits rather than twelve. The fingerprint is read from
the process, never declared, and carries nothing that identifies the host.
"""

import json
import platform
from importlib import metadata

from standard_quant_tools.modeling.agent.models import InspectModelInput
from standard_quant_tools.modeling.agent.tools import inspect_model
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.registry.environment import (
    PACKAGES,
    THREAD_VARIABLES,
    environment_fingerprint,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


class TestTheFingerprintIsReadFromTheProcess:
    def test_the_versions_are_the_installed_ones(self):
        fingerprint = environment_fingerprint()
        for name in ("numpy", "pandas", "scikit-learn", "scipy"):
            assert fingerprint["packages"][name] == metadata.version(name)
        assert fingerprint["python"] == platform.python_version()

    def test_an_absent_optional_package_is_none_not_missing(self, monkeypatch):
        """'not installed' is a fact about the environment, so the key is
        present with None rather than dropped."""
        import standard_quant_tools.modeling.registry.environment as env

        monkeypatch.setattr(env, "PACKAGES", ("numpy", "a-package-nobody-has"))
        fingerprint = env.environment_fingerprint()
        assert fingerprint["packages"]["a-package-nobody-has"] is None
        assert fingerprint["packages"]["numpy"] == metadata.version("numpy")

    def test_thread_caps_are_recorded_whether_or_not_set(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        threads = environment_fingerprint()["threads"]
        assert threads["OMP_NUM_THREADS"] == "3"
        assert threads["MKL_NUM_THREADS"] is None
        assert set(THREAD_VARIABLES) <= set(threads)
        assert threads["cpu_count"] >= 1

    def test_the_native_block_agrees_with_the_capability_report(self):
        from standard_quant_tools.modeling.capabilities import _native_detail

        native = environment_fingerprint()["native_extension"]
        detail = _native_detail()
        assert native["available"] == detail["available"]
        assert native["exports"] == detail["exports"]
        assert native["stale"] == detail["stale"]

    def test_it_is_plain_json_and_names_no_host(self):
        fingerprint = environment_fingerprint()
        text = json.dumps(fingerprint)
        assert platform.node() not in text or platform.node() == ""
        for forbidden in ("path", "hostname", "user"):
            assert forbidden not in fingerprint
            assert forbidden not in fingerprint["native_extension"]

    def test_every_declared_package_has_a_key(self):
        assert set(environment_fingerprint()["packages"]) == set(PACKAGES)


class TestTheManifestCarriesIt:
    def test_a_registered_model_records_the_environment(self, patched_multi_factory):
        dataset = build_dataset(
            DatasetSpec(
                universe=["AAA", "BBB", "CCC"],
                start="2022-01-01",
                end="2023-12-31",
                features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
                target=TargetSpec(horizon=5),
                benchmark="SPY",
            )
        )
        result = run_experiment(
            dataset,
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
            dataset_id="ds_env",
        )
        manifest = load_manifest(result["model_id"])
        assert manifest.environment["packages"]["scikit-learn"] == metadata.version(
            "scikit-learn"
        )
        assert manifest.environment["python"] == platform.python_version()
        lineage = inspect_model(
            InspectModelInput(model_id=result["model_id"], view="lineage")
        )
        assert lineage.data["environment"]["packages"]["numpy"] == metadata.version("numpy")
