from herd.services.manager import ProcessManager


def test_least_busy_load_balancer():
    """Verifies that ProcessManager selects the running model with the lowest in-flight request count."""
    pm = ProcessManager()
    pm.running_models = {
        "/path/model1": {"model_name": "model1", "in_flight": 5},
        "/path/model2": {"model_name": "model2", "in_flight": 1},
        "/path/model3": {"model_name": "model3", "in_flight": 3},
    }

    selected = pm.select_least_busy_pool_model(["model1", "model2", "model3"])
    assert selected == "model2"
