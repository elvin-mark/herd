from herd.services.agent import AgentSession


def test_agent_plan_creation_and_update():
    """Verifies that create_plan and update_plan tools populate and update the plan checklist."""
    session = AgentSession(model_name="test-model", gateway_url="http://127.0.0.1:11434")

    # 1. Create plan
    res = session.registry.execute(
        "create_plan",
        '["Inspect files", "Add router", "Run pytest"]',
    )
    assert "Created task plan with 3 steps" in res
    assert len(session.plan) == 3

    summary = session._get_plan_summary()
    assert "Inspect files" in summary
    assert "Add router" in summary

    # 2. Update plan step 1 status
    res_update = session.registry.execute(
        "update_plan",
        '{"step_number": 1, "status": "completed", "notes": "Inspected successfully"}',
    )
    assert "Updated step 1 status to 'completed'" in res_update
    assert session.plan[0]["status"] == "completed"

    summary_after = session._get_plan_summary()
    assert "[✓] Step 1: Inspect files (Inspected successfully)" in summary_after


def test_agent_batch_action_execution():
    """Verifies that ToolRegistry and AgentSession execute tools cleanly."""
    session = AgentSession(model_name="test-model", gateway_url="http://127.0.0.1:11434")

    res_list = session.registry.execute("list_dir", ".")
    assert isinstance(res_list, str)

    res_memory = session.registry.execute("save_memory", "Always test before submitting commits")
    assert "Successfully saved to long-term memory" in res_memory


def test_agent_history_compression():
    """Verifies that _compress_history_if_needed compresses older turns when context budget is exceeded."""
    session = AgentSession(model_name="test-model", gateway_url="http://127.0.0.1:11434")

    for i in range(15):
        session.history.append(
            {
                "role": "assistant",
                "content": f'{{"thought": "Thought {i}", "action": "list_dir", "action_input": "."}}',
            }
        )
        session.history.append(
            {
                "role": "user",
                "content": f"Observation: Found file_{i}.py with heavy content padding "
                + ("x" * 600),
            }
        )

    initial_len = len(session.history)
    assert initial_len > 25

    session._compress_history_if_needed(max_turn_messages=6, max_char_budget=2000)

    compressed_len = len(session.history)
    assert compressed_len < initial_len
    assert session.history[0]["role"] == "system"
    assert "📜 Compressed Prior Conversation History" in session.history[2]["content"]
