import visual_redesign_full as visual


def test_interval_substitute_only_appears_in_second_half():
    events, players, *_ = visual.load_all()
    first, *_ = visual._half_network_data(events, players, visual.HOME_ID, 1)
    second, *_ = visual._half_network_data(events, players, visual.HOME_ID, 2)

    assert "Ousmane Dembélé" not in first.index
    assert "Ousmane Dembélé" in second.index


def test_first_half_log_does_not_name_interval_arrival():
    events, players, *_ = visual.load_all()
    *_, substitutions, _ = visual._half_network_data(
        events, players, visual.HOME_ID, 1
    )

    assert all(on_name == "—" for minute, on_name, _ in substitutions if minute == 45)

