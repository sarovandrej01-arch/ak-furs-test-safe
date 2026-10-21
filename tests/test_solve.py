"""Проверка единой команды, точного JSON и согласованности Q1–Q4."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import pandas as pd

import solve as runner
from src.bank import paid_order_ids, reconcile_bank
from src.chats import analyze_chats, violation_ids
from src.chats import ChatAnalysis
from src.crm import load_clean_crm, summarize_q1
from src.loaders import DATA_DIR, load_answers_template, load_bank, load_catalog, load_chats
from src.pricing import find_price_violations

ROOT = Path(__file__).resolve().parents[1]


def run_cli(folder, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "solve.py"), str(folder)],
                          cwd=cwd, capture_output=True, encoding="utf-8", timeout=60)


@pytest.fixture(scope="module")
def actual():
    completed = run_cli(DATA_DIR)
    assert completed.returncode == 0, completed.stderr
    return completed, json.loads(completed.stdout, parse_float=Decimal)


@pytest.fixture
def copied_data(tmp_path):
    folder = tmp_path / "другая папка данных"
    folder.mkdir()
    for filename in runner.FILES:
        shutil.copyfile(DATA_DIR / filename, folder / filename)
    return folder


def test_success_structure_types_and_clean_stdout(actual):
    completed, answers = actual
    assert completed.stderr == ""
    assert completed.stdout.startswith("{\n") and completed.stdout.endswith("}\n")
    assert any("А" <= char <= "я" for char in completed.stdout)
    assert set(answers) == set(load_answers_template())
    runner.validate_answers(answers, load_answers_template())
    assert all(type(order) is int for order in answers["q2a_paid_orders_without_payment"])
    assert type(answers["q2b_unidentified_payments_sum_rub"]) is Decimal
    assert type(answers["notes"]) is str
    assert "float" not in completed.stdout


def test_exact_decimal_serialization_without_float():
    value = Decimal("12345678901234567890.01")
    text = runner.dumps_exact({"сумма": value, "строка": 'текст "0.01"\n'})
    assert '"сумма": 12345678901234567890.01' in text
    assert json.loads(text, parse_float=Decimal)["сумма"] == value
    assert json.loads(text)["строка"] == 'текст "0.01"\n'


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), 0.1, {1: "ключ"}])
def test_json_rejects_unsafe_values(value):
    with pytest.raises((TypeError, ValueError)):
        runner.dumps_exact(value)


def test_sorted_unique_arrays(actual):
    _, answers = actual
    for key in ["q2a_paid_orders_without_payment", "q4_chats_meeting_before_price"]:
        assert answers[key] == sorted(set(answers[key]))


def test_identical_repeated_run(actual):
    completed, _ = actual
    again = run_cli(DATA_DIR)
    assert again.returncode == 0
    assert again.stdout.encode("utf-8") == completed.stdout.encode("utf-8")
    assert again.stderr == ""


def test_alternative_path_outside_project_and_changed_data(copied_data, tmp_path, actual):
    # Реально меняем новый вход, чтобы исключить чтение штатной папки или готового ответа.
    (copied_data / "02_bank_statement_rub.csv").write_text(
        "Служебная строка;;;;;;\nДата операции;Время;Сумма;Валюта;Тип;Контрагент;Назначение платежа\n"
        "01.01.2025;12:00;0,01;RUR;Поступление;Тест;Пополнение\n", encoding="cp1251")
    result = run_cli(copied_data, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout, parse_float=Decimal)
    assert data["q2b_unidentified_payments_count"] == 1
    assert data["q2b_unidentified_payments_sum_rub"] == Decimal("0.01")
    assert data["q2a_paid_orders_without_payment"] == sorted(paid_order_ids(load_clean_crm(copied_data / runner.FILES[0])))
    for key in ["q1_unique_orders", "q1_unique_order_lines", "q3_price_violation_lines_total", "q4_chats_meeting_before_price"]:
        assert data[key] == actual[1][key]
    assert "другая папка данных" in data["run_command"]


def test_missing_directory(tmp_path):
    result = run_cli(tmp_path / "нет папки")
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Папка исходных данных" in result.stderr
    assert "Traceback" not in result.stderr


def test_file_instead_of_directory(tmp_path):
    path = tmp_path / "file"
    path.write_text("не каталог", encoding="utf-8")
    result = run_cli(path)
    assert result.returncode != 0 and result.stdout == ""


def test_missing_source_file(copied_data):
    (copied_data / "04_client_chats.zip").unlink()
    result = run_cli(copied_data)
    assert result.returncode != 0 and result.stdout == ""
    assert "04_client_chats.zip" in result.stderr


def test_invalid_template(copied_data):
    (copied_data / "answers_template.json").write_text("{}", encoding="utf-8")
    result = run_cli(copied_data)
    assert result.returncode != 0 and result.stdout == ""
    assert "схеме" in result.stderr


def test_corrupt_archive(copied_data):
    (copied_data / "04_client_chats.zip").write_bytes(b"invalid zip")
    result = run_cli(copied_data)
    assert result.returncode != 0 and result.stdout == ""
    assert "Ошибка расчёта" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("key,value", [
    ("q1_unique_orders", True), ("q1_unique_orders", -1),
    ("q2a_paid_orders_without_payment", ["75"]),
    ("q2a_paid_orders_without_payment", [2, 1]),
    ("q2a_paid_orders_without_payment", [1, 1]),
    ("q2b_unidentified_payments_sum_rub", "0.01"),
    ("q2b_unidentified_payments_sum_rub", Decimal("0.001")),
    ("q4_chats_meeting_before_price", [1]),
    ("q4_chats_meeting_before_price", ["Z", "A"]),
    ("q3_price_violation_lines_by_manager", {"имя": -1}),
    ("notes", ""),
])
def test_schema_validation(actual, key, value):
    data = deepcopy(actual[1])
    data[key] = value
    with pytest.raises(ValueError):
        runner.validate_answers(data, load_answers_template())


def test_reuse_of_single_cleaned_crm(monkeypatch):
    calls, received = [], []
    clean, bank, pricing = runner.clean_crm, runner.reconcile_bank, runner.find_price_violations

    def wrapped_clean(raw):
        result = clean(raw)
        calls.append(result)
        return result

    def wrapped_bank(crm, source):
        received.append(crm)
        return bank(crm, source)

    def wrapped_pricing(crm, source):
        received.append(crm)
        return pricing(crm, source)

    monkeypatch.setattr(runner, "clean_crm", wrapped_clean)
    monkeypatch.setattr(runner, "reconcile_bank", wrapped_bank)
    monkeypatch.setattr(runner, "find_price_violations", wrapped_pricing)
    runner.solve(DATA_DIR)
    assert len(calls) == 1
    assert len(received) == 2 and all(frame is calls[0] for frame in received)


def test_incomplete_q3_stops_json_output(monkeypatch, capsys):
    report = SimpleNamespace(diagnostics=pd.DataFrame({"issue": ["missing_price"]}))
    monkeypatch.setattr(runner, "find_price_violations", lambda crm, sheets: report)
    assert runner.main([str(DATA_DIR)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Q3 требует проверки" in output.err and "missing_price" in output.err


def test_q4_warning_goes_only_to_stderr(monkeypatch, capsys):
    flagged = ChatAnalysis("TEST", None, None, False, "Требует проверки", ({"position": 1},))
    monkeypatch.setattr(runner, "analyze_chats", lambda chats: [flagged])
    assert runner.main([str(DATA_DIR)]) == 0
    output = capsys.readouterr()
    data = json.loads(output.out, parse_float=Decimal)
    assert data["q4_chats_meeting_before_price"] == []
    assert "Предупреждение: Q4" in output.err
    assert "Предупреждение" not in output.out


def test_module_consistency_and_original_hashes(actual):
    def hashes():
        return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in runner.FILES}
    before = hashes()
    answers, warnings = runner.solve(DATA_DIR)
    crm = load_clean_crm()
    assert {key: answers[key] for key in summarize_q1(crm)} == summarize_q1(crm)
    bank = reconcile_bank(crm, load_bank())
    assert answers["q2a_paid_orders_without_payment"] == list(bank.paid_orders_without_payment)
    assert answers["q2b_unidentified_payments_count"] == bank.unidentified_count
    assert answers["q2b_unidentified_payments_sum_rub"] == bank.unidentified_sum_rub
    prices = find_price_violations(crm, load_catalog()).answers()
    assert {key: answers[key] for key in prices} == prices
    assert answers["q4_chats_meeting_before_price"] == violation_ids(analyze_chats(load_chats()))
    assert warnings == ()
    assert answers == actual[1]
    assert hashes() == before
