"""Смысловые контрпримеры и интеграционные проверки Q4."""
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json

import pytest

from src.chats import analyze_chat, analyze_chats, classify_message, violation_ids, write_diagnostics
from src.loaders import Chat, DATA_DIR, Message, load_bank, load_catalog, load_chats
from src.crm import load_clean_crm, summarize_q1
from src.bank import reconcile_bank
from src.pricing import find_price_violations


def message(text, author="Менеджер"):
    return Message(datetime(2025, 1, 1), author, text)


def chat(*messages, identifier="TEST"):
    return Chat(identifier, "client", "A1", "manager", tuple(messages), "test.txt", "")


@pytest.mark.parametrize("text", [
    "Цена 250000", "Стоимость этой модели — 250 000 ₽.", "250000 ₽",
    "250 тыс.", "250к", "Цена 250K", "Шуба стоит 250000 рублей",
    "Цена — двести пятьдесят тысяч рублей.", "Стоимость — два миллиона двести тысяч рублей.",
    "От 300 тысяч", "Около 400 тысяч", "Стоимость изделия — 250 000 рублей.",
    "Стоимость изделия — 250 000 рублей.",
    "Стоимость изделия — 250 000 рублей.",
])
def test_concrete_price_formats(text):
    assert classify_message(message(text)).is_price


@pytest.mark.parametrize("text", [
    "Размер 46-48, рост 168", "Завтра в 15:00 удобно?", "Телефон +7 999 250-00-00",
    "Дата 25.01.2025", "Срок 250 дней", "Можем оплатить в 2-3 платежа.",
    "Возможна предоплата 30%, остальное при получении.", "Доставка 500 рублей",
    "Стоимость примерки 500 рублей", "Ваш бюджет 250 тысяч?",
    "Вы говорили о 250000 рублях?", "Цену пока не могу назвать",
    "На счёте 250000 рублей", "Шоурум открыт с 10 до 21",
])
def test_not_a_product_price(text):
    assert not classify_message(message(text)).is_price


@pytest.mark.parametrize("text", [
    "Шуба стоит 250 000 рублей с доставкой",
    "Стоимость жилета с доставкой — двести пятьдесят тысяч рублей",
    "Цена 250 000", "Цена 250\u00a0000", "Стоимость модели 1 250 000",
])
def test_audit_product_price_with_delivery_or_grouped_amount(text):
    result = classify_message(message(text))
    assert result.is_price
    assert not result.is_invitation


@pytest.mark.parametrize("text", [
    "Доставка стоит 2 000 рублей", "Стоимость доставки 2 000",
    "Предоплата за шубу 30 000 рублей", "Стоимость примерки 1 000 рублей",
])
def test_audit_separate_service_is_not_product_price(text):
    assert not classify_message(message(text)).is_price


@pytest.mark.parametrize("text", [
    "Не жду вас завтра, визит пока не согласован",
    "Мы не ждём вас сегодня", "Не будем ждать вас завтра",
    "Не приглашаю вас на примерку", "Можете не приезжать завтра",
])
def test_audit_negation_applies_to_invitation(text):
    assert not classify_message(message(text)).is_invitation


@pytest.mark.parametrize("text", [
    "Не переживайте и приезжайте к нам завтра",
    "Не жду подтверждения и приглашаю вас на примерку",
    "Не жду вас завтра, но жду вас в пятницу",
    "Не нужно торопиться с выбором, приезжайте на примерку",
])
def test_audit_unrelated_negation_does_not_cancel_invitation(text):
    assert classify_message(message(text)).is_invitation


@pytest.mark.parametrize("price", [
    "Шуба стоит 250 000 рублей с доставкой", "Цена 250 000",
    "Стоимость с доставкой — двести пятьдесят тысяч рублей",
])
@pytest.mark.parametrize("invite_first", [False, True])
def test_audit_full_conversation_event_order(price, invite_first):
    events = [message(price), message("Когда сможете подъехать?")]
    if invite_first:
        events.reverse()
    result = analyze_chat(chat(message("Сколько стоит жилет?", "Клиент"),
                               events[0], message("Понятно", "Клиент"), events[1]))
    assert result.violation is invite_first
    assert result.first_price.position == (4 if invite_first else 2)
    assert result.first_invitation.position == (2 if invite_first else 4)


def test_audit_negated_invitation_before_price_in_full_conversation():
    result = analyze_chat(chat(
        message("Не жду вас завтра, визит пока не согласован"),
        message("А сколько стоит жилет?", "Клиент"),
        message("Цена 250 000"), message("Приходите к нам в пятницу")))
    assert not result.violation
    assert result.first_price.position == 3
    assert result.first_invitation.position == 4


def test_audit_customer_price_does_not_hide_early_invitation():
    result = analyze_chat(chat(message("Цена 250 000", "Клиент"),
                               message("Приезжайте к нам"),
                               message("Шуба стоит 250 000 рублей с доставкой")))
    assert result.violation
    assert result.first_price.position == 3


@pytest.mark.parametrize("text", [
    "Приглашаю на примерку в ателье — когда вам удобно?",
    "Когда сможете подъехать на примерку?",
    "Давайте я запишу вас на встречу в шоурум. Завтра в 15:00 удобно?",
    "Лучше приезжайте к нам, всё покажем вживую!",
    "Заглядывайте к нам завтра", "Давайте встретимся завтра",
    "Жду вас завтра в салоне", "К нам завтра к трём удобно?", "Давайте к нам завтра!",
])
def test_visit_intent_without_required_keyword(text):
    assert classify_message(message(text)).is_invitation


@pytest.mark.parametrize("text", [
    "Наш шоурум находится у метро, работаем ежедневно с 10 до 21.",
    "Все изделия шьём в нашем ателье, мерки снимаем индивидуально.",
    "Примерка может понадобиться позже", "Не приезжайте завтра",
    "Пока не нужно приезжать", "Визит отменён",
])
def test_location_description_is_not_invitation(text):
    assert not classify_message(message(text)).is_invitation


@pytest.mark.parametrize("messages,expected", [
    ([message("Цена 250000"), message("Приезжайте к нам")], False),
    ([message("Приезжайте к нам"), message("Цена 250000")], True),
    ([message("Приезжайте к нам")], True),
    ([message("Цена 250000")], False),
    ([message("Цена 250000", "Клиент"), message("Приезжайте к нам")], True),
    ([message("Приезжайте к нам", "Клиент"), message("Цена 250000")], False),
    ([message("Приезжайте к нам"), message("Цена 250000"), message("Приходите завтра")], True),
    ([message("Цена 250000"), message("Приезжайте к нам"), message("Цена 240000")], False),
    ([], False),
])
def test_event_order(messages, expected):
    assert analyze_chat(chat(*messages)).violation is expected


def test_same_message_is_flagged_without_strict_precedence():
    result = analyze_chat(chat(message("Цена 250000. Приезжайте к нам!")))
    assert not result.violation
    assert result.first_price.position == result.first_invitation.position == 1
    assert result.review_flags


def test_original_sequence_overrides_clock_sort():
    first = message("Приезжайте к нам")
    second = replace(message("Цена 250000"), timestamp=first.timestamp - timedelta(days=1))
    result = analyze_chat(chat(first, second))
    assert result.violation
    assert result.first_invitation.position == 1
    assert result.first_price.position == 2


def test_short_reply_to_price_question():
    result = classify_message(message("250000"), message("Сколько стоит?", "Клиент"))
    assert result.is_price
    assert not classify_message(message("250000")).is_price


def test_unknown_semantics_and_roles_flagged():
    assert classify_message(message("Цену обсудим при личном знакомстве")).review_reasons
    assert classify_message(message("Цена 250000", "Система")).review_reasons


def test_no_id_dependency_and_duplicate_ids_rejected():
    original = chat(message("Приходите завтра"))
    assert analyze_chat(replace(original, chat_id="NEW")).violation == analyze_chat(original).violation
    with pytest.raises(ValueError, match="Повторяющиеся"):
        analyze_chats([original, original])


def test_diagnostics_are_reproducible_and_contain_evidence(tmp_path):
    results = analyze_chats([chat(message("Приезжайте к нам"), message("Цена 250000"))])
    path = tmp_path / "q4.json"
    write_diagnostics(results, path)
    before = path.read_bytes()
    write_diagnostics(results, path)
    assert path.read_bytes() == before
    data = json.loads(before)
    evidence = data["violations"][0]
    assert evidence["first_invitation"]["position"] == 1
    assert evidence["first_price"]["position"] == 2
    assert evidence["first_invitation"]["text"] == "Приезжайте к нам"
    assert evidence["reason"]
    assert data["q4_chats_meeting_before_price"] == ["TEST"]


def test_missing_price_after_invitation_and_renamed_copy():
    original = chat(message("Приезжайте в салон"))
    result = analyze_chat(original)
    assert result.first_invitation is not None
    assert result.first_price is None
    assert result.violation
    assert analyze_chat(replace(original, chat_id="ARBITRARY")).violation


def test_real_archive_invariants_source_integrity_and_regressions():
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in DATA_DIR.iterdir() if p.is_file()}
    before = hashes()
    rows, sheets = load_clean_crm(), load_catalog()
    q1, q2, q3 = summarize_q1(rows), reconcile_bank(rows, load_bank()), find_price_violations(rows, sheets).answers()
    chats = load_chats()
    results = analyze_chats(chats)
    assert len(results) == len(chats) == 150
    assert results == analyze_chats(load_chats())
    assert violation_ids(results) == sorted(set(violation_ids(results)))
    originals = {c.chat_id: c for c in chats}
    for result in results:
        invitation, price = result.first_invitation, result.first_price
        assert result.violation == (invitation is not None and (price is None or invitation.position < price.position))
        for event in [invitation, price]:
            if event:
                source = originals[result.chat_id].messages[event.position - 1]
                assert source.author == "Менеджер"
                assert source.text == event.text
        # Перестановка только ID не меняет смыслового решения и позиций.
        renamed = analyze_chat(replace(originals[result.chat_id], chat_id="RENAMED"))
        assert (renamed.violation, renamed.first_price, renamed.first_invitation) == (result.violation, price, invitation)
    assert summarize_q1(load_clean_crm()) == q1
    after_q2 = reconcile_bank(load_clean_crm(), load_bank())
    assert (after_q2.unidentified_count, after_q2.unidentified_sum_rub, after_q2.paid_orders_without_payment) == (q2.unidentified_count, q2.unidentified_sum_rub, q2.paid_orders_without_payment)
    assert find_price_violations(load_clean_crm(), load_catalog()).answers() == q3
    assert hashes() == before
