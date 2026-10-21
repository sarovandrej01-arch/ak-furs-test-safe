"""Воспроизводимый анализ смысла ценовых предложений и приглашений Q4."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from .loaders import Chat, DATA_DIR, Message, load_chats


NOTES = (
    "События определяются только по сообщениям менеджера; цены клиента не учитываются.",
    "Учитывается первая конкретная цена, включая первоначальную, приблизительную и цену от указанной суммы.",
    "Предоплата, скидка, рассрочка и стоимость доставки сами по себе не являются ценой изделия.",
    "Информация об адресе/режиме работы/пошиве не является приглашением без намерения организовать визит.",
    "Порядок исходных сообщений сохраняется независимо от времени; позиции начинаются с 1.",
    "Цена и приглашение в одном сообщении не считаются строгим предшествованием; случай отмечается для проверки.",
    "Незнакомые смысловые формулировки могут требовать расширения правил: диагностические флаги не заменяют понимание всех возможных текстов.",
)


def _text(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


_BASE_WORD = (
    r"(?:один|одна|одно|дв[ае]|три|четыре|пять|шесть|семь|восемь|девять|десять|"
    r"одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|"
    r"семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|сорок|пятьдесят|"
    r"шестьдесят|семьдесят|восемьдесят|девяносто|сто|двести|триста|четыреста|"
    r"пятьсот|шестьсот|семьсот|восемьсот|девятьсот)"
)
_SCALE = r"(?:тысяч[аи]?|миллион(?:а|ов)?)"
_NUMBER = r"[0-9]+(?:[ .\u00a0][0-9]{3})*(?:[,.][0-9]+)?"
_WORDS = _BASE_WORD + r"(?:\s+(?:" + _BASE_WORD + "|" + _SCALE + r"))*"
_AMOUNT = re.compile(r"(?<!\w)(?:" + _NUMBER + "|" + _WORDS + r")\s*(?:₽|руб(?:лей|ля|ль)?\b|тыс\b|млн\b|[кk]\b|" + _SCALE + r"\b)(?:\s+руб(?:лей|ля|ль)\b)?")
_PRICE_SCOPE = re.compile(r"\b(?:цен[ауы]|стоимост\w*|стоит|обойдется|итоговая|выйдет|это будет|для вас|за изделие|за модель|шуб\w*|полушуб\w*|жилет\w*|шапк\w*|палантин\w*)\b")
_BARE_AMOUNT = re.compile(r"(?<![\w.,:/+\-])(?:[0-9]{1,3}(?: [0-9]{3})+|[0-9]{4,})(?![\w]|[.,:/-][0-9])")
_AUXILIARY = re.compile(r"\b(?:предоплат\w*|аванс\w*|рассроч\w*|платеж\w*|бюджет\w*|скидка|снятие мерок|стоимость примерки|цена примерки)\b")
_DELIVERY = re.compile(r"\bдоставк\w*\b")
_INCLUDED_DELIVERY = re.compile(r"\b(?:(?:с|включая|вместе с) доставк\w*|доставка (?:уже )?включена)\b")
_DELIVERY_PRICE = re.compile(r"\b(?:(?:цена|стоимость) доставки|доставка\s+(?:стоит|обойдется|выйдет))\b")
_QUESTION_OR_QUOTE = re.compile(r"(?:сколько|какой|какая|ваш бюджет|вы (?:говорили|видели|сказали)|у конкурентов|клиент (?:назвал|сказал))")
_PRICE_NEGATION = re.compile(r"(?:не (?:могу|буду|готова?|назову)|не стоит|не [0-9]|цен\w* (?:пока )?не|назов\w* позже)")
_VISIT = re.compile(r"\b(?:приезжайте|приходите|подъезжайте|заезжайте|заглядывайте|заходите)\b")
_PLACE = r"(?:примерк\w*|встреч\w*|ателье|шоурум\w*|салон\w*|к нам|вживую)"
_INVITATION = [
    re.compile(r"\bприглаша(?:ю|ем)\b.{0,80}" + _PLACE),
    re.compile(r"(?:когда|сможете|можете|давайте|удобно).{0,70}(?:подъехать|приехать|заехать|зайти|встретиться|примерить)"),
    re.compile(r"\b(?:запишу|запишем|записать|записываю|забронирую)\b.{0,80}" + _PLACE),
    re.compile(r"\b(?:давайте|предлагаю)\b.{0,40}(?:встретимся|увидимся|подберем вживую|примерим)"),
    re.compile(r"\b(?:ждем вас|жду вас|(?:буду|будем) ждать вас)\b"),
    re.compile(r"\bдавайте к нам\b"),
    re.compile(r"к нам.{0,30}(?:завтра|сегодня|в субботу|в воскресенье).{0,30}(?:удобно|сможете|\?)"),
]
_INVITE_NEGATION = re.compile(r"(?:не (?:нужно|надо|стоит|приезжайте|приходите|приглашаю|приглашаем|жду|ждем|будем ждать|буду ждать)|визит отменен)")
_NEGATIVE_PREFIX = re.compile(r"\bне\s+(?:(?:нужно|надо|стоит|могу|можем|буду|будем)\s+)?$")
_NEGATIVE_VISIT_ACTION = re.compile(r"\bне\s+(?:приехать|приезжать|подъехать|заехать|зайти|встретиться|примерить)\b")
_HINT = re.compile(r"(?:цен\w*|стоимост\w*|приглаш\w*|примерк\w*|встрет\w*|приех\w*|подъех\w*|шоурум)")
_NEUTRAL = re.compile(r"(?:шоурум находится|работаем ежедневно|изделия шьем|мерки снимаем индивидуально)")


def _auxiliary_cost(clause: str) -> bool:
    """Отличить отдельную услугу от доставки, включённой в цену изделия."""
    if _AUXILIARY.search(clause) or _DELIVERY_PRICE.search(clause):
        return True
    return bool(_DELIVERY.search(clause) and not _INCLUDED_DELIVERY.search(clause))


def _has_invitation(clause: str) -> bool:
    """Проверить отрицание у конкретного действия, сохранив другие приглашения.

    В «не жду подтверждения и приглашаю» отрицание не относится к приглашению;
    в «не жду вас» оно стоит непосредственно перед распознанным действием.
    """
    for rule in [_VISIT, *_INVITATION]:
        for match in rule.finditer(clause):
            if _NEGATIVE_PREFIX.search(clause[:match.start()]):
                continue
            if _NEGATIVE_VISIT_ACTION.search(match[0]):
                continue
            return True
    return False


@dataclass(frozen=True)
class Classification:
    """В одном сообщении могут одновременно быть оба события."""
    is_price: bool
    is_invitation: bool
    reasons: tuple[str, ...]
    review_reasons: tuple[str, ...]

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(name for flag, name in [(self.is_price, "цена"),
                     (self.is_invitation, "приглашение")] if flag) or ("прочее",)


def classify_message(message: Message, previous: Message | None = None) -> Classification:
    """Применить сочетания суммы, предмета, намерения и отрицаний, а не одно слово.

    Короткий числовой ответ допускается в контексте вопроса клиента о цене.
    Сомнительные реплики возвращаются с флагом для локальной проверки.
    """
    role = _text(message.author)
    if role in {"клиент", "покупатель"}:
        return Classification(False, False, ("Реплика клиента не создаёт события менеджера",), ())
    if role not in {"менеджер", "продавец", "консультант"}:
        return Classification(False, False, (), ("Неизвестная роль автора",))
    text = _text(message.text)
    prior_price_question = previous is not None and _text(previous.author) == "клиент" and bool(
        re.search(r"сколько|цен[ау]|стоимост", _text(previous.text)))
    price, invitation = False, False
    reasons, reviews = [], []
    # Не разрезаем разделители тысяч и десятичные знаки внутри суммы.
    clauses = re.split(r"[!;\n]+|(?<=[.?])\s+|,\s+(?=[а-я])", text)
    for clause in clauses:
        has_amount = bool(_AMOUNT.search(clause))
        standalone_amount = bool(re.fullmatch(r"\s*(?:(?:от|около|примерно|порядка|ориентировочно)\s+)?" + _AMOUNT.pattern + r"[ .,!]*", clause))
        amount_is_offer = has_amount and bool(_PRICE_SCOPE.search(clause) or prior_price_question or standalone_amount)
        bare_price = bool(_BARE_AMOUNT.search(clause)) and bool(_PRICE_SCOPE.search(clause) or prior_price_question)
        auxiliary = _auxiliary_cost(clause)
        question = bool(_QUESTION_OR_QUOTE.search(clause))
        negated_price = bool(_PRICE_NEGATION.search(clause))
        if (amount_is_offer or bare_price) and not (auxiliary or question or negated_price):
            price = True
            reasons.append("Конкретная сумма изделия: денежное выражение или ответ о цене")
        elif has_amount and (question or negated_price):
            reviews.append("Сумма в вопросе, цитате или отрицании не принята за объявленную цену")
        elif has_amount and not auxiliary and not amount_is_offer:
            reviews.append("Денежная сумма без ясного утверждения цены изделия")
        if _has_invitation(clause):
            invitation = True
            reasons.append("Предложен физический визит или согласование записи, а не описание места")
    if price and invitation:
        reviews.append("Цена и приглашение в одном сообщении: строгий порядок сообщений не установлен")
    if not price and not invitation and _HINT.search(text) and not (
        _NEUTRAL.search(text) or _INVITE_NEGATION.search(text) or _auxiliary_cost(text)
    ):
        reviews.append("Есть смысловой сигнал цены/визита, но нет уверенно распознанного события")
    return Classification(price, invitation, tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(reviews)))


@dataclass(frozen=True)
class Event:
    position: int
    text: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ChatAnalysis:
    chat_id: str
    first_invitation: Event | None
    first_price: Event | None
    violation: bool
    reason: str
    review_flags: tuple[dict, ...]


def analyze_chat(chat: Chat) -> ChatAnalysis:
    """Выбрать первые события по порядку источника, не по сортировке времени."""
    invitation, price = None, None
    flags = []
    for position, message in enumerate(chat.messages, 1):
        result = classify_message(message, chat.messages[position - 2] if position > 1 else None)
        if result.is_invitation and invitation is None:
            invitation = Event(position, message.text, result.reasons)
        if result.is_price and price is None:
            price = Event(position, message.text, result.reasons)
        if result.review_reasons:
            flags.append({"position": position, "text": message.text, "reasons": result.review_reasons})
    violation = invitation is not None and (price is None or invitation.position < price.position)
    reason = ("Приглашения менеджера нет" if invitation is None else
              "Приглашение есть, цена менеджером не названа" if price is None else
              "Приглашение раньше первой цены" if violation else
              "Цена и приглашение в одном сообщении" if invitation.position == price.position else
              "Первая цена названа до приглашения")
    return ChatAnalysis(chat.chat_id, invitation, price, violation, reason, tuple(flags))


def analyze_chats(chats: Iterable[Chat]) -> list[ChatAnalysis]:
    """Вернуть решения по всем чатам; не допускать неоднозначных повторов ID."""
    analyses = [analyze_chat(chat) for chat in chats]
    if len({result.chat_id for result in analyses}) != len(analyses):
        raise ValueError("Повторяющиеся ID чатов")
    return sorted(analyses, key=lambda result: result.chat_id)


def violation_ids(analyses: Iterable[ChatAnalysis]) -> list[str]:
    return sorted({result.chat_id for result in analyses if result.violation})


def write_diagnostics(analyses: list[ChatAnalysis], path: str | Path) -> None:
    """Сохранить локальные доказательства; стандартный путь исключён из Git."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "q4_chats_meeting_before_price": violation_ids(analyses),
        "processed_count": len(analyses),
        "review_chat_count": sum(bool(result.review_flags) for result in analyses),
        "violations": [asdict(result) for result in analyses if result.violation],
        "needs_review": [asdict(result) for result in analyses if result.review_flags],
        "all_decisions": [asdict(result) for result in analyses],
        "notes": NOTES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q4: приглашение до объявления цены")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "04_client_chats.zip")
    parser.add_argument("--diagnostics", type=Path, default=DATA_DIR / "diagnostics" / "q4_report.json")
    args = parser.parse_args()
    results = analyze_chats(load_chats(args.input))
    write_diagnostics(results, args.diagnostics)
    print(json.dumps({"q4_chats_meeting_before_price": violation_ids(results),
                      "processed_count": len(results), "violation_count": len(violation_ids(results)),
                      "review_chat_count": sum(bool(result.review_flags) for result in results),
                      "diagnostics_path": str(args.diagnostics)}, ensure_ascii=False, indent=2))
