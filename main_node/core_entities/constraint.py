from dataclasses import dataclass
import re
from typing import Protocol


class Constraint(Protocol):
    def is_satisfied(self, parameters: dict[str, str]) -> bool:
        pass

@dataclass(frozen=True)
class Variable:
    param_name: str
    cat_name: str
    def is_satisfied(self, parameters: dict[str, str]) -> bool:
        actual_cat_path = parameters.get(self.param_name)
        return actual_cat_path is not None and actual_cat_path.endswith(self.cat_name)

@dataclass(frozen=True)
class Not:
    operand: Constraint
    def is_satisfied(self, parameters: dict[str, str]) -> bool:
        return not self.operand.is_satisfied(parameters)

@dataclass(frozen=True)
class And:
    operands: list[Constraint]
    def is_satisfied(self, parameters: dict[str, str]) -> bool:
        return all(op.is_satisfied(parameters) for op in self.operands)

@dataclass(frozen=True)
class Or:
    operands: list[Constraint]
    def is_satisfied(self, parameters: dict[str, str]) -> bool:
        return any(op.is_satisfied(parameters) for op in self.operands)

class ConstraintParser:
    def __init__(self, text: str):
        self.tokens = self.tokenize(text)
        self.pos = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.replace(" and ", " && ").replace(" or ", " || ") #.replace(" not ", "!")
        text = re.sub(r"(\s|\(|^)not(\s|\()", r"\1!\2", text)
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue

            if text[i] in "()!":
                tokens.append(text[i])
                i += 1
                continue

            if text.startswith("&&", i):
                tokens.append("&&")
                i += 2
                continue

            if text.startswith("||", i):
                tokens.append("||")
                i += 2
                continue

            # Variable
            start = i
            while (
                i < len(text)
                and not text[i].isspace()
                and text[i] not in "()!"
                and not text.startswith("&&", i)
                and not text.startswith("||", i)
            ):
                i += 1
            tokens.append(text[start:i])

        return tokens

    def current(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, token: str | None = None) -> str:
        current = self.current()
        if current is None:
            raise ValueError("Unexpected end of expression")
        if token is not None and current != token:
            raise ValueError(f"Expected {token!r}, got {current!r}")
        self.pos += 1
        return current

    def parse(self) -> Constraint:
        result = self.parse_or()
        if self.current() is not None:
            raise ValueError(f"Unexpected token: {self.current()!r}")
        return result

    # or := and ("or" and)*
    def parse_or(self) -> Constraint:
        result = [self.parse_and()]
        while self.current() == "||":
            self.consume()
            result.append(self.parse_and())
        return result[0] if len(result) == 1 else Or(result)

    # and := not ("and" not)*
    def parse_and(self) -> Constraint:
        result = [self.parse_not()]
        while self.current() == "&&":
            self.consume()
            result.append(self.parse_not())
        return result[0] if len(result) == 1 else And(result)

    # not := ("not" | "!")* atom
    def parse_not(self) -> Constraint:
        if self.current() == "!":
            self.consume()
            return Not(self.parse_not())
        return self.parse_atom()

    # atom := VARIABLE | "(" or ")"
    def parse_atom(self) -> Constraint:
        token = self.current()

        if token is None:
            raise ValueError("Expected expression")

        if token == "(":
            self.consume("(")
            result = self.parse_or()
            self.consume(")")
            return result

        if token in ("&&", "||", "!", ")"):
            raise ValueError(f"Unexpected token: {token!r}")

        self.consume()
        *_, param, value = token.rsplit(".", 2)
        return Variable(param, value)