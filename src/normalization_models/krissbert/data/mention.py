from dataclasses import dataclass

@dataclass
class Mention:
    cui: str
    start: int
    end: int
    text: str