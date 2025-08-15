# sentenialx/models/orchestrator/versioning.py
from dataclasses import dataclass

@dataclass
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> "SemVer":
        a, b, c = [int(x) for x in s.split(".")]
        return cls(a, b, c)

    def bump(self, strategy: str) -> "SemVer":
        if strategy == "major":
            return SemVer(self.major + 1, 0, 0)
        if strategy == "minor":
            return SemVer(self.major, self.minor + 1, 0)
        return SemVer(self.major, self.minor, self.patch + 1)

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
