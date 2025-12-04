# Sentenial-X/core/compliance/fine_impact_estimator.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EstimatorConfig:
    """
    Core configuration for fine estimation.
    All amounts are in the same currency.
    """
    # Per-record baseline fine (e.g., data breach per affected record)
    base_per_record: float = 10.0
    # Percentage-of-revenue baseline fine (as fraction, e.g., 0.02 = 2%)
    base_revenue_pct: float = 0.02
    # Statutory/global cap on fines
    statutory_cap: float = 50_000_000.0
    # Minimum floor to avoid trivial fines (optional)
    absolute_floor: float = 10_000.0
    # Daily penalty for ongoing non-compliance (optional)
    daily_penalty: float = 0.0

    # Severity multipliers
    severity_multipliers: Dict[str, float] = None
    # Jurisdiction multipliers (e.g., "EU": 1.5, "US": 1.0)
    jurisdiction_multipliers: Dict[str, float] = None
    # Repeat offense multipliers (0=none, 1=first repeat, etc.)
    repeat_offense_multipliers: Dict[int, float] = None
    # Intent multipliers (e.g., "negligent", "reckless", "wilful")
    intent_multipliers: Dict[str, float] = None
    # Mitigation credits (fraction deducted; e.g., remediation, cooperation)
    mitigation_credits: Dict[str, float] = None

    def __post_init__(self):
        if self.severity_multipliers is None:
            self.severity_multipliers = {
                "low": 0.5,
                "moderate": 1.0,
                "high": 1.5,
                "critical": 2.5,
            }
        if self.jurisdiction_multipliers is None:
            self.jurisdiction_multipliers = {
                "US": 1.0,
                "EU": 1.4,
                "UK": 1.2,
                "APAC": 0.9,
            }
        if self.repeat_offense_multipliers is None:
            self.repeat_offense_multipliers = {
                0: 1.0,
                1: 1.3,
                2: 1.6,
                3: 2.0,
            }
        if self.intent_multipliers is None:
            self.intent_multipliers = {
                "none": 1.0,       # strict liability or unclear intent
                "negligent": 1.2,
                "reckless": 1.6,
                "wilful": 2.2,
            }
        if self.mitigation_credits is None:
            self.mitigation_credits = {
                "remediation": 0.15,   # rapid fix
                "cooperation": 0.10,   # open disclosure, assistance
                "compliance_program": 0.10,  # established controls
                "self_report": 0.10,
            }


@dataclass
class EstimationInput:
    affected_records: int
    annual_revenue: float
    severity: str = "moderate"       # low | moderate | high | critical
    jurisdiction: str = "US"         # US | EU | UK | APAC | custom
    repeat_offenses: int = 0         # 0..n
    intent: str = "none"             # none | negligent | reckless | wilful
    days_non_compliant: int = 0
    mitigation_tags: Optional[Dict[str, float]] = None  # override credits per tag (optional)


@dataclass
class EstimationResult:
    low: float
    mid: float
    high: float
    capped: bool
    breakdown: Dict[str, float]


class FineImpactEstimator:
    """
    Computes a fine range (low/mid/high) and an explainable breakdown.
    Not a legal prediction: use for scenario planning and internal risk assessment.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None):
        self.cfg = config or EstimatorConfig()

    def estimate(self, inp: EstimationInput) -> EstimationResult:
        # 1) Base paths: per-record vs revenue percentage
        base_per_record = self.cfg.base_per_record * max(inp.affected_records, 0)
        base_revenue_pct = self.cfg.base_revenue_pct * max(inp.annual_revenue, 0.0)
        base = max(base_per_record, base_revenue_pct)

        # 2) Multipliers: severity, jurisdiction, repeats, intent
        sev_mult = self._get(self.cfg.severity_multipliers, inp.severity, default=1.0)
        jur_mult = self._get(self.cfg.jurisdiction_multipliers, inp.jurisdiction, default=1.0)
        rep_mult = self._get(self.cfg.repeat_offense_multipliers, inp.repeat_offenses, default=1.0)
        intent_mult = self._get(self.cfg.intent_multipliers, inp.intent, default=1.0)

        # 3) Daily penalty for ongoing non-compliance
        daily_total = self.cfg.daily_penalty * max(inp.days_non_compliant, 0)

        # 4) Apply multipliers to base
        gross = base * sev_mult * jur_mult * rep_mult * intent_mult + daily_total

        # 5) Apply mitigation credits (deductions as fractions)
        mitigation_map = inp.mitigation_tags or self.cfg.mitigation_credits
        total_credit = 0.0
        for tag, credit in mitigation_map.items():
            # Cap per-tag credit to a reasonable bound
            total_credit += max(0.0, min(credit, 0.35))
        total_credit = min(total_credit, 0.6)  # overall cap on deductions
        net = gross * (1.0 - total_credit)

        # 6) Floors and caps
        capped = False
        net = max(net, self.cfg.absolute_floor)
        if net > self.cfg.statutory_cap:
            net = self.cfg.statutory_cap
            capped = True

        # 7) Range: low/mid/high around the net, with severity spread
        spread = self._severity_spread(inp.severity)
        low = max(self.cfg.absolute_floor, net * (1.0 - spread))
        high = min(self.cfg.statutory_cap, net * (1.0 + spread))
        mid = net

        breakdown = {
            "base_per_record": base_per_record,
            "base_revenue_pct": base_revenue_pct,
            "base_selected": base,
            "severity_mult": sev_mult,
            "jurisdiction_mult": jur_mult,
            "repeat_mult": rep_mult,
            "intent_mult": intent_mult,
            "daily_total": daily_total,
            "gross_before_mitigation": gross,
            "total_credit_fraction": total_credit,
            "net_after_mitigation": mid,
            "floor_applied": float(mid == self.cfg.absolute_floor),
            "cap_applied": float(capped),
            "spread_fraction": spread,
        }

        return EstimationResult(
            low=round(low, 2),
            mid=round(mid, 2),
            high=round(high, 2),
            capped=capped,
            breakdown={k: (round(v, 4) if isinstance(v, float) else v) for k, v in breakdown.items()}
        )

    def _get(self, mapping: Dict, key, default: float) -> float:
        return mapping.get(key, default)

    def _severity_spread(self, severity: str) -> float:
        # Wider uncertainty for higher severity scenarios
        return {
            "low": 0.15,
            "moderate": 0.20,
            "high": 0.30,
            "critical": 0.40,
        }.get(severity, 0.20)


# Minimal demo
if __name__ == "__main__":
    estimator = FineImpactEstimator(
        EstimatorConfig(
            base_per_record=20.0,
            base_revenue_pct=0.015,   # 1.5%
            statutory_cap=35_000_000.0,
            absolute_floor=25_000.0,
            daily_penalty=5_000.0,
            jurisdiction_multipliers={"US": 1.0, "EU": 1.5, "UK": 1.2},
        )
    )

    case = EstimationInput(
        affected_records=250_000,
        annual_revenue=300_000_000.0,
        severity="high",
        jurisdiction="EU",
        repeat_offenses=1,
        intent="negligent",
        days_non_compliant=7,
        mitigation_tags={"remediation": 0.15, "cooperation": 0.10}
    )

    result = estimator.estimate(case)
    print("Fine range:", {"low": result.low, "mid": result.mid, "high": result.high, "capped": result.capped})
    print("Breakdown:", result.breakdown)
