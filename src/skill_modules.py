from src.core.models import SkillModule, Tier

_BASIC = [
    SkillModule.SQL_QUERYING,
    SkillModule.DATA_VISUALIZATION,
    SkillModule.REPORT_GENERATION,
    SkillModule.SCHEDULED_REPORTING,
    SkillModule.HARD_RULE_ANOMALY,
]

_ADVANCED = _BASIC + [
    SkillModule.SPREADSHEET_ANALYSIS,
    SkillModule.LOOKER_INTEGRATION,
    SkillModule.STATISTICAL_ANOMALY,
]

_ENTERPRISE = _ADVANCED + [
    SkillModule.MACHINE_LEARNING,
    SkillModule.PRESENTATION_BUILDING,
]

_TIER_MAP = {
    Tier.BASIC: _BASIC,
    Tier.ADVANCED: _ADVANCED,
    Tier.ENTERPRISE: _ENTERPRISE,
}


class SkillModuleRegistry:
    @staticmethod
    def defaults_for_tier(tier: Tier) -> list[SkillModule]:
        return list(_TIER_MAP[tier])

    @staticmethod
    def is_enabled(skill: SkillModule, enabled: list[SkillModule]) -> bool:
        return skill in enabled
