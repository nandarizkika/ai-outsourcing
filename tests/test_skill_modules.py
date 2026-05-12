from src.skill_modules import SkillModuleRegistry
from src.core.models import SkillModule, Tier


def test_basic_tier_has_core_skills():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModule.SQL_QUERYING in skills
    assert SkillModule.DATA_VISUALIZATION in skills
    assert SkillModule.REPORT_GENERATION in skills
    assert SkillModule.SCHEDULED_REPORTING in skills
    assert SkillModule.HARD_RULE_ANOMALY in skills


def test_basic_tier_excludes_advanced_skills():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModule.MACHINE_LEARNING not in skills
    assert SkillModule.PRESENTATION_BUILDING not in skills
    assert SkillModule.STATISTICAL_ANOMALY not in skills


def test_advanced_tier_includes_sheets_and_looker():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.ADVANCED)
    assert SkillModule.SPREADSHEET_ANALYSIS in skills
    assert SkillModule.LOOKER_INTEGRATION in skills
    assert SkillModule.STATISTICAL_ANOMALY in skills


def test_enterprise_tier_includes_ml_and_deck():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.ENTERPRISE)
    assert SkillModule.MACHINE_LEARNING in skills
    assert SkillModule.PRESENTATION_BUILDING in skills


def test_is_enabled():
    skills = SkillModuleRegistry.defaults_for_tier(Tier.BASIC)
    assert SkillModuleRegistry.is_enabled(SkillModule.SQL_QUERYING, skills) is True
    assert SkillModuleRegistry.is_enabled(SkillModule.MACHINE_LEARNING, skills) is False
