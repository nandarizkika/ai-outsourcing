import pytest
from src.core.skill_registry import SkillRegistry, SkillDefinition, skill_registry
from src.core.models import SkillModule, Tier, ClientConfig, Channel


def test_built_in_skills_are_pre_registered():
    assert skill_registry.get(SkillModule.SQL_QUERYING) is not None
    assert skill_registry.get(SkillModule.MACHINE_LEARNING) is not None
    assert skill_registry.get(SkillModule.DEEP_ANALYSIS) is not None
    assert skill_registry.get(SkillModule.NLP_MODELING) is not None


def test_get_returns_none_for_unknown_skill():
    assert skill_registry.get("not_a_real_skill_xyz") is None


def test_register_custom_skill():
    registry = SkillRegistry()
    registry.register("custom_skill_abc", SkillDefinition(
        name="Custom Skill",
        description="A custom skill for testing",
        tier=Tier.BASIC,
        agent=None,
        requires=[],
    ))
    defn = registry.get("custom_skill_abc")
    assert defn is not None
    assert defn["name"] == "Custom Skill"


def test_all_returns_all_registered_skills():
    registry = SkillRegistry()
    registry.register("skill_a", SkillDefinition(name="A", description="a", tier=Tier.BASIC, agent=None, requires=[]))
    registry.register("skill_b", SkillDefinition(name="B", description="b", tier=Tier.BASIC, agent=None, requires=[]))
    assert "skill_a" in registry.all()
    assert "skill_b" in registry.all()


def test_is_enabled_checks_client_config():
    config = ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=[SkillModule.SQL_QUERYING, SkillModule.DEEP_ANALYSIS],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )
    assert skill_registry.is_enabled(SkillModule.SQL_QUERYING, config) is True
    assert skill_registry.is_enabled(SkillModule.MACHINE_LEARNING, config) is False


def test_new_skillmodule_entries_exist():
    assert SkillModule.NLP_MODELING == "nlp_modeling"
    assert SkillModule.HYPOTHESIS_TESTING == "hypothesis_testing"
    assert SkillModule.SEGMENTATION == "segmentation"
    assert SkillModule.AB_TESTING == "ab_testing"
    assert SkillModule.DEEP_ANALYSIS == "deep_analysis"


def test_client_config_enabled_skills_accepts_strings():
    config = ClientConfig(
        client_id="c1", name="Test", tier=Tier.ENTERPRISE,
        enabled_skills=["sql_querying", "custom_skill_xyz"],
        account_mode="vendor", active_channels=[Channel.SLACK],
    )
    assert "sql_querying" in config.enabled_skills
    assert "custom_skill_xyz" in config.enabled_skills
