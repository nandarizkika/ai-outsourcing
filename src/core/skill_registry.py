from typing import TypedDict
from src.core.models import ClientConfig, SkillModule, Tier


class SkillDefinition(TypedDict):
    name: str
    description: str
    tier: Tier
    agent: str | None
    requires: list[str]


class SkillRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, SkillDefinition] = {}

    def register(self, skill_id: str, defn: SkillDefinition) -> None:
        self._registry[skill_id] = defn

    def get(self, skill_id: str) -> SkillDefinition | None:
        return self._registry.get(skill_id)

    def all(self) -> dict[str, SkillDefinition]:
        return dict(self._registry)

    def is_enabled(self, skill_id: str, config: ClientConfig) -> bool:
        return skill_id in config.enabled_skills


def _build_default_registry() -> SkillRegistry:
    r = SkillRegistry()
    _B, _A, _E = Tier.BASIC, Tier.ADVANCED, Tier.ENTERPRISE

    r.register(SkillModule.SQL_QUERYING, SkillDefinition(name="SQL Querying", tier=_B, agent="SQLAgent", requires=[], description="Natural language → SQL → structured results"))
    r.register(SkillModule.DATA_VISUALIZATION, SkillDefinition(name="Data Visualization", tier=_B, agent="ChartAgent", requires=[SkillModule.SQL_QUERYING], description="Generate charts from query results"))
    r.register(SkillModule.REPORT_GENERATION, SkillDefinition(name="Report Generation", tier=_B, agent="Orchestrator", requires=[], description="Written narrative analysis from data"))
    r.register(SkillModule.SCHEDULED_REPORTING, SkillDefinition(name="Scheduled Reporting", tier=_B, agent="JobScheduler", requires=[], description="Cron-based automated report delivery"))
    r.register(SkillModule.HARD_RULE_ANOMALY, SkillDefinition(name="Hard-Rule Anomaly", tier=_A, agent="AnomalyAgent", requires=[SkillModule.SQL_QUERYING], description="Flag when metrics breach client-defined rules"))
    r.register(SkillModule.STATISTICAL_ANOMALY, SkillDefinition(name="Statistical Anomaly", tier=_A, agent="AnomalyAgent", requires=[SkillModule.SQL_QUERYING], description="Pattern-based deviation from historical baseline"))
    r.register(SkillModule.FUNNEL_ANALYSIS, SkillDefinition(name="Funnel Analysis", tier=_A, agent="SQLAgent", requires=[SkillModule.SQL_QUERYING], description="Conversion rates and drop-off counts per stage"))
    r.register(SkillModule.COHORT_ANALYSIS, SkillDefinition(name="Cohort Analysis", tier=_A, agent="SQLAgent", requires=[SkillModule.SQL_QUERYING], description="Retention rates grouped by acquisition period"))
    r.register(SkillModule.SPREADSHEET_ANALYSIS, SkillDefinition(name="Spreadsheet Analysis", tier=_A, agent="SpreadsheetAgent", requires=[], description="Upload Excel/CSV and analyse in-place without SQL"))
    r.register(SkillModule.MACHINE_LEARNING, SkillDefinition(name="Machine Learning", tier=_E, agent="MLAgent", requires=[SkillModule.SQL_QUERYING], description="Forecast, regression, classification + model tuning"))
    r.register(SkillModule.PRESENTATION_BUILDING, SkillDefinition(name="Presentation Building", tier=_E, agent="DeckAgent", requires=[SkillModule.REPORT_GENERATION], description="Auto-generate PPTX slide decks with storyline"))
    r.register(SkillModule.NLP_MODELING, SkillDefinition(name="NLP Modeling", tier=_E, agent="MLAgent", requires=[], description="Text classification, NER, similarity via TF-IDF + embeddings"))
    r.register(SkillModule.HYPOTHESIS_TESTING, SkillDefinition(name="Hypothesis Testing", tier=_E, agent="AnalystAgent", requires=[SkillModule.SQL_QUERYING], description="Statistical significance tests"))
    r.register(SkillModule.SEGMENTATION, SkillDefinition(name="Segmentation", tier=_E, agent="AnalystAgent", requires=[SkillModule.SQL_QUERYING], description="KMeans clustering to find segments"))
    r.register(SkillModule.AB_TESTING, SkillDefinition(name="A/B Testing", tier=_E, agent="AnalystAgent", requires=[SkillModule.SQL_QUERYING], description="Statistical comparison of experiment variants"))
    r.register(SkillModule.DEEP_ANALYSIS, SkillDefinition(name="Deep Analysis", tier=_E, agent="AnalystAgent", requires=[SkillModule.SQL_QUERYING, SkillModule.MACHINE_LEARNING], description="ReAct loop: root cause → hypothesize → test → prototype"))
    return r


skill_registry = _build_default_registry()
