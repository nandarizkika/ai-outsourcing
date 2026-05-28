import json
from typing import Dict, List, Optional


class BusinessContextManager:
    """Manages business glossary, metrics, and table relationships."""

    def __init__(self, context_file: str = "business_context.json"):
        try:
            with open(context_file, 'r') as f:
                self.context = json.load(f)
        except FileNotFoundError:
            self.context = {
                "business_glossary": {},
                "table_relationships": [],
                "table_join_patterns": {}
            }

    def get_glossary_context(self) -> str:
        """Format business glossary for LLM prompt."""
        glossary = self.context.get("business_glossary", {})
        if not glossary:
            return ""

        lines = ["## Business Glossary\n"]
        for term, details in glossary.items():
            lines.append(f"- **{term}** ({details.get('name', term)}): {details.get('definition', '')}")
            if details.get('calculation'):
                lines.append(f"  - Calculation: {details['calculation']}")
            if details.get('filters'):
                lines.append(f"  - Filters: {', '.join(details['filters'])}")
            lines.append("")

        return "\n".join(lines)

    def get_relationship_context(self) -> str:
        """Format table relationships for LLM prompt."""
        relationships = self.context.get("table_relationships", [])
        if not relationships:
            return ""

        lines = ["## Table Relationships\n"]
        for rel in relationships[:15]:  # Top 15 relationships
            t1, t2 = rel['table1'], rel['table2']
            lines.append(f"- {t1} connects to {t2}")

        return "\n".join(lines)

    def get_join_suggestions(self, table: str) -> List[str]:
        """Get tables that commonly join with the given table."""
        patterns = self.context.get("table_join_patterns", {})
        return patterns.get(table, [])

    def get_metric_sql(self, metric: str) -> Optional[Dict]:
        """Get SQL template for a business metric."""
        glossary = self.context.get("business_glossary", {})
        return glossary.get(metric.lower())

    def get_full_context_prompt(self) -> str:
        """Get complete business context as formatted text for LLM."""
        sections = [
            self.get_glossary_context(),
            self.get_relationship_context(),
        ]
        return "\n\n".join(filter(None, sections))
