"""Demo data for testing and demonstration."""

from app.models.components import ChemicalComponent, MaterialComposition
from app.models.relationships import (
    ComponentRelationship,
    SubstitutionRelationship,
    SynergyRelationship,
)
from app.models.properties import QuantitativeProperty
from app.models.extraction import ExtractionResult


def get_demo_extraction_result() -> ExtractionResult:
    """
    Get sample extraction result for demo mode.

    Includes realistic data for Portland Cement, Fly Ash, and Silica Fume
    with substitution and synergy relationships for material discovery.
    """

    # ==========================================================================
    # Materials with Components
    # ==========================================================================

    portland_cement = MaterialComposition(
        material_name="Portland Cement",
        material_type="cement",
        components=[
            ChemicalComponent(
                name="Calcium Oxide",
                formula="CaO",
                percentage=65.0,
                role="binder",
                category="oxide",
            ),
            ChemicalComponent(
                name="Silicon Dioxide",
                formula="SiO2",
                percentage=22.0,
                role="binder",
                category="oxide",
            ),
            ChemicalComponent(
                name="Aluminum Oxide",
                formula="Al2O3",
                percentage=5.5,
                role="modifier",
                category="oxide",
            ),
            ChemicalComponent(
                name="Iron Oxide",
                formula="Fe2O3",
                percentage=3.5,
                role="flux",
                category="oxide",
            ),
            ChemicalComponent(
                name="Magnesium Oxide",
                formula="MgO",
                percentage=2.0,
                role="minor",
                category="oxide",
            ),
            ChemicalComponent(
                name="Sulfur Trioxide",
                formula="SO3",
                percentage=2.0,
                role="setting_regulator",
                category="oxide",
            ),
        ],
        properties={
            "compressive_strength": "40-60 MPa",
            "setting_time_initial": "45 min",
            "setting_time_final": "375 min",
            "specific_gravity": 3.15,
        },
        source="ASTM C150",
    )

    fly_ash = MaterialComposition(
        material_name="Fly Ash (Class F)",
        material_type="supplementary_cementitious_material",
        components=[
            ChemicalComponent(
                name="Silicon Dioxide",
                formula="SiO2",
                percentage=55.0,
                role="pozzolanic",
                category="oxide",
            ),
            ChemicalComponent(
                name="Aluminum Oxide",
                formula="Al2O3",
                percentage=25.0,
                role="pozzolanic",
                category="oxide",
            ),
            ChemicalComponent(
                name="Iron Oxide",
                formula="Fe2O3",
                percentage=10.0,
                role="inert",
                category="oxide",
            ),
            ChemicalComponent(
                name="Calcium Oxide",
                formula="CaO",
                percentage=5.0,
                role="reactive",
                category="oxide",
            ),
        ],
        properties={
            "fineness": "300-400 m²/kg",
            "loss_on_ignition": "<6%",
            "specific_gravity": 2.3,
        },
        source="ASTM C618",
    )

    silica_fume = MaterialComposition(
        material_name="Silica Fume",
        material_type="supplementary_cementitious_material",
        components=[
            ChemicalComponent(
                name="Silicon Dioxide",
                formula="SiO2",
                percentage=92.0,
                role="pozzolanic",
                category="oxide",
            ),
            ChemicalComponent(
                name="Carbon",
                formula="C",
                percentage=3.0,
                role="impurity",
                category="inorganic",
            ),
            ChemicalComponent(
                name="Iron Oxide",
                formula="Fe2O3",
                percentage=1.5,
                role="minor",
                category="oxide",
            ),
        ],
        properties={
            "specific_surface": "15000-25000 m²/kg",
            "particle_size": "0.1-0.3 µm",
            "specific_gravity": 2.2,
        },
        source="ASTM C1240",
    )

    # ==========================================================================
    # General Relationships
    # ==========================================================================

    relationships = [
        ComponentRelationship(
            source="CaO",
            target="SiO2",
            relationship_type="REACTS_WITH",
            description="Calcium oxide reacts with silica during hydration to form C-S-H gel",
            strength=0.95,
        ),
        ComponentRelationship(
            source="C-S-H",
            target="Compressive Strength",
            relationship_type="AFFECTS",
            description="Calcium silicate hydrate is the primary contributor to concrete strength",
            strength=0.9,
        ),
        ComponentRelationship(
            source="Al2O3",
            target="Setting Time",
            relationship_type="AFFECTS",
            description="Aluminum oxide content influences the setting behavior of cement",
            strength=0.7,
        ),
        ComponentRelationship(
            source="SO3",
            target="Setting Time",
            relationship_type="AFFECTS",
            description="Sulfur trioxide (as gypsum) controls setting time to prevent flash set",
            strength=0.85,
        ),
        ComponentRelationship(
            source="SiO2",
            target="Durability",
            relationship_type="AFFECTS",
            description="Silica from pozzolans improves long-term durability through pozzolanic reaction",
            strength=0.8,
        ),
    ]

    # ==========================================================================
    # Substitution Relationships (NEW - for material discovery)
    # ==========================================================================

    substitutions = [
        SubstitutionRelationship(
            original="Portland Cement",
            substitute="Fly Ash",
            max_ratio=0.30,
            effects="Reduced early strength (-15% at 7 days), improved durability (+20% chloride resistance), reduced heat of hydration",
            conditions="Class F fly ash with LOI < 6%",
            property_changes={
                "strength_7d": -15,
                "durability": 20,
                "heat_of_hydration": -25,
            },
        ),
        SubstitutionRelationship(
            original="Portland Cement",
            substitute="Silica Fume",
            max_ratio=0.10,
            effects="Increased strength (+15-20%), reduced permeability, improved chemical resistance",
            conditions="Requires superplasticizer for workability, thorough mixing essential",
            property_changes={
                "strength_28d": 18,
                "permeability": -40,
                "workability": -30,
            },
        ),
        SubstitutionRelationship(
            original="Portland Cement",
            substitute="Ground Granulated Blast Furnace Slag",
            max_ratio=0.70,
            effects="Reduced early strength, significantly improved durability, lower CO2 footprint",
            conditions="Slag activity index > 75%, proper curing essential",
            property_changes={
                "strength_7d": -25,
                "durability": 35,
                "co2_reduction": 40,
            },
        ),
        SubstitutionRelationship(
            original="Natural Aggregate",
            substitute="Recycled Concrete Aggregate",
            max_ratio=0.30,
            effects="Slight reduction in strength (-5-10%), increased water absorption",
            conditions="RCA must be clean, properly graded, <1% contaminants",
            property_changes={"strength_28d": -8, "water_absorption": 15},
        ),
    ]

    # ==========================================================================
    # Synergy Relationships (NEW - for material discovery)
    # ==========================================================================

    synergies = [
        SynergyRelationship(
            component1="Silica Fume",
            component2="Superplasticizer",
            effect_type="SYNERGY",
            effect="Enables high silica fume content (8-10%) with acceptable workability; superplasticizer disperses ultrafine particles",
            strength=0.9,
            mechanism="Superplasticizer prevents agglomeration of ultrafine silica particles",
            conditions="Polycarboxylate-based superplasticizer recommended",
        ),
        SynergyRelationship(
            component1="Fly Ash",
            component2="Silica Fume",
            effect_type="SYNERGY",
            effect="Combined pozzolanic reaction improves both early and late strength; silica fume compensates for fly ash's slow reaction",
            strength=0.85,
            mechanism="Multi-scale pozzolanic activity from particle size distribution",
            conditions="Typical ratio: 20% fly ash + 5-8% silica fume",
        ),
        SynergyRelationship(
            component1="Chlorides",
            component2="Steel Reinforcement",
            effect_type="ANTAGONISTIC",
            effect="Causes steel corrosion and structural degradation; chloride ions break passive oxide layer on steel",
            strength=0.95,
            mechanism="Chloride-induced pitting corrosion destroys protective passive layer",
            conditions="Critical chloride threshold: 0.2-0.4% by cement weight",
        ),
        SynergyRelationship(
            component1="High Alkali Content",
            component2="Reactive Silica Aggregates",
            effect_type="ANTAGONISTIC",
            effect="Causes alkali-silica reaction (ASR) leading to expansion and cracking",
            strength=0.9,
            mechanism="Alkali hydroxides react with amorphous silica forming expansive gel",
            conditions="Risk when alkali content > 0.6% Na2Oe and reactive aggregates present",
        ),
        SynergyRelationship(
            component1="Calcium Aluminate",
            component2="Gypsum",
            effect_type="SYNERGY",
            effect="Controls setting time; forms ettringite which regulates early hydration",
            strength=0.8,
            mechanism="CO3A + gypsum → ettringite (slow reaction) instead of flash set",
            conditions="Gypsum content 3-5% optimum",
        ),
    ]

    # ==========================================================================
    # Quantitative Properties (NEW)
    # ==========================================================================

    quantitative_properties = [
        QuantitativeProperty(
            name="Compressive Strength",
            value=45.0,
            unit="MPa",
            conditions="28 days, water cured",
            min_value=40.0,
            max_value=60.0,
            test_method="ASTM C39",
        ),
        QuantitativeProperty(
            name="Initial Setting Time",
            value=120,
            unit="minutes",
            conditions="Normal consistency",
            min_value=45,
            max_value=375,
            test_method="ASTM C191",
        ),
        QuantitativeProperty(
            name="Water-Cement Ratio",
            value=0.45,
            unit="ratio",
            conditions="Normal strength concrete",
            min_value=0.35,
            max_value=0.55,
        ),
    ]

    return ExtractionResult(
        materials=[portland_cement, fly_ash, silica_fume],
        relationships=relationships,
        substitutions=substitutions,
        synergies=synergies,
        quantitative_properties=quantitative_properties,
        source_document="Demo: Concrete Materials Composition Database",
        extraction_confidence=0.95,
    )
