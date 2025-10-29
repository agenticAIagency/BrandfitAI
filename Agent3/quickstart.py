"""
Quick Start Script for Agent 3
Demonstrates complete persona building workflow with sample data
"""

import asyncio
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Agent 3 components
from bdi.engine import BDIEngine
from modules.ingestion import DataIngestionModule
from modules.features import FeatureExtractionModule
from modules.fusion import MultimodalFusionModule
from modules.temporal import TemporalAnalysisModule
from modules.summarization import PersonaSummarizationModule
from modules.validation import ValidationModule
from memory.store import PersonaMemoryStore
from memory.versioning import VersionManager
from utils.confidence import ConfidenceCalculator


async def run_persona_builder_demo():
    """
    Complete demonstration of Agent 3 persona building pipeline
    """
    print("\n" + "="*70)
    print("🤖 Agent 3: Creator Persona Builder - Demo")
    print("="*70 + "\n")
    
    # Configuration
    creator_id = "demo_creator_001"
    
    # Initialize components
    print("📦 Initializing components...")
    ingestion = DataIngestionModule()
    features = FeatureExtractionModule(max_workers=5)
    fusion = MultimodalFusionModule()
    temporal = TemporalAnalysisModule()
    summarization = PersonaSummarizationModule()
    validation = ValidationModule()
    memory_store = PersonaMemoryStore()
    version_manager = VersionManager()
    confidence_calc = ConfidenceCalculator()
    bdi_engine = BDIEngine(creator_id=creator_id)
    
    print("✅ Components initialized\n")
    
    # Step 1: Load Data
    print("📥 STEP 1: Loading creator data...")
    try:
        # Try loading real data first
        raw_data = await ingestion.load_creator_data(creator_id)
        print(f"   Loaded {len(raw_data)} posts from filesystem")
    except FileNotFoundError:
        # Generate sample data if no real data exists
        print("   No real data found, generating sample data...")
        raw_data = await ingestion.load_sample_data(creator_id, num_posts=15)
        print(f"   Generated {len(raw_data)} sample posts")
    
    # Get data statistics
    stats = ingestion.get_data_statistics(raw_data)
    print(f"   ├─ Temporal span: {stats['temporal_span_days']} days")
    print(f"   ├─ Content types: {stats['content_type_distribution']}")
    print(f"   └─ Avg likes: {stats['avg_likes']:.0f}\n")
    
    # Step 2: BDI Planning
    print("🧠 STEP 2: BDI adaptive planning...")
    bdi_engine.update_beliefs({
        "raw_data": raw_data,
        "data_quality": 0.8,
        "posts_count": len(raw_data)
    })
    bdi_engine.set_desires()
    execution_plan = bdi_engine.formulate_intentions()
    
    print(f"   Strategy: {execution_plan['strategy']}")
    print(f"   Intentions: {len(bdi_engine.intentions)} steps")
    print(f"   Estimated duration: {execution_plan['estimated_duration_seconds']}s\n")
    
    # Step 3: Pre-validation
    print("✓ STEP 3: Validating input data...")
    validation_result = validation.validate_input_quality(raw_data, min_posts=10)
    
    if validation_result["is_valid"]:
        print(f"   ✅ Validation passed (score: {validation_result['score']:.2f})")
        print(f"   └─ Completeness: {validation_result['completeness_rate']:.1%}\n")
    else:
        print(f"   ⚠️  Validation issues:")
        for issue in validation_result["issues"]:
            print(f"      - {issue}")
        print()
    
    # Step 4: Feature Extraction
    print("🔍 STEP 4: Extracting multimodal features...")
    feature_vectors = await features.extract_parallel(raw_data)
    print(f"   ✅ Extracted features from {len(feature_vectors)} posts")
    
    # Compute statistics
    feature_stats = features.compute_feature_statistics(feature_vectors)
    print(f"   ├─ Text diversity: {feature_stats['text_features']['diversity_score']:.2f}")
    print(f"   ├─ Visual diversity: {feature_stats['visual_features']['diversity_score']:.2f}")
    print(f"   └─ Engagement consistency: {feature_stats['engagement_patterns']['consistency_score']:.2f}\n")
    
    # Step 5: Multimodal Fusion
    print("🔀 STEP 5: Fusing multimodal features...")
    fused_data = fusion.fuse(feature_vectors)
    print(f"   ✅ Fusion complete")
    print(f"   ├─ Consistency score: {fused_data['consistency_metrics']['overall']:.2f}")
    print(f"   ├─ Authenticity score: {fused_data['authenticity_score']['overall_score']:.2f}")
    print(f"   ├─ Theme clusters: {fused_data['theme_clusters']['num_themes']}")
    print(f"   └─ Theme diversity: {fused_data['theme_clusters']['diversity_score']:.2f}\n")
    
    # Step 6: Temporal Analysis
    print("⏱️ STEP 6: Analyzing temporal patterns...")
    temporal_insights = temporal.analyze_evolution(fused_data, raw_data)
    print(f"   ✅ Temporal analysis complete")
    print(f"   ├─ Posting pattern: {temporal_insights['posting_cadence']['posting_pattern']}")
    print(f"   ├─ Style evolution: {temporal_insights['style_evolution']['evolution_trend']}")
    print(f"   ├─ Engagement trend: {temporal_insights['engagement_trends']['trend']}")
    print(f"   └─ Temporal consistency: {temporal_insights['temporal_consistency_score']:.2f}\n")
    
    # Step 7: Persona Summarization
    print("📝 STEP 7: Generating persona summary...")
    persona_draft = await summarization.generate_persona(
        fused_data=fused_data,
        temporal_insights=temporal_insights,
        brand_context=None
    )
    print(f"   ✅ Persona generated")
    print(f"   ├─ Headline: {persona_draft['creator_summary']['headline']}")
    print(f"   ├─ Content quality: {persona_draft['content_profile']['content_quality']}")
    print(f"   ├─ Growth stage: {persona_draft['growth_trajectory']['current_stage']}")
    print(f"   └─ Momentum: {persona_draft['growth_trajectory']['momentum']}\n")
    
    # Step 8: Calculate Confidence
    print("📊 STEP 8: Calculating confidence score...")
    confidence_score = confidence_calc.calculate(
        num_posts=len(raw_data),
        data_quality=validation_result["score"],
        temporal_coverage=temporal_insights["coverage_days"],
        consistency_score=fused_data["consistency_metrics"]["overall"]
    )
    interpretation = confidence_calc.interpret_confidence(confidence_score)
    
    persona_draft["confidence_score"] = confidence_score
    persona_draft["recommendation"] = interpretation["recommendation"]
    
    print(f"   ✅ Confidence: {confidence_score:.2f}")
    print(f"   ├─ Level: {interpretation['level']}")
    print(f"   ├─ Status: {interpretation['recommendation']}")
    print(f"   └─ Description: {interpretation['description']}\n")
    
    # Step 9: Output Validation
    print("✓ STEP 9: Validating output...")
    output_validation = validation.validate_output(persona_draft)
    
    if output_validation["is_valid"]:
        print(f"   ✅ Output validation passed\n")
    else:
        print(f"   ⚠️  Validation errors:")
        for error in output_validation["errors"]:
            print(f"      - {error}")
        print()
    
    # Step 10: Persist to Memory
    print("💾 STEP 10: Persisting to memory store...")
    version_id = version_manager.create_version(creator_id)
    
    final_persona = {
        "creator_id": creator_id,
        "persona": persona_draft,
        "confidence_score": confidence_score,
        "version": version_id,
        "analyzed_posts_count": len(raw_data),
        "created_at": "2024-10-09T12:00:00Z",
        "status": interpretation["recommendation"],
        "recommendation": interpretation["recommendation"]
    }
    
    success = memory_store.save_persona(creator_id, final_persona, version_id)
    
    if success:
        print(f"   ✅ Persona saved (version: {version_id})")
        print(f"   └─ Storage location: ./data/personas/{creator_id}/\n")
    
    # Summary
    print("="*70)
    print("✨ PERSONA BUILDING COMPLETE")
    print("="*70)
    print(f"\nCreator ID: {creator_id}")
    print(f"Version: {version_id}")
    print(f"Posts Analyzed: {len(raw_data)}")
    print(f"Confidence: {confidence_score:.2f} ({interpretation['level']})")
    print(f"Status: {interpretation['recommendation']}")
    
    print("\n📋 Key Metrics:")
    print(f"   • Consistency: {fused_data['consistency_metrics']['overall']:.2f}")
    print(f"   • Authenticity: {fused_data['authenticity_score']['overall_score']:.2f}")
    print(f"   • Temporal Consistency: {temporal_insights['temporal_consistency_score']:.2f}")
    print(f"   • Theme Diversity: {fused_data['theme_clusters']['diversity_score']:.2f}")
    
    print("\n🎯 Persona Highlights:")
    print(f"   • {persona_draft['creator_summary']['headline']}")
    print(f"   • Traits: {', '.join(persona_draft['creator_summary']['personality_traits'][:3])}")
    print(f"   • Primary Themes: {', '.join(persona_draft['content_profile']['primary_themes'][:3])}")
    
    if persona_draft.get('opportunities'):
        print(f"\n💡 Opportunities:")
        for opp in persona_draft['opportunities'][:3]:
            print(f"   • {opp}")
    
    if persona_draft.get('red_flags'):
        print(f"\n⚠️  Red Flags:")
        for flag in persona_draft['red_flags']:
            print(f"   • {flag}")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70 + "\n")
    
    # Return persona for further use
    return final_persona


def display_persona_json(persona: dict, output_file: str = "demo_persona.json"):
    """Save and display persona JSON"""
    print(f"\n💾 Saving persona to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(persona, f, indent=2)
    
    print(f"✅ Persona saved to {output_file}")
    print("\nTo view the complete persona:")
    print(f"   cat {output_file}")
    print("\nTo use in your application:")
    print(f"   import json")
    print(f"   with open('{output_file}') as f:")
    print(f"       persona = json.load(f)")


if __name__ == "__main__":
    print("\n🚀 Starting Agent 3 Demo...\n")
    
    try:
        # Run the demo
        persona = asyncio.run(run_persona_builder_demo())
        
        # Save output
        display_persona_json(persona)
        
        print("\n✅ All done! Check the generated persona file.\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}\n")
        exit(1)