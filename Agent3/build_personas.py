#!/usr/bin/env python3
"""
Agent 3: Automated DCPR Builder
Processes creators from list, builds/updates personas automatically

NO interactive menu - automated batch processing
"""
import sys
import asyncio
import logging
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

from modules.ingestion import DataIngestionModule
from modules.dcpr_calculator import DCPRCalculator
from modules.summarization import PersonaSummarizationModule
from modules.validation import ValidationModule
from memory.store import PersonaMemoryStore
from memory.versioning import VersionManager

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/agent3.log', encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout) # Explicitly set stream for handler
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
ingestion = DataIngestionModule()
calculator = DCPRCalculator()
summarization = PersonaSummarizationModule()
validation = ValidationModule()
memory_store = PersonaMemoryStore()
version_manager = VersionManager()

# ============================================================
# CONFIGURATION: List creators to process
# ============================================================
CREATOR_IDS_TO_PROCESS = [
    "demo_creator_001",
    "demo_creator_002",
    # Add more creator IDs here
]

# Process in parallel?
MAX_CONCURRENT_CREATORS = 3


async def process_creator(creator_id: str) -> bool:
    """
    Process a single creator: build or update DCPR
    
    Args:
        creator_id: Creator identifier
        
    Returns:
        Success status
    """
    try:
        logger.info(f"{'='*70}")
        logger.info(f"Processing creator: {creator_id}")
        logger.info(f"{'='*70}")
        
        # Step 1: Check if stats exist (determines strategy)
        stats_exist = memory_store.stats_exist(creator_id)
        
        if stats_exist:
            logger.info(f"✓ Stats found for {creator_id} - INCREMENTAL UPDATE")
            success = await incremental_update(creator_id)
        else:
            logger.info(f"✗ No stats for {creator_id} - FULL BUILD")
            success = await full_build(creator_id)
        
        if success:
            logger.info(f"✅ Successfully processed {creator_id}")
        else:
            logger.error(f"❌ Failed to process {creator_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error processing {creator_id}: {e}", exc_info=True)
        return False


async def full_build(creator_id: str) -> bool:
    """
    Full build: process all posts from scratch
    
    Args:
        creator_id: Creator identifier
        
    Returns:
        Success status
    """
    try:
        # Step 1: Load ALL posts
        logger.info("Loading all posts...")
        try:
            all_posts = await ingestion.load_creator_data(creator_id, file_pattern="*.json")
        except FileNotFoundError:
            logger.warning("No real data found, generating sample data")
            all_posts = await ingestion.load_sample_data(creator_id, num_posts=25)
        
        if len(all_posts) < 5:
            logger.error(f"Insufficient posts: {len(all_posts)} < 5 required")
            return False
        
        logger.info(f"Loaded {len(all_posts)} posts")
        
        # Step 2: Validate input quality
        logger.info("Validating input quality...")
        input_validation = validation.validate_input_quality(all_posts, min_posts=5)
        
        if not input_validation["is_valid"]:
            logger.warning(f"Input quality issues: {input_validation['issues']}")
            logger.info("Proceeding anyway (quality score: {:.2f})".format(input_validation['score']))
        
        # Step 3: Calculate DCPR from scratch
        logger.info("Calculating DCPR from scratch...")
        dcpr, stats = calculator.create_from_scratch(all_posts)
        
        # Set creator_id and version
        version_id = version_manager.create_version(creator_id)
        dcpr["creator_id"] = creator_id
        dcpr["persona_version"] = version_id
        
        logger.info(f"Calculated metrics for {stats['total_posts_analyzed']} posts")
        
        # Step 4: Generate identity summary with LLM
        logger.info("Generating identity summary with LLM...")
        identity_summary = await summarization.generate_identity_summary(dcpr)
        dcpr["identity_summary"] = identity_summary
        
        # Step 5: Validate output
        logger.info("Validating DCPR output...")
        output_validation = validation.validate_dcpr_output(dcpr)
        
        if not output_validation["is_valid"]:
            logger.error(f"Validation failed: {output_validation['errors']}")
            return False
        
        if output_validation["warnings"]:
            logger.warning(f"Validation warnings: {output_validation['warnings']}")
        
        # Step 6: Save DCPR and Stats
        logger.info("Saving DCPR and stats...")
        success = memory_store.save_persona_and_stats(
            creator_id=creator_id,
            dcpr=dcpr,
            stats=stats,
            version=version_id
        )
        
        if not success:
            logger.error("Failed to save DCPR")
            return False
        
        logger.info(f"✅ Full build complete - Version: {version_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error in full build: {e}", exc_info=True)
        return False


async def incremental_update(creator_id: str) -> bool:
    """
    Incremental update: process only new posts
    
    Args:
        creator_id: Creator identifier
        
    Returns:
        Success status
    """
    try:
        # Step 1: Load existing stats
        logger.info("Loading existing stats...")
        old_stats = memory_store.load_latest_stats(creator_id)
        
        if not old_stats:
            logger.warning("Stats file corrupted or missing, falling back to full build")
            return await full_build(creator_id)
        
        old_post_count = old_stats["total_posts_analyzed"]
        last_post_date = old_stats.get("last_post_date")
        
        logger.info(f"Existing stats: {old_post_count} posts, last: {last_post_date}")
        
        # Step 2: Load new posts only
        logger.info("Loading new posts...")
        try:
            new_posts = await ingestion.load_new_posts_since(creator_id, last_post_date)
        except Exception as e:
            logger.error(f"Error loading new posts: {e}")
            # Fallback: load all and filter
            all_posts = await ingestion.load_creator_data(creator_id, file_pattern="*.json")
            new_posts = [p for p in all_posts if p.get("metadata", {}).get("upload_time", "") > last_post_date]
        
        if len(new_posts) == 0:
            logger.info("No new posts to process")
            return True
        
        logger.info(f"Found {len(new_posts)} new posts")
        
        # Step 3: Update incrementally
        logger.info("Updating DCPR incrementally...")
        updated_dcpr, updated_stats = calculator.update_incrementally(old_stats, new_posts)
        
        # Set creator_id and new version
        new_version = version_manager.create_version(creator_id)
        updated_dcpr["creator_id"] = creator_id
        updated_dcpr["persona_version"] = new_version
        
        new_post_count = updated_stats["total_posts_analyzed"]
        logger.info(f"Updated: {old_post_count} → {new_post_count} posts")
        
        # Step 4: Generate new identity summary
        logger.info("Generating updated identity summary...")
        identity_summary = await summarization.generate_identity_summary(updated_dcpr)
        updated_dcpr["identity_summary"] = identity_summary
        
        # Step 5: Validate
        output_validation = validation.validate_dcpr_output(updated_dcpr)
        
        if not output_validation["is_valid"]:
            logger.error(f"Validation failed: {output_validation['errors']}")
            return False
        
        # Step 6: Save
        logger.info("Saving updated DCPR and stats...")
        success = memory_store.save_persona_and_stats(
            creator_id=creator_id,
            dcpr=updated_dcpr,
            stats=updated_stats,
            version=new_version
        )
        
        if not success:
            logger.error("Failed to save updated DCPR")
            return False
        
        logger.info(f"✅ Incremental update complete - Version: {new_version}")
        return True
        
    except Exception as e:
        logger.error(f"Error in incremental update: {e}", exc_info=True)
        return False


async def process_all_creators():
    """
    Process all creators (potentially in parallel)
    """
    logger.info("\n" + "="*70)
    logger.info("Agent 3: DCPR Builder - Automated Execution")
    logger.info("="*70)
    logger.info(f"Creators to process: {len(CREATOR_IDS_TO_PROCESS)}")
    logger.info(f"Max concurrent: {MAX_CONCURRENT_CREATORS}")
    logger.info("="*70 + "\n")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CREATORS)
    
    async def process_with_semaphore(creator_id: str):
        async with semaphore:
            return await process_creator(creator_id)
    
    # Process all creators
    tasks = [process_with_semaphore(cid) for cid in CREATOR_IDS_TO_PROCESS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Summary
    successes = sum(1 for r in results if r is True)
    failures = len(results) - successes
    
    logger.info("\n" + "="*70)
    logger.info("EXECUTION COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Successful: {successes}")
    logger.info(f"❌ Failed: {failures}")
    logger.info(f"📊 Total: {len(results)}")
    logger.info("="*70 + "\n")


def main():
    """Main entry point"""
    # Create directories
    Path("./logs").mkdir(exist_ok=True)
    Path("./data/creators").mkdir(parents=True, exist_ok=True)
    Path("./data/personas").mkdir(parents=True, exist_ok=True)
    
    # Run async processing
    asyncio.run(process_all_creators())


if __name__ == "__main__":
    main()