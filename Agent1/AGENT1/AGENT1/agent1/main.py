from agent import DataIngestionAgent
import json
import os

# Configure Gemini 2.5 Flash
os.environ["LLM_PROVIDER"] = "google"
os.environ["GEMINI_API_KEY"] = "AIzaSyBwkU_V1E3OYak9g0M4CRDSNZDevPL1h1E"
os.environ["LLM_MODEL"] = "gemini-2.5-flash"

# Instagram credentials
INSTAGRAM_USERNAME = "laxm.ankumar3735"
INSTAGRAM_PASSWORD = "laxman12345"

# Gemini API key
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# URLs to scrape
REEL_URLS = [
    "https://www.instagram.com/runtimebrt/reel/DMaeWnlvG4p/",
    "https://www.instagram.com/runtimebrt/reel/DMelIKWPHVW/",
    "https://www.instagram.com/runtimebrt/reel/DMkLREsvjKe/",
    "https://www.instagram.com/runtimebrt/reel/DMnYh4RPlou/",
    "https://www.instagram.com/runtimebrt/reel/DMxaytfBdVD/",
    "https://www.instagram.com/runtimebrt/reel/DM0OgvlP8tS/",
    "https://www.instagram.com/runtimebrt/reel/DM2hcxGv2j1/",
    "https://www.instagram.com/runtimebrt/reel/DM5FvWUP4fF/",
    "https://www.instagram.com/runtimebrt/reel/DM96RswPCeF/",
    "https://www.instagram.com/runtimebrt/reel/DNAVMQzPpp5/",
    "https://www.instagram.com/runtimebrt/reel/DONOXphj17e/",
    "https://www.instagram.com/runtimebrt/reel/DON2bXrjxwF/",
    "https://www.instagram.com/runtimebrt/reel/DOQEuuFD2HA/",
    # Add remaining 78 URLs here...
]

def main():
    # Check if API key is set
    if GEMINI_API_KEY == "your-gemini-api-key-here":
        print("❌ ERROR: Please add your Gemini API key in main.py")
        print("   Get it from: https://aistudio.google.com/apikey")
        return
    
    print("="*70)
    print(" "*20 + "AGENT 1: DATA INGESTION")
    print("="*70)
    
    # Initialize Agent 1
    agent1 = DataIngestionAgent(
        instagram_username=INSTAGRAM_USERNAME,
        instagram_password=INSTAGRAM_PASSWORD,
        gemini_api_key=GEMINI_API_KEY
    )
    
    # Run scraping
    result = agent1.scrape(
        reel_urls=REEL_URLS,
        creator_name="runtimebrt",
        force=False  # Set to True to ignore memory
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    # Show memory stats
    stats = agent1.memory.get_stats()
    print("\n" + "="*70)
    print("MEMORY STATS:")
    print("="*70)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()