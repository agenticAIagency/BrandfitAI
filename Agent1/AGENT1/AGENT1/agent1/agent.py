from crewai import Agent, Task, Crew, LLM
from tools.instagram_tools import AnalyticsTool, DownloaderTool
from memory.url_memory import URLMemory
from typing import List
import os


class DataIngestionAgent:
    """Agent 1: Data Collector with Memory"""
    
    def __init__(self, instagram_username: str, instagram_password: str, gemini_api_key: str):
        self.memory = URLMemory()
        self.insta_user = instagram_username
        self.insta_pass = instagram_password
        
        # Set up Gemini 2.5 Flash LLM
        llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=gemini_api_key
        )
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Instagram Data Collector",
            goal="Efficiently scrape Instagram data without duplicates",
            backstory="Expert at collecting Instagram analytics and videos with memory tracking",
            tools=[AnalyticsTool(), DownloaderTool()],
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    def scrape(self, reel_urls: List[str], creator_name: str, force: bool = False):
        """
        Scrape creator data with memory checking
        
        Args:
            reel_urls: List of Instagram URLs
            creator_name: Creator username
            force: If True, ignore memory and scrape all
        """
        print(f"\n{'='*70}")
        print(f"🎯 Agent 1: Scraping @{creator_name}")
        print(f"{'='*70}\n")
        
        # Check memory
        if force:
            urls_to_scrape = reel_urls
            print("⚠️ Force mode: Scraping all URLs")
        else:
            urls_to_scrape = self.memory.get_new_urls(reel_urls)
        
        if not urls_to_scrape:
            print("\n✅ All URLs already scraped!")
            return {"status": "up_to_date", "creator": creator_name}
        
        # Create tasks
        task1 = Task(
            description=f"""
            Scrape analytics for {len(urls_to_scrape)} URLs from @{creator_name}
            Use Analytics Scraper tool with credentials.
            """,
            expected_output="Analytics scraped successfully",
            agent=self.agent
        )
        
        task2 = Task(
            description=f"""
            Download {len(urls_to_scrape)} reels for @{creator_name}
            Use Reel Downloader tool.
            """,
            expected_output="Reels downloaded successfully",
            agent=self.agent,
            context=[task1]
        )
        
        # Execute
        print(f"\n🔄 Scraping {len(urls_to_scrape)} new URLs...\n")
        crew = Crew(
            agents=[self.agent],
            tasks=[task1, task2],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Update memory
        self.memory.add_urls(urls_to_scrape, creator_name)
        
        print(f"\n✅ Scraping complete!")
        
        return {
            "status": "success",
            "creator": creator_name,
            "scraped": len(urls_to_scrape),
            "total": len(reel_urls)
        }