from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type
import sys
import os

# Add parent directory to import your scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scrape2 import run_scraper
from instagram_reel_downloader import run_downloader


class AnalyticsInput(BaseModel):
    reel_urls: List[str] = Field(..., description="Instagram reel URLs")
    creator_name: str = Field(..., description="Creator username")
    username: str = Field(..., description="Instagram login")
    password: str = Field(..., description="Instagram password")


class AnalyticsTool(BaseTool):
    name: str = "Analytics Scraper"
    description: str = "Scrapes Instagram reel analytics (likes, views, comments)"
    args_schema: Type[BaseModel] = AnalyticsInput
    
    def _run(self, reel_urls: List[str], creator_name: str, username: str, password: str) -> str:
        try:
            print(f"\n🔍 Scraping analytics for {len(reel_urls)} URLs...")
            run_scraper(reel_urls, creator_name, username, password)
            return f"✅ Scraped {len(reel_urls)} URLs"
        except Exception as e:
            return f"❌ Error: {str(e)}"


class DownloaderInput(BaseModel):
    reel_urls: List[str] = Field(..., description="Instagram reel URLs")
    creator_name: str = Field(..., description="Creator username")


class DownloaderTool(BaseTool):
    name: str = "Reel Downloader"
    description: str = "Downloads Instagram reels as MP4 files"
    args_schema: Type[BaseModel] = DownloaderInput
    
    def _run(self, reel_urls: List[str], creator_name: str) -> str:
        try:
            print(f"\n⬇️ Downloading {len(reel_urls)} reels...")
            run_downloader(reel_urls, creator_name)
            return f"✅ Downloaded {len(reel_urls)} reels"
        except Exception as e:
            return f"❌ Error: {str(e)}"