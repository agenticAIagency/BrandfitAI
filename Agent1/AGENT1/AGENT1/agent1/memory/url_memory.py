from llama_index.core import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from typing import List
import os


class URLMemory:
    """Tracks scraped URLs using LlamaIndex"""
    
    def __init__(self, storage_dir: str = "storage"):
        os.makedirs(storage_dir, exist_ok=True)
        self.path = os.path.join(storage_dir, "docstore.json")
        
        # Load or create
        if os.path.exists(self.path):
            self.store = SimpleDocumentStore.from_persist_path(self.path)
            print(f"📂 Loaded memory: {len(self.store.docs)} URLs")
        else:
            self.store = SimpleDocumentStore()
            print("🆕 Created new memory")
    
    def add_urls(self, urls: List[str], creator: str) -> None:
        """Save scraped URLs"""
        for url in urls:
            doc = Document(
                text=url,
                metadata={"creator": creator},
                doc_id=f"url_{hash(url)}"
            )
            self.store.add_documents([doc])
        
        self.store.persist(self.path)
        print(f"💾 Saved {len(urls)} URLs for @{creator}")
    
    def get_new_urls(self, all_urls: List[str]) -> List[str]:
        """Filter out already scraped URLs"""
        scraped = {doc.text for doc in self.store.docs.values()}
        new = [url for url in all_urls if url not in scraped]
        
        print(f"\n📊 Memory Check:")
        print(f"   Total: {len(all_urls)} | Scraped: {len(scraped)} | New: {len(new)}")
        
        return new
    
    def get_stats(self) -> dict:
        """Memory statistics"""
        return {
            "total_urls": len(self.store.docs),
            "storage": self.path
        }
