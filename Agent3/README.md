# Agent 3: Automated DCPR Builder

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure LLM
cp .env.example .env
# Edit .env - choose Gemini OR Ollama

# 3. Add creators to process
# Edit build_personas.py line 30:
# CREATOR_IDS_TO_PROCESS = ["creator1", "creator2"]

# 4. Run automated processing
python build_personas.py
```

## 📋 Configuration

### Option 1: Gemini API (Cloud)

```bash
# .env
GEMINI_API_KEY=your_key_here
USE_OLLAMA=false
```

Get key: https://makersuite.google.com/app/apikey

### Option 2: Local Ollama

```bash
# 1. Install Ollama
curl https://ollama.ai/install.sh | sh

# 2. Pull model
ollama pull llama3.2

# 3. Configure .env
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

## 🎯 How It Works

```
build_personas.py
    ↓
For each creator in list:
    ↓
Check if stats exist?
    ↓
NO  → Full Build          YES → Incremental Update
    ↓                          ↓
Load all posts            Load only new posts
    ↓                          ↓
Calculate from scratch    Update existing stats
    ↓                          ↓
Generate identity         Generate identity
    ↓                          ↓
Save DCPR + Stats         Save updated DCPR + Stats
```

## 📁 Input Data Structure

Place Agent 2 outputs here:
```
data/creators/
├── creator_001/
│   ├── post_001.json
│   ├── post_002.json
│   └── reel_001.json
├── creator_002/
│   └── ...
```

## 📊 Output Structure

```
data/personas/
├── creator_001/
│   ├── dcpr_latest.json
│   ├── dcpr_stats_latest.json
│   ├── dcpr_v20241009_120530_abc.json
│   └── dcpr_stats_v20241009_120530_abc.json
```

## ⚙️ Configuration Options

### build_personas.py

```python
# Line 30: List of creators to process
CREATOR_IDS_TO_PROCESS = [
    "creator_001",
    "creator_002",
]

# Line 36: Concurrent processing
MAX_CONCURRENT_CREATORS = 3  # Process 3 at once
```

## 🔄 Incremental Updates

**First run** (no stats exist):
- Processes ALL posts
- Calculates complete DCPR
- Saves DCPR + Stats

**Subsequent runs** (stats exist):
- Loads existing stats
- Processes ONLY new posts
- Updates stats incrementally
- Recalculates DCPR from updated stats

**Efficiency**: 90% faster for small updates

## 📝 Logs

```bash
# View logs
tail -f ./logs/agent3.log

# Search for errors
grep ERROR ./logs/agent3.log
```

## 🧪 Testing

### Test with Sample Data
```python
# build_personas.py uses sample data if no real files found
CREATOR_IDS_TO_PROCESS = ["test_creator"]
```

### Test Single Creator
```python
CREATOR_IDS_TO_PROCESS = ["creator_001"]
MAX_CONCURRENT_CREATORS = 1
```

## 🚨 Troubleshooting

### "No posts found"
- Check `data/creators/{creator_id}/` has JSON files
- Or let it generate sample data automatically

### "Gemini API error"
- Verify GEMINI_API_KEY in .env
- Or switch to Ollama (USE_OLLAMA=true)

### "Ollama connection refused"
- Start Ollama: `ollama serve`
- Verify: `curl http://localhost:11434`

### "Validation failed"
- Check logs for specific errors
- Verify Agent 2 output format matches schema

## 📈 Performance

| Creators | Posts Each | Time (Full Build) | Time (Incremental) |
|----------|------------|-------------------|---------------------|
| 1        | 50         | ~30s              | ~3s (1 new post)   |
| 5        | 50         | ~2min             | ~15s (5 new posts) |
| 10       | 50         | ~4min             | ~30s (10 new posts)|

**With parallel processing** (MAX_CONCURRENT_CREATORS=3):
- 10 creators: ~1.5min

## 🔧 Advanced Usage

### Process Specific Creators
```python
import asyncio
from build_personas import process_creator

async def main():
    await process_creator("specific_creator_id")

asyncio.run(main())
```

### Custom Processing Logic
```python
# Edit build_personas.py
# Modify process_creator() function
# Add custom pre/post processing
```

## 📦 Dependencies

- fastapi==0.104.1 (optional - for API later)
- numpy==1.26.2
- scikit-learn==1.3.2
- google-generativeai==0.3.2 (if using Gemini)
- requests (if using Ollama)

## ✅ Success Indicators

```
INFO - Processing creator: creator_001
INFO - ✓ Stats found for creator_001 - INCREMENTAL UPDATE
INFO - Found 3 new posts
INFO - Updated: 45 → 48 posts
INFO - ✅ Successfully processed creator_001
```

## 🎯 Next Steps

1. **Place real data** in `data/creators/`
2. **Configure LLM** (Gemini or Ollama)
3. **Set creator list** in build_personas.py
4. **Run**: `python build_personas.py`
5. **Check output** in `data/personas/`