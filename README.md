# Digital Creator Matching System: Multi-Agent Architecture Report

This document details the architecture, flow, inputs, outputs, and specific technology usage for each agent within the multi-agent system, based on the provided code structure.

The system is designed as a pipeline where the structured output of one agent serves as the input for the next, reflecting a **Hierarchical Design Pattern**.

---

## 1. Agent 2: The Interpreter / Analyzer

**Role**: Converts raw social media posts (media and metadata) into rich, structured JSON files for downstream analysis.

### Flow and Logic
1.  **Input Acquisition**: Finds raw post folders containing `metadata.json` and media files (`.mp4`, `.jpg`).
2.  **Parallel Execution**: Uses Python's **Multiprocessing `Pool`** (`agent2_processor.py`) to process posts concurrently.
3.  **Multimodal Analysis**: Calls the **LLM (Gemini API)**, providing the media file and prompt to generate comprehensive content, visual, and linguistic insights.
4.  **Guardrails**: Enforces output structure using **Pydantic Schemas** (`modules/schemas.py`) for the LLM response.
5.  **Embedding**: Uses **Sentence Transformer** to generate a vector representation of the post content.
6.  **Output Save**: Saves the final structured data, including the embedding, as a Rich JSON Post file.

| Concept | Usage | Details |
| :--- | :--- | :--- |
| **Input** | Raw Post Data | `metadata.json` + media file (`.mp4`, `.jpg`). |
| **Output** | Rich JSON Posts | Structured JSON with analysis and embedding vector. |
| **Tools** | LLM, Sentence Transformer | Gemini API for multimodal analysis. |
| **Design Pattern** | **N/A** (Batch) | Uses Multiprocessing for execution parallelism. |
| **LangGraph / CrewAI** | Not Used | Simple Python batch processing is used. |
| **Guardrails** | Used | **Pydantic Schemas** enforce structured LLM output. |
| **Llama Index** | Not Used | |
| **LTM / STM** | Stateless | Processes data and passes output directly. |

---

## 2. Agent 3: The Persona Builder (DCPR)

**Role**: Aggregates structured post data into a comprehensive **Digital Creator Persona Report (DCPR)**, exposed as a service for efficient, versioned updates.

### Flow and Logic
1.  **Service Entry**: Exposed via **FastAPI** endpoints (`main.py`).
2.  **Orchestration**: Managed by **LangGraph** (`agent3_graph.py`).
3.  **Hierarchical Routing**: A conditional node checks the **LTM** (via `memory/store.py`).
    * **IF** stats exist $\rightarrow$ Route to **Incremental Update**.
    * **ELSE** $\rightarrow$ Route to **Full Build**.
    *(This implements the **BDI Design Pattern**)*.
4.  **DCPR Calculation**: Aggregates statistical metrics (`modules/dcpr_calculator.py`).
5.  **LLM Summarization**: Uses an **LLM** (Gemini/Ollama) to generate a qualitative summary layer.
6.  **Validation & Save**: Applies **Guardrails** (`modules/validation.py`) and saves versioned reports to **LTM** (`PersonaMemoryStore`). The aggregate counts are saved to **STM** for the next incremental run.

| Concept | Usage | Details |
| :--- | :--- | :--- |
| **Input** | Rich JSON Posts | Structured posts from Agent 2. |
| **Output** | Versioned DCPR Files | DCPR (LTM) & Aggregate Stats (STM). |
| **Design Pattern** | **Hierarchical, BDI** | Conditional flow management (Full vs. Incremental). |
| **LangGraph** | Used | Core workflow and state management. |
| **FastAPI** | Used | Exposes the agent as a service API. |
| **Guardrails** | Used | **Pydantic Validation** on final DCPR schema. |
| **Long Term Memory** | Used | `PersonaMemoryStore` for versioned historical DCPR data. |
| **Short Term Memory** | Used | `dcpr_stats_latest.json` for fast incremental updates. |
| **Langsmith** | Configured | Tracing and monitoring of the LangGraph execution. |

---

## 3. Agent 4: The Creator Evaluator

**Role**: Calculates performance scores, performs clustering, and translates DCPR data into structured, quantitative metrics for the matchmaker.

### Flow and Logic
1.  **Service Entry**: Exposed via **FastAPI** endpoint (`main.py`).
2.  **Orchestration**: Managed by **LangGraph** (`agent4_graph.py`) in a sequential pipeline.
3.  **Ingestion**: Loads all latest DCPR files from Agent 3's **LTM**.
4.  **Clustering**: Uses **Sentence Transformer** and **KMeans** (`modules/clustering.py`) to group creators by similarity.
5.  **Scoring**: Calculates weighted **Quality Score** and **Performance Score** based on raw metrics (`modules/scoring.py`).
6.  **Output Save**: Saves the final result as a master CSV file (`creator_evaluation_latest.csv`) to its **LTM**.

| Concept | Usage | Details |
| :--- | :--- | :--- |
| **Input** | DCPR JSON Files | Latest persona reports from Agent 3's LTM. |
| **Output** | Evaluation CSV | Creator scores, cluster IDs, and semantic vectors. |
| **Design Pattern** | **N/A** | Sequential pipeline, purely computational. |
| **LangGraph** | Used | Sequential workflow management. |
| **FastAPI** | Used | Exposes the evaluation trigger as a service API. |
| **Tools** | Sentence Transformer, Scikit-learn, Pandas | Used for scientific computing, embedding, and clustering. |
| **Long Term Memory** | Used | `EvaluationStore` persists the output CSV for Agent 5. |
| **Llama Index** | Not Used | |
| **Crew API** | Not Used | |

---

## 4. Agent 5: The Matchmaker / Outreach Planner

**Role**: Takes a user query (campaign brief), matches the best creators using vector search, and drafts personalized outreach emails.

### Flow and Logic
1.  **Orchestration**: Uses native **Python `asyncio`** (`agent5_processor.py`) for parallel execution.
2.  **Query Parsing**: An **LLM** is used with a **structured output schema (Guardrail)** to convert the human brief into machine-readable constraints (e.g., min-followers, keywords).
3.  **Matching (ReAct - Action/Observation)**:
    * **Vector Search**: Embeds the user query and finds semantic matches against creator vectors (from Agent 4's CSV).
    * **Filtering**: Applies hard constraints from the parsed query.
    * **Ranking**: Ranks final candidates by combined Quality/Performance scores and similarity.
4.  **Email Drafting (Hierarchical Task Delegation)**: Uses **`asyncio.gather`** to launch parallel **LLM** calls for the top N candidates. The LLM drafts the email, using the matched creator's full DCPR (**LTM**) for deep personalization.

| Concept | Usage | Details |
| :--- | :--- | :--- |
| **Input** | User Query (Text) | Campaign brief. |
| **Output** | Ranked List & Emails | List of top creators and saved personalized email drafts. |
| **Design Pattern** | **ReAct Variant, Hierarchical** | **ReAct** (Reasoning $\rightarrow$ Tool Use $\rightarrow$ Final Action). **Hierarchical** (Parallel task delegation via `asyncio`). |
| **Guardrails** | Used | Pydantic Schema used on LLM output for Query Parsing. |
| **Long Term Memory** | Consumed | Reads Agent 4's Evaluation CSV and Agent 3's DCPR JSON files for context. |
| **Crew API** | Not Used | Native `asyncio` is used instead of CrewAI. |
| **MCP** | Not Used | |
