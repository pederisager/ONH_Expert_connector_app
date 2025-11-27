
import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from app.config_loader import load_app_config, load_models_config, load_staff_profiles
from app.index import IndexPaths, LocalVectorStore, create_embedding_backend
from app.match_engine import MatchEngine
from app.rag import EmbeddingRetriever, RetrievalQuery
from app.routes import _retrieval_result_to_match_item, _match_via_retriever, MatchRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("--- Starting Reproduction Script ---")
    
    # 1. Load Configuration
    print("Loading configuration...")
    app_config = load_app_config()
    models_config = load_models_config()
    staff_profiles = load_staff_profiles()
    
    # 2. Initialize Components
    print("Initializing components...")
    match_engine = MatchEngine(
        embedding_model_name=models_config.embedding_model.name,
        embedding_backend=models_config.embedding_model.backend,
        embedding_device=models_config.embedding_model.device or "auto",
        embedding_endpoint=models_config.embedding_model.endpoint,
    )

    index_root = (PROJECT_ROOT / app_config.rag.index_root).resolve()
    index_paths = IndexPaths(root=index_root)
    vector_store = LocalVectorStore(index_paths.vectors_dir)
    retriever_embedder = create_embedding_backend(models_config, app_config=app_config)
    
    embedding_retriever = EmbeddingRetriever(
        vector_store=vector_store,
        embedder=retriever_embedder,
        min_score=app_config.results.min_similarity_score,
        max_chunks_per_staff=3,
    )
    
    vector_index_ready = len(vector_store) > 0
    print(f"Vector index ready: {vector_index_ready} (Items: {len(vector_store)})")

    # 3. Define Test Query
    test_theme = "psykologi" # A common theme likely to be in the dataset
    print(f"Test Query: '{test_theme}'")
    
    # 4. Run Retrieval
    if vector_index_ready:
        print("\n--- Running EmbeddingRetriever ---")
        query = RetrievalQuery(text=test_theme, top_k=5)
        try:
            results = embedding_retriever.retrieve(query)
            print(f"Found {len(results)} results.")
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Staff: {result.staff_name} ({result.staff_slug})")
                print(f"  Score: {result.score}")
                print(f"  Chunks: {len(result.chunks)}")
                for j, chunk in enumerate(result.chunks):
                    print(f"    Chunk {j+1} (Length: {len(chunk.text)}): {chunk.text[:100]}...")
                    
                # Convert to MatchItem to see the final output format
                match_item = _retrieval_result_to_match_item(result, [test_theme])
                if match_item:
                    print(f"  MatchItem Score Breakdown: {match_item.score_breakdown}")
                    print(f"  MatchItem Citations: {[c.snippet[:50] + '...' for c in match_item.citations]}")

        except Exception as e:
            print(f"EmbeddingRetriever failed: {e}")
    else:
        print("Vector index not ready. Skipping EmbeddingRetriever.")

    # 5. Run Legacy Matcher (if needed or for comparison)
    # Note: This requires fetching documents which might be slow/complex to mock here without full state.
    # We'll focus on the RAG part first as that's the primary complaint.

if __name__ == "__main__":
    asyncio.run(main())
