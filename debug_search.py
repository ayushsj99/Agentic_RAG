# debug_search.py
from database.database_config import vector_store

# Search for the exact text
results = vector_store.similarity_search("23F1002471", k=20)

print(f"\n{'='*70}")
print(f"SEARCHING FOR: '23F1002471'")
print(f"{'='*70}\n")

if not results:
    print("❌ CRITICAL: Roll number '23F1002471' NOT FOUND in database!")
    print("   This means the chunk was either:")
    print("   1. Never ingested")
    print("   2. Split incorrectly during chunking")
    print("   3. Embedded poorly and ranked very low")
else:
    print(f"✅ Found {len(results)} results\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:300]}")
        print()

# Also search for author name
print(f"\n{'='*70}")
print(f"SEARCHING FOR: 'Ayush Jadhav'")
print(f"{'='*70}\n")

results2 = vector_store.similarity_search("Ayush Jadhav", k=20)
if results2:
    print(f"✅ Found {len(results2)} results\n")
    for i, doc in enumerate(results2[:3], 1):
        print(f"--- Result {i} ---")
        print(f"Content: {doc.page_content[:300]}")
        print()
else:
    print("❌ Author name also not found!")