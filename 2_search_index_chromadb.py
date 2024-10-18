from chromadb_clip import VideoChromaDb

# Example: Search for a video using a text query
vdb = VideoChromaDb('db_chromadb_video4')
search_text = "A chiếc thuyền với chữ Me Kong"
results = vdb.search_by_text(search_text)

# Display results
for result in results['ids']:
    print(f"Found match: {result}")