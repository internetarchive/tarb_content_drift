curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5000/calculate_relevance'
curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5001/calculate_lda_relevance'
curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5002/calculate_llm_relevance'
  