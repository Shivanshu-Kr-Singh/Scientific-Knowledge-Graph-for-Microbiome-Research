# Scientific Knowledge Graph API

REST API for querying the microbiome research knowledge graph.

## Overview

This API provides HTTP endpoints that wrap the `ResearchQueryEngine` methods, exposing all 5 research queries as POST endpoints with JSON request/response.

**Validates: Requirements 1.1, 1.2, 1.3**

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Running the API

### Development Mode

```bash
# Run with auto-reload
uvicorn api.query_api:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python -m api.query_api
```

### Production Mode

```bash
# Run with multiple workers
uvicorn api.query_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Root Endpoints

#### GET /
Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

#### GET /health
Health check endpoint to verify API and Neo4j connectivity.

```bash
curl http://localhost:8000/health
```

### Query Endpoints

All query endpoints use POST with JSON request bodies.

#### 1. Cross-Study Associations

**POST /query/cross-study-associations**

Find taxa with consistent disease associations across multiple studies.

**Request:**
```json
{
  "disease": "Type 2 Diabetes",
  "study_type": "RCT",
  "min_papers": 3,
  "confidence_threshold": 0.7,
  "require_open_data": true
}
```

**Parameters:**
- `disease` (string, required): Disease entity name
- `study_type` (string, default: "RCT"): One of "RCT", "observational", "meta_analysis", "any"
- `min_papers` (int, default: 3): Minimum number of papers required
- `confidence_threshold` (float, default: 0.7): Minimum confidence score (0.0-1.0)
- `require_open_data` (bool, default: true): Only include papers with open data

**Example:**
```bash
curl -X POST http://localhost:8000/query/cross-study-associations \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Type 2 Diabetes",
    "study_type": "RCT",
    "min_papers": 3,
    "confidence_threshold": 0.7,
    "require_open_data": true
  }'
```

#### 2. Intervention Evidence

**POST /query/intervention-evidence**

Find interventions with RCT-level evidence for modifying specific taxa.

**Request:**
```json
{
  "intervention_types": ["probiotic", "FMT"],
  "min_sample_size": 50,
  "evidence_strength": "strong"
}
```

**Parameters:**
- `intervention_types` (array of strings, required): List of intervention types
- `min_sample_size` (int, default: 50): Minimum total sample size
- `evidence_strength` (string, default: "strong"): One of "strong", "moderate", "weak", "any"

**Example:**
```bash
curl -X POST http://localhost:8000/query/intervention-evidence \
  -H "Content-Type: application/json" \
  -d '{
    "intervention_types": ["probiotic", "FMT"],
    "min_sample_size": 50,
    "evidence_strength": "strong"
  }'
```

#### 3. Methodology Landscape

**POST /query/methodology-landscape**

Survey data availability and methodology trends over time.

**Request:**
```json
{
  "year_start": 2020,
  "year_end": 2024,
  "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
  "require_deposited_data": true
}
```

**Parameters:**
- `year_start` (int, required): Start year (inclusive)
- `year_end` (int, required): End year (inclusive)
- `sequencing_methods` (array of strings, required): List of sequencing methods
- `require_deposited_data` (bool, default: true): Only include papers with deposited data

**Example:**
```bash
curl -X POST http://localhost:8000/query/methodology-landscape \
  -H "Content-Type: application/json" \
  -d '{
    "year_start": 2020,
    "year_end": 2024,
    "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
    "require_deposited_data": true
  }'
```

#### 4. Top Associations

**POST /query/top-associations**

Find top taxa associated with a disease ranked by evidence quality.

**Request:**
```json
{
  "disease": "IBD",
  "top_n": 10,
  "min_confidence": 0.7
}
```

**Parameters:**
- `disease` (string, required): Disease entity name
- `top_n` (int, default: 10): Maximum number of taxa to return
- `min_confidence` (float, default: 0.7): Minimum confidence score (0.0-1.0)

**Example:**
```bash
curl -X POST http://localhost:8000/query/top-associations \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "IBD",
    "top_n": 10,
    "min_confidence": 0.7
  }'
```

#### 5. Conflicting Evidence

**POST /query/conflicting-evidence**

Find taxa with conflicting associations (increased vs decreased).

**Request:**
```json
{
  "disease": "Crohn's Disease",
  "min_papers_per_direction": 2
}
```

**Parameters:**
- `disease` (string, required): Disease entity name
- `min_papers_per_direction` (int, default: 2): Minimum papers required for each direction

**Example:**
```bash
curl -X POST http://localhost:8000/query/conflicting-evidence \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Crohns Disease",
    "min_papers_per_direction": 2
  }'
```

### Cache Management Endpoints

#### POST /cache/invalidate

Invalidate all cached query results. Call this after loading new data into the graph.

```bash
curl -X POST http://localhost:8000/cache/invalidate
```

#### GET /cache/stats

Get cache statistics including hit rate and size.

```bash
curl http://localhost:8000/cache/stats
```

## Response Format

All query endpoints return a `QueryResponse` with the following structure:

```json
{
  "success": true,
  "query_result": {
    "query_id": "550e8400-e29b-41d4-a716-446655440000",
    "query_description": "Cross-study associations for Type 2 Diabetes",
    "results": [
      {
        "taxon_name": "Bacteroides fragilis",
        "paper_count": 5,
        "consensus_confidence": 0.85,
        "direction": "increased"
      }
    ],
    "result_count": 1,
    "execution_time_ms": 245.3,
    "executed_at": "2024-01-15T10:30:00.000Z",
    "aggregation_method": "weighted_average",
    "confidence_threshold": 0.7,
    "timeout": false,
    "error": null
  },
  "error": null
}
```

**Fields:**
- `success` (bool): Whether the query executed successfully
- `query_result` (object): Query result with data and metadata
  - `query_id` (string): Unique identifier for this query execution
  - `query_description` (string): Human-readable description
  - `results` (array): List of result records
  - `result_count` (int): Number of results returned
  - `execution_time_ms` (float): Query execution time in milliseconds
  - `executed_at` (string): ISO timestamp
  - `aggregation_method` (string, optional): Evidence aggregation method
  - `confidence_threshold` (float, optional): Confidence threshold applied
  - `timeout` (bool): Whether the query timed out
  - `error` (string, optional): Error message if query failed
- `error` (string, optional): API-level error message

## Error Handling

The API returns standard HTTP status codes:

- **200 OK**: Query executed successfully
- **400 Bad Request**: Invalid request parameters
- **422 Unprocessable Entity**: Request validation failed
- **500 Internal Server Error**: Query execution failed
- **503 Service Unavailable**: Query engine not initialized or Neo4j connection failed

Error responses include a `detail` field with the error message:

```json
{
  "detail": "Parameter 'study_type' must be one of ['RCT', 'observational', 'meta_analysis', 'any']"
}
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest api/test_query_api.py -v

# Run specific test class
pytest api/test_query_api.py::TestCrossStudyAssociationsEndpoint -v

# Run with coverage
pytest api/test_query_api.py --cov=api --cov-report=html
```

## Security Features

The API implements several security features (Requirement 18):

1. **Input Validation**: All request parameters are validated using Pydantic models
2. **Parameterized Queries**: All Cypher queries use parameters to prevent injection attacks
3. **Type Checking**: Strong type validation for all inputs
4. **Error Handling**: Graceful error handling with informative messages
5. **Logging**: All requests and errors are logged for audit trail

## Performance

- **Query Caching**: Results are cached for 24 hours (Requirement 13.5)
- **Connection Pooling**: Neo4j driver uses connection pooling
- **Async Support**: FastAPI provides async request handling
- **Multiple Workers**: Can run with multiple Uvicorn workers for production

## Integration Example

Python client example:

```python
import requests

# Query cross-study associations
response = requests.post(
    "http://localhost:8000/query/cross-study-associations",
    json={
        "disease": "Type 2 Diabetes",
        "study_type": "RCT",
        "min_papers": 3,
        "confidence_threshold": 0.7,
        "require_open_data": True
    }
)

if response.status_code == 200:
    data = response.json()
    if data["success"]:
        results = data["query_result"]["results"]
        for result in results:
            print(f"Taxon: {result['taxon_name']}")
            print(f"Papers: {result['paper_count']}")
            print(f"Confidence: {result['consensus_confidence']}")
    else:
        print(f"Query failed: {data['error']}")
else:
    print(f"HTTP error: {response.status_code}")
```

## Deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.query_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t knowledge-graph-api .
docker run -p 8000:8000 --env-file .env knowledge-graph-api
```

### Docker Compose

Add to `docker-compose.yml`:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - neo4j
```

## Monitoring

The API provides several monitoring endpoints:

- `/health`: Health check with Neo4j connectivity status
- `/cache/stats`: Cache performance metrics
- Logs: All requests and errors are logged using loguru

## License

See project LICENSE file.
