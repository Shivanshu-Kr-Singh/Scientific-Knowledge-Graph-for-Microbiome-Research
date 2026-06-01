# Neo4j Rollback Mechanism Guide

## Overview

This guide explains how to use the rollback mechanism for the Neo4j database migration. The rollback mechanism ensures safe migration from the old flat relationship system to the new enhanced semantic knowledge graph system by maintaining backups and providing restoration capabilities.

**Requirements:** 16.4 - Maintain old Neo4j database instance as rollback option until migration is validated

## Architecture

The rollback mechanism consists of three main components:

1. **Backup Script** (`backup_neo4j.py`) - Creates snapshots of the Neo4j database
2. **Rollback Script** (`rollback_neo4j.py`) - Restores database from backups
3. **Dual Database Configuration** - Maintains both old and new Neo4j instances during migration

## Quick Start

### 1. Create a Backup Before Migration

Before running any migration, create a backup of your current Neo4j database:

```bash
# Using default settings (reads from .env)
python scripts/backup_neo4j.py

# With custom settings
python scripts/backup_neo4j.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password your_password \
  --backup-dir data/backups
```

This will create:
- A Cypher export file with all nodes and relationships
- A metadata JSON file with backup statistics
- A `latest_backup.json` file pointing to the most recent backup

### 2. Run Migration

After creating a backup, run the migration to the enhanced schema:

```bash
python scripts/migrate_to_enhanced_schema.py \
  --old-uri bolt://localhost:7687 \
  --new-uri bolt://localhost:7688
```

### 3. Validate Migration

Test the new system thoroughly:
- Run all five research queries
- Verify data integrity
- Check performance metrics
- Validate provenance tracking

### 4. Rollback if Needed

If issues are discovered, rollback to the backup:

```bash
# Rollback to latest backup (with confirmation prompt)
python scripts/rollback_neo4j.py

# Rollback to specific backup
python scripts/rollback_neo4j.py --backup neo4j_backup_20260527_103645

# Dry run (verify backup without restoring)
python scripts/rollback_neo4j.py --dry-run

# Force rollback without confirmation (use with caution!)
python scripts/rollback_neo4j.py --force
```

## Detailed Usage

### Backup Script (`backup_neo4j.py`)

#### Create a Backup

```bash
# Basic backup
python scripts/backup_neo4j.py

# Named backup
python scripts/backup_neo4j.py --name pre_migration_backup

# Custom backup directory
python scripts/backup_neo4j.py --backup-dir /path/to/backups
```

#### List Available Backups

```bash
python scripts/backup_neo4j.py --list
```

Output:
```
Found 3 backup(s):
  - neo4j_backup_20260527_143022 (2026-05-27T14:30:22.123456+00:00)
    Nodes: 1250, Relationships: 3420
  - neo4j_backup_20260527_103645 (2026-05-27T10:36:45.789012+00:00)
    Nodes: 1200, Relationships: 3300
  - pre_migration_backup (2026-05-26T15:20:10.456789+00:00)
    Nodes: 1150, Relationships: 3200
```

#### Verify a Backup

```bash
python scripts/backup_neo4j.py --verify neo4j_backup_20260527_103645
```

### Rollback Script (`rollback_neo4j.py`)

#### List Available Backups

```bash
python scripts/rollback_neo4j.py --list
```

#### Rollback to Latest Backup

```bash
# With confirmation prompt
python scripts/rollback_neo4j.py

# Without confirmation (automated scripts)
python scripts/rollback_neo4j.py --force
```

#### Rollback to Specific Backup

```bash
python scripts/rollback_neo4j.py --backup neo4j_backup_20260527_103645
```

#### Dry Run (Verify Without Restoring)

```bash
python scripts/rollback_neo4j.py --dry-run
```

This will:
- Verify the backup is valid
- Check all required files exist
- NOT modify the database

#### Skip Verification (Not Recommended)

```bash
python scripts/rollback_neo4j.py --skip-verification
```

**Warning:** Only use this if you're certain the backup is valid.

## Configuration

### Environment Variables

The scripts read configuration from environment variables (typically in `.env`):

```bash
# Old database (to be backed up)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# New database (for enhanced schema)
NEO4J_NEW_URI=bolt://localhost:7688
NEO4J_NEW_USER=neo4j
NEO4J_NEW_PASSWORD=your_password
```

### Dual Database Setup

During migration, maintain two Neo4j instances:

1. **Old Database** (port 7687) - Original flat schema
2. **New Database** (port 7688) - Enhanced semantic schema

#### Docker Compose Example

```yaml
version: '3.8'

services:
  neo4j-old:
    image: neo4j:5.13
    container_name: neo4j-old
    ports:
      - "7687:7687"
      - "7474:7474"
    environment:
      - NEO4J_AUTH=neo4j/your_password
    volumes:
      - neo4j-old-data:/data

  neo4j-new:
    image: neo4j:5.13
    container_name: neo4j-new
    ports:
      - "7688:7687"
      - "7475:7474"
    environment:
      - NEO4J_AUTH=neo4j/your_password
    volumes:
      - neo4j-new-data:/data

volumes:
  neo4j-old-data:
  neo4j-new-data:
```

Start both instances:

```bash
docker-compose up -d
```

## Backup Format

### Directory Structure

```
data/backups/
├── neo4j_backup_20260527_103645_export.cypher
├── neo4j_backup_20260527_103645_metadata.json
├── neo4j_backup_20260527_143022_export.cypher
├── neo4j_backup_20260527_143022_metadata.json
├── latest_backup.json
└── rollback_20260527_150000.json
```

### Metadata File Format

```json
{
  "backup_name": "neo4j_backup_20260527_103645",
  "timestamp": "2026-05-27T10:36:45.789012+00:00",
  "neo4j_uri": "bolt://localhost:7687",
  "statistics": {
    "nodes": {
      "Paper": 450,
      "Taxon": 320,
      "Disease": 180,
      "Method": 300
    },
    "relationships": {
      "HAS_TAXON": 1200,
      "HAS_DISEASE": 800,
      "USES_METHOD": 1420
    },
    "total_nodes": 1250,
    "total_relationships": 3420
  },
  "cypher_file": "neo4j_backup_20260527_103645_export.cypher",
  "backup_method": "cypher_export"
}
```

### Cypher Export Format

The Cypher export file contains:

```cypher
// Neo4j Database Backup
// Created: 2026-05-27T10:36:45.789012+00:00
// Source: bolt://localhost:7687

// ========== NODES ==========

// Paper nodes
CREATE (:Paper {id: "10.1234/example", title: "Example Paper", year: 2024});
CREATE (:Paper {id: "10.5678/another", title: "Another Paper", year: 2025});

// Taxon nodes
CREATE (:Taxon {id: "Bacteroides fragilis", name: "Bacteroides fragilis"});

// ========== RELATIONSHIPS ==========

// HAS_TAXON relationships
MATCH (source:Paper {id: "10.1234/example"}), (target:Taxon {id: "Bacteroides fragilis"}) 
CREATE (source)-[:HAS_TAXON {confidence: 0.85}]->(target);
```

## Best Practices

### 1. Always Backup Before Migration

```bash
# Create a named backup before major changes
python scripts/backup_neo4j.py --name pre_migration_$(date +%Y%m%d)
```

### 2. Verify Backups Immediately

```bash
# Verify the backup was created successfully
python scripts/backup_neo4j.py --verify neo4j_backup_20260527_103645
```

### 3. Test Rollback in Development

Before relying on rollback in production, test it in a development environment:

```bash
# 1. Create backup
python scripts/backup_neo4j.py --name test_backup

# 2. Make some changes to the database
# ... (modify data) ...

# 3. Test rollback
python scripts/rollback_neo4j.py --backup test_backup --force

# 4. Verify restoration
python scripts/backup_neo4j.py --list
```

### 4. Maintain Multiple Backups

Keep backups at different stages:

```bash
# Before migration
python scripts/backup_neo4j.py --name pre_migration

# After initial migration
python scripts/backup_neo4j.py --name post_initial_migration

# After validation
python scripts/backup_neo4j.py --name post_validation
```

### 5. Monitor Backup Size

Large databases may produce large backup files. Monitor disk space:

```bash
# Check backup directory size
du -sh data/backups/

# Remove old backups if needed
rm data/backups/neo4j_backup_20260401_*
```

### 6. Automate Backups

Add to your migration workflow:

```bash
#!/bin/bash
# migration_workflow.sh

set -e  # Exit on error

echo "Step 1: Creating backup..."
python scripts/backup_neo4j.py --name pre_migration_$(date +%Y%m%d_%H%M%S)

echo "Step 2: Running migration..."
python scripts/migrate_to_enhanced_schema.py

echo "Step 3: Validating migration..."
# Run validation tests here

echo "Migration completed successfully!"
```

## Troubleshooting

### Backup Fails with Connection Error

**Problem:** Cannot connect to Neo4j database

**Solution:**
1. Verify Neo4j is running: `docker ps` or `systemctl status neo4j`
2. Check connection settings in `.env`
3. Test connection: `cypher-shell -a bolt://localhost:7687 -u neo4j -p your_password`

### Backup File is Too Large

**Problem:** Cypher export file is very large (>1GB)

**Solution:**
1. Consider using Neo4j's native backup tools for large databases
2. Compress backup files: `gzip data/backups/*.cypher`
3. Use incremental backups for frequent snapshots

### Rollback Takes Too Long

**Problem:** Restoration is slow for large databases

**Solution:**
1. Increase batch size in the script (modify `batch_size` variable)
2. Use Neo4j's native restore tools for large databases
3. Consider using database snapshots instead of Cypher exports

### Rollback Verification Fails

**Problem:** Restored statistics don't match backup

**Solution:**
1. Check for errors in the rollback log
2. Verify the Cypher file is not corrupted
3. Try restoring to a fresh database instance
4. Contact support if issue persists

### Out of Disk Space

**Problem:** Backup fails due to insufficient disk space

**Solution:**
1. Clean up old backups: `rm data/backups/neo4j_backup_202604*`
2. Move backups to external storage
3. Use compression: `gzip data/backups/*.cypher`

## Migration Workflow

### Complete Migration Process

```bash
# 1. Pre-migration backup
echo "Creating pre-migration backup..."
python scripts/backup_neo4j.py --name pre_migration

# 2. Verify backup
echo "Verifying backup..."
python scripts/backup_neo4j.py --verify pre_migration

# 3. Run migration
echo "Running migration..."
python scripts/migrate_to_enhanced_schema.py \
  --old-uri bolt://localhost:7687 \
  --new-uri bolt://localhost:7688

# 4. Validate new system
echo "Validating new system..."
# Run your validation tests here
# python scripts/validate_migration.py

# 5. If validation fails, rollback
if [ $? -ne 0 ]; then
    echo "Validation failed! Rolling back..."
    python scripts/rollback_neo4j.py --backup pre_migration --force
    exit 1
fi

# 6. If validation passes, create post-migration backup
echo "Creating post-migration backup..."
python scripts/backup_neo4j.py --name post_migration

echo "Migration completed successfully!"
```

### Decommissioning Old System

**Only after migration is fully validated:**

```bash
# 1. Create final backup of old system
python scripts/backup_neo4j.py --name final_old_system_backup

# 2. Verify backup
python scripts/backup_neo4j.py --verify final_old_system_backup

# 3. Stop old Neo4j instance
docker stop neo4j-old

# 4. Archive old data
tar -czf neo4j-old-archive-$(date +%Y%m%d).tar.gz data/backups/

# 5. Move archive to long-term storage
mv neo4j-old-archive-*.tar.gz /path/to/archive/
```

**Requirements:** Per Requirement 16.5, do NOT decommission the old system until all five research queries are successfully completed and validated on production data.

## Security Considerations

### 1. Protect Backup Files

Backups contain sensitive data. Secure them appropriately:

```bash
# Set restrictive permissions
chmod 600 data/backups/*.cypher
chmod 600 data/backups/*.json

# Encrypt backups for long-term storage
gpg --encrypt --recipient your-key data/backups/neo4j_backup_*.cypher
```

### 2. Secure Database Credentials

Never commit credentials to version control:

```bash
# Add to .gitignore
echo "data/backups/" >> .gitignore
echo ".env" >> .gitignore
```

### 3. Audit Rollback Operations

All rollback operations are logged:

```bash
# View rollback history
cat data/backups/rollback_*.json
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the migration logs in `data/migration_report.json`
3. Check Neo4j logs: `docker logs neo4j-old` or `docker logs neo4j-new`
4. Consult the main migration documentation: `scripts/MIGRATION_README.md`

## References

- **Requirements:** 16.4 - Maintain old Neo4j database instance as rollback option
- **Migration Script:** `scripts/migrate_to_enhanced_schema.py`
- **Migration Guide:** `scripts/MIGRATION_README.md`
- **Neo4j Documentation:** https://neo4j.com/docs/
