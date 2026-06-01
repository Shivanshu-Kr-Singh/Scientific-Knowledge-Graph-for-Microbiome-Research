# Neo4j Rollback Mechanism - Quick Reference

## Quick Commands

### Backup Operations

```bash
# Create backup (auto-named with timestamp)
python scripts/backup_neo4j.py

# Create named backup
python scripts/backup_neo4j.py --name pre_migration

# List all backups
python scripts/backup_neo4j.py --list

# Verify a backup
python scripts/backup_neo4j.py --verify backup_name
```

### Rollback Operations

```bash
# Rollback to latest backup (with confirmation)
python scripts/rollback_neo4j.py

# Rollback to specific backup
python scripts/rollback_neo4j.py --backup backup_name

# Dry run (test without modifying database)
python scripts/rollback_neo4j.py --dry-run

# Force rollback (no confirmation - use with caution!)
python scripts/rollback_neo4j.py --force

# List available backups
python scripts/rollback_neo4j.py --list
```

## Migration Workflow

```bash
# 1. Backup
python scripts/backup_neo4j.py --name pre_migration

# 2. Verify
python scripts/backup_neo4j.py --verify pre_migration

# 3. Migrate
python scripts/migrate_to_enhanced_schema.py

# 4. If issues occur, rollback
python scripts/rollback_neo4j.py --backup pre_migration
```

## Environment Variables

```bash
# Old database (for rollback)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# New database (enhanced schema)
NEO4J_NEW_URI=bolt://localhost:7688
NEO4J_NEW_USER=neo4j
NEO4J_NEW_PASSWORD=your_password
```

## Docker Setup

```bash
# Start dual Neo4j instances
docker-compose -f docker-compose.neo4j-dual.yml up -d

# Check status
docker-compose -f docker-compose.neo4j-dual.yml ps

# Stop instances
docker-compose -f docker-compose.neo4j-dual.yml down
```

## Access URLs

- **Old database browser:** http://localhost:7474
- **New database browser:** http://localhost:7475
- **Old database bolt:** bolt://localhost:7687
- **New database bolt:** bolt://localhost:7688

## Backup Directory Structure

```
data/backups/
├── {backup_name}_export.cypher      # Cypher statements
├── {backup_name}_metadata.json      # Backup info
├── latest_backup.json               # Latest backup reference
└── rollback_{timestamp}.json        # Rollback reports
```

## Common Issues

### Connection Error
```bash
# Check Neo4j is running
docker ps

# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p your_password
```

### Backup Too Large
```bash
# Compress backups
gzip data/backups/*.cypher

# Clean old backups
rm data/backups/backup_202604*
```

### Rollback Verification Failed
```bash
# Check backup integrity
python scripts/backup_neo4j.py --verify backup_name

# Try dry run first
python scripts/rollback_neo4j.py --backup backup_name --dry-run
```

## Safety Checklist

- [ ] Backup created before migration
- [ ] Backup verified successfully
- [ ] Both Neo4j instances running
- [ ] Environment variables configured
- [ ] Migration tested in development
- [ ] Rollback tested in development
- [ ] Validation tests ready

## Emergency Rollback

If migration fails and immediate rollback is needed:

```bash
# 1. Stop migration if running
# (Ctrl+C or kill process)

# 2. Rollback immediately
python scripts/rollback_neo4j.py --force

# 3. Verify restoration
python scripts/backup_neo4j.py --list
```

## Support

- **Full Documentation:** `scripts/ROLLBACK_GUIDE.md`
- **Task Summary:** `scripts/TASK_7.2_COMPLETION_SUMMARY.md`
- **Migration Guide:** `scripts/MIGRATION_README.md`
- **Tests:** `scripts/test_backup_rollback.py`

## Requirements

**Requirement 16.4:** Maintain old Neo4j database instance as rollback option until migration is validated

✅ Implemented with backup/rollback scripts and dual database configuration
