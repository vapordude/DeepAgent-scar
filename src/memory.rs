use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PersistentStore {
    pub episodic_memory: String,
    pub working_memory: String,
    pub tool_memory: String,
    pub full_context: String,
}

pub struct MemoryManager {
    db: sled::Db,
}

impl MemoryManager {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn save_memory(&self, session_id: &str, store: &PersistentStore) -> Result<()> {
        let serialized = serde_json::to_vec(store)?;
        self.db.insert(session_id, serialized)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn load_memory(&self, session_id: &str) -> Result<Option<PersistentStore>> {
        if let Some(data) = self.db.get(session_id)? {
            let store: PersistentStore = serde_json::from_slice(&data)?;
            Ok(Some(store))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_save_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().to_str().unwrap();

        let manager = MemoryManager::new(db_path).unwrap();
        let session = "test_session_1";

        let mut store = PersistentStore::default();
        store.working_memory = "Thinking about how to rewrite to rust.".to_string();

        manager.save_memory(session, &store).unwrap();

        let loaded = manager.load_memory(session).unwrap().unwrap();
        assert_eq!(loaded.working_memory, "Thinking about how to rewrite to rust.");
    }
}
