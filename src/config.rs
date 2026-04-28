use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Clone, Default)]
#[allow(dead_code)]
pub struct Config {
    // API Config
    pub toolbench_api: Option<String>,
    pub toolbench_service_url: Option<String>,
    pub google_serper_api: Option<String>,
    pub use_jina: Option<bool>,
    pub jina_api_key: Option<String>,
    pub tmdb_access_token: Option<String>,
    pub spotipy_client_id: Option<String>,
    pub spotipy_client_secret: Option<String>,
    pub spotipy_redirect_uri: Option<String>,
    pub webshop_service_url: Option<String>,

    // Model Config
    pub model_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub tokenizer_path: Option<String>,

    pub aux_model_name: Option<String>,
    pub aux_base_url: Option<String>,
    pub aux_api_key: Option<String>,
    pub aux_tokenizer_path: Option<String>,

    pub vqa_model_name: Option<String>,
    pub vqa_base_url: Option<String>,
    pub vqa_api_key: Option<String>,

    pub tool_retriever_model_path: Option<String>,
    pub tool_retriever_api_base: Option<String>,

    // Data Path Config (omitted full list for brevity, parsing gracefully via Option)
    pub toolbench_data_path: Option<String>,
    pub toolbench_toolset_path: Option<String>,
    pub toolbench_corpus_tsv_path: Option<String>,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config() {
        let yaml = r#"
model_name: "QwQ-32B"
base_url: "http://localhost:8080/v1"
api_key: "empty"
use_jina: true
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model_name.unwrap(), "QwQ-32B");
        assert_eq!(config.base_url.unwrap(), "http://localhost:8080/v1");
        assert_eq!(config.api_key.unwrap(), "empty");
        assert_eq!(config.use_jina.unwrap(), true);
    }
}
