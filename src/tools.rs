use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use rhai::{Engine, Scope, Dynamic};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use scraper::{Html, Selector};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, args: Value) -> Result<Value>;
}

pub struct WebSearch {
    client: Client,
    api_key: String,
}

impl WebSearch {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web using Google Serper API. Args: {\"query\": \"search text\"}"
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'query' string in args"))?;

        let payload = serde_json::json!({ "q": query });

        let res = self.client.post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            return Err(anyhow!("Serper API returned error: {}", res.status()));
        }

        let body: Value = res.json().await?;
        Ok(body)
    }
}

pub struct WebPageReader {
    client: Client,
}

impl WebPageReader {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }
}

#[async_trait]
impl Tool for WebPageReader {
    fn name(&self) -> &str {
        "web_page_reader"
    }

    fn description(&self) -> &str {
        "Reads a web page and extracts its textual content for reasoning-in-documents. Args: {\"url\": \"https://example.com\"}"
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let url = args.get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'url' string in args"))?;

        let res = self.client.get(url).send().await?;

        if !res.status().is_success() {
            return Err(anyhow!("Failed to fetch URL: HTTP {}", res.status()));
        }

        let html = res.text().await?;
        let document = Html::parse_document(&html);

        let selector = Selector::parse("body").unwrap();
        let body = document.select(&selector).next().map(|el| el.text().collect::<Vec<_>>().join(" "));

        let content = body.unwrap_or_else(|| "No content found".to_string());

        // Truncate to avoid exploding context window
        let max_chars = 8000;
        let content = if content.chars().count() > max_chars {
            let truncated: String = content.chars().take(max_chars).collect();
            format!("{}... (truncated)", truncated)
        } else {
            content
        };

        Ok(serde_json::json!({ "content": content }))
    }
}

pub struct RhaiTool {
    name: String,
    description: String,
    engine: Arc<Mutex<Engine>>,
    script: String,
}

impl RhaiTool {
    pub fn new(name: String, description: String, script: String) -> Self {
        let engine = Engine::new();
        // Here we could register extra Rust functions into the engine
        // so the script can do complex stuff (like HTTP requests).
        // For now we just use a basic engine.

        Self {
            name,
            description,
            engine: Arc::new(Mutex::new(engine)),
            script,
        }
    }
}

#[async_trait]
impl Tool for RhaiTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let engine = self.engine.lock().await;
        let mut scope = Scope::new();

        // Pass JSON string to rhai script
        let args_str = serde_json::to_string(&args)?;
        scope.push("args_json", args_str);

        let result: Dynamic = engine.eval_with_scope(&mut scope, &self.script)
            .map_err(|e| anyhow!("Rhai execution error: {}", e))?;

        let result_str = result.to_string();

        // Attempt to parse result back as JSON
        if let Ok(v) = serde_json::from_str::<Value>(&result_str) {
            Ok(v)
        } else {
            Ok(Value::String(result_str))
        }
    }
}

#[derive(Default)]
pub struct ToolManager {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    #[allow(dead_code)]
    pub fn get_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn search_tools(&self, query: &str) -> Vec<Value> {
        let query = query.to_lowercase();
        self.tools.values()
            .filter(|tool| {
                tool.name().to_lowercase().contains(&query)
                || tool.description().to_lowercase().contains(&query)
            })
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description()
                })
            })
            .collect()
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        if let Some(tool) = self.tools.get(name) {
            tool.call(args).await
        } else {
            Err(anyhow!("Tool '{}' not found", name))
        }
    }
}
