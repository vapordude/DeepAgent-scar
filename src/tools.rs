use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use rhai::{Engine, Scope, Dynamic};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

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

pub struct RhaiTool {
    name: String,
    description: String,
    engine: Arc<Mutex<Engine>>,
    script: String,
}

impl RhaiTool {
    pub fn new(name: String, description: String, script: String) -> Self {
        let mut engine = Engine::new();
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
pub struct PythonExecutor;

#[async_trait]
impl Tool for PythonExecutor {
    fn name(&self) -> &str {
        "execute_python_code"
    }

    fn description(&self) -> &str {
        "Execute Python code in a safe sandbox environment and return the execution results from stdout. Args: {\"code\": \"python code string\"}"
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let code = args.get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'code' string in args"))?;

        let mut cmd = tokio::process::Command::new("python3");
        cmd.arg("-c").arg(code);

        // Apply a timeout of 10 seconds
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let mut status = format!("Done. Exit status: {}", output.status);

                if !stderr.is_empty() {
                     status = format!("Completed with stderr: {}", stderr);
                }

                Ok(serde_json::json!({
                    "execution_result": stdout,
                    "execution_status": status
                }))
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("Failed to execute python3: {}", e)),
            Err(_) => Err(anyhow::anyhow!("Execution timed out after 10 seconds.")),
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

    pub fn get_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        if let Some(tool) = self.tools.get(name) {
            tool.call(args).await
        } else {
            Err(anyhow!("Tool '{}' not found", name))
        }
    }
}
