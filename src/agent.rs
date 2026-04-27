use anyhow::Result;
use async_openai::{
    Client, config::OpenAIConfig,
    types::{CreateCompletionRequestArgs},
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::Config;
use crate::memory::{MemoryManager, PersistentStore};
use crate::tools::ToolManager;

pub struct Agent {
    config: Config,
    memory: Arc<Mutex<MemoryManager>>,
    tools: Arc<Mutex<ToolManager>>,
    client: Client<OpenAIConfig>,
    session_id: String,
}

impl Agent {
    pub fn new(
        config: Config,
        memory: Arc<Mutex<MemoryManager>>,
        tools: Arc<Mutex<ToolManager>>,
        session_id: String,
    ) -> Self {
        let api_key = config.api_key.clone().unwrap_or_else(|| "empty".to_string());
        let base_url = config.base_url.clone().unwrap_or_else(|| "http://localhost:8080/v1".to_string());

        let client_config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url);

        let client = Client::with_config(client_config);

        Self { config, memory, tools, client, session_id }
    }

    pub async fn run_loop(&mut self, initial_prompt: &str) -> Result<()> {
        let model = self.config.model_name.clone().unwrap_or_else(|| "QwQ-32B".to_string());

        let mut prompt = {
            let mem = self.memory.lock().await;
            if let Some(store) = mem.load_memory(&self.session_id)? {
                format!("{}\n{}", store.full_context, initial_prompt)
            } else {
                initial_prompt.to_string()
            }
        };

        let mut loop_count = 0;
        let max_loops = 50;

        info!("Starting reasoning loop for session {}", self.session_id);

        while loop_count < max_loops {
            loop_count += 1;
            info!("Loop {}", loop_count);

            let request = CreateCompletionRequestArgs::default()
                .model(&model)
                .prompt(&prompt)
                .max_tokens(4000_u16)
                .temperature(0.7)
                .top_p(0.8)
                .stop(vec![
                    "[END_TOOL_CALL]".to_string(),
                    "[END_TOOL_SEARCH]".to_string(),
                    "[FOLD_THOUGHT]".to_string()
                ])
                .build()?;

            let response = self.client.completions().create(request).await?;
            let output_text = response.choices.first()
                .map(|c| c.text.clone())
                .unwrap_or_default();

            prompt.push_str(&output_text);

            if output_text.contains("[BEGIN_TOOL_CALL]") {
                if let Some(start) = output_text.find("[BEGIN_TOOL_CALL]") {
                    let call_str = &output_text[start + "[BEGIN_TOOL_CALL]".len()..];

                    if let Ok(args) = serde_json::from_str::<Value>(call_str.trim()) {
                        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let fn_args = args.get("arguments").cloned().unwrap_or(Value::Null);

                        info!("Calling tool: {}", name);
                        let tools = self.tools.lock().await;
                        let result = tools.call_tool(name, fn_args).await;

                        let resp_str = match result {
                            Ok(v) => serde_json::to_string(&v)?,
                            Err(e) => format!("{{\"error\": \"{}\"}}", e),
                        };
                        prompt.push_str(&format!("\n[BEGIN_TOOL_RESPONSE]{}[END_TOOL_RESPONSE]\n", resp_str));
                    } else {
                        warn!("Failed to parse tool call JSON");
                        prompt.push_str("\n[BEGIN_TOOL_RESPONSE]{\"error\": \"invalid JSON\"}[END_TOOL_RESPONSE]\n");
                    }
                }
            } else if output_text.contains("[FOLD_THOUGHT]") {
                info!("Folding thought. Saving state.");
                let mut store = PersistentStore::default();
                store.full_context = prompt.clone();
                let mem = self.memory.lock().await;
                mem.save_memory(&self.session_id, &store)?;
            } else {
                info!("Generation finished or no tool markers found.");
                break;
            }
        }

        info!("Saving final context.");
        let mut store = PersistentStore::default();
        store.full_context = prompt.clone();
        let mem = self.memory.lock().await;
        mem.save_memory(&self.session_id, &store)?;

        Ok(())
    }
}
