mod env;
mod agent;
mod config;
mod memory;
mod tools;

use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

use crate::agent::Agent;
use crate::config::Config;
use crate::memory::MemoryManager;
use crate::tools::{ToolManager, WebSearch, RhaiTool, PythonExecutor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "./config/base_config.yaml")]
    config_path: String,

    #[arg(short, long, default_value = "What is 2+2?")]
    query: String,

    #[arg(short, long, default_value = "default_session")]
    session_id: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    let args = Args::parse();
    info!("Starting DeepAgent RS with session: {}", args.session_id);

    // Load Config
    let config = Config::load(&args.config_path).unwrap_or_else(|e| {
        error!("Failed to load config: {}, using defaults", e);
        Config::default()
    });

    // Initialize Memory Manager
    let memory = MemoryManager::new("./data/agent_memory")?;
    let memory_arc = Arc::new(Mutex::new(memory));

    // Initialize Tool Manager and register tools
    let mut tools = ToolManager::new();

    if let Some(serper_key) = &config.google_serper_api {
        let web_search = Box::new(WebSearch::new(serper_key.clone()));
        tools.register(web_search);
    }

    // Demonstrate "Evolving and Self-Modifying" via script tool
    let dynamic_script = r#"
        // Parse incoming args
        let input = parse_json(args_json);
        let a = input.a;
        let b = input.b;
        let c = a + b;
        `{"result": ${c}}`
    "#.to_string();

    let rhai_tool = Box::new(RhaiTool::new(
        "math_add".to_string(),
        "Adds two numbers. Args: {\"a\": 1, \"b\": 2}".to_string(),
        dynamic_script
    ));
    tools.register(rhai_tool);

    // Register Python Executor
    let python_executor = Box::new(PythonExecutor);
    tools.register(python_executor);

    let tools_arc = Arc::new(Mutex::new(tools));

    // Create Agent
    let mut agent = Agent::new(config, memory_arc, tools_arc, args.session_id);

    // Run reasoning loop
    info!("Executing query: {}", args.query);
    if let Err(e) = agent.run_loop(&args.query).await {
        error!("Agent execution failed: {:?}", e);
    } else {
        info!("Agent execution completed successfully.");
    }

    Ok(())
}
