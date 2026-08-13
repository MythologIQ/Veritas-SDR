//! Runtime initialization and IPC server setup for GG-CORE.

use gg_core::cli::get_socket_path;
use gg_core::config as gg_config;
use gg_core::engine::InferenceParams;
use gg_core::ipc::server;
use gg_core::models::load_model_dispatch;
use gg_core::shutdown::ShutdownResult;
use gg_core::{Runtime, RuntimeConfig};

use gg_core::cli::CliIpcClient;

/// One model requested for preload when the standalone daemon starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupModelSpec {
    pub path: String,
    pub model_id: Option<String>,
}

/// Extract the startup model arguments from `gg-core-cli serve`.
///
/// This first #106 slice deliberately supports one startup model. The dynamic
/// load/unload IPC work can reuse the same canonical loader/lifecycle seam.
pub fn parse_startup_model(args: &[String]) -> Result<Option<StartupModelSpec>, String> {
    let mut path = None;
    let mut model_id = None;
    let mut i = 2;

    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --model".to_string())?;
                if path.is_some() {
                    return Err("Only one startup --model is supported in this release".into());
                }
                path = Some(value.clone());
                i += 2;
            }
            "--model-id" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing value for --model-id".to_string())?;
                if value.trim().is_empty() {
                    return Err("--model-id cannot be empty".into());
                }
                model_id = Some(value.clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    if model_id.is_some() && path.is_none() {
        return Err("--model-id requires --model <PATH>".into());
    }

    Ok(path.map(|path| StartupModelSpec { path, model_id }))
}

/// Load a startup model through the same containment and lifecycle path used by
/// embedded consumers: validate path -> metadata -> backend dispatch -> atomic
/// registry/engine registration.
pub async fn preload_startup_model(
    runtime: &Runtime,
    spec: &StartupModelSpec,
) -> Result<String, Box<dyn std::error::Error>> {
    let validated = runtime.model_loader.validate_path(&spec.path)?;
    let metadata = runtime.model_loader.load_metadata(&validated)?;
    let model_id = spec
        .model_id
        .clone()
        .unwrap_or_else(|| metadata.name.clone());
    let model = load_model_dispatch(validated.as_path(), &model_id)?;

    runtime
        .model_lifecycle
        .load(model_id.clone(), metadata, model)
        .await?;

    Ok(model_id)
}

/// Load runtime configuration from environment.
pub fn load_config() -> RuntimeConfig {
    let env = gg_config::load();
    RuntimeConfig {
        base_path: env.base_path,
        auth_token: env.auth_token,
        session_timeout: env.session_timeout,
        max_context_length: env.max_context_length,
        request_queue: env.request_queue,
        resource_limits: env.resource_limits,
        batch: env.batch,
        shutdown_timeout: env.shutdown_timeout,
        connections: env.connections,
        ipc_server: env.ipc_server,
        ..Default::default()
    }
}

/// Run the inference CLI command.
pub async fn run_inference(args: &[String]) -> i32 {
    let mut model_id = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 256usize;
    let mut stream = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_id = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Missing value for --model");
                    return 1;
                }
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Missing value for --prompt");
                    return 1;
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    max_tokens = args[i + 1].parse().unwrap_or(256);
                    i += 2;
                } else {
                    eprintln!("Missing value for --max-tokens");
                    return 1;
                }
            }
            "--stream" => {
                stream = true;
                i += 1;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                return 1;
            }
        }
    }

    if model_id.is_empty() || prompt.is_empty() {
        eprintln!(
            "Usage: GG-CORE infer --model <MODEL> --prompt <PROMPT> [--max-tokens N] [--stream]"
        );
        return 1;
    }

    let socket_path = get_socket_path();
    let client = CliIpcClient::new(socket_path);
    let params = InferenceParams {
        max_tokens,
        ..Default::default()
    };

    let result = if stream {
        client
            .send_streaming_inference(&model_id, &prompt, &params)
            .await
    } else {
        client.send_inference(&model_id, &prompt, &params).await
    };

    match result {
        Ok(output) => {
            if !stream {
                println!("{}", output);
            }
            0
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Run the IPC server with the given runtime.
pub async fn run_ipc_server(runtime: Runtime) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = get_socket_path();
    let handler = std::sync::Arc::new(runtime.ipc_handler);
    let connections = runtime.connections;
    let shutdown = runtime.shutdown;
    let shutdown_timeout = runtime.config.shutdown_timeout;
    let ipc_config = runtime.config.ipc_server.clone();

    let worker_shutdown = tokio_util::sync::CancellationToken::new();
    let worker_handle = gg_core::scheduler::spawn_worker_with_registry(
        runtime.request_queue.clone(),
        runtime.inference_engine.clone(),
        Some(runtime.model_lifecycle.clone()),
        Some(runtime.model_registry.clone()),
        Some(runtime.resource_limits.clone()),
        Some(runtime.security.clone()),
        worker_shutdown.clone(),
    );

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let server_handle = tokio::spawn(server::run_server(
        socket_path,
        handler,
        connections,
        shutdown_rx,
        ipc_config,
    ));

    tokio::signal::ctrl_c().await?;
    eprintln!("Shutdown signal received, draining...");

    let _ = shutdown_tx.send(true);

    match shutdown.initiate(shutdown_timeout).await {
        ShutdownResult::Complete => eprintln!("Shutdown complete"),
        ShutdownResult::Timeout { remaining } => {
            eprintln!("Shutdown timeout, {} requests remaining", remaining);
        }
    }

    worker_shutdown.cancel();
    runtime.request_queue.wake();
    let _ = worker_handle.await;

    if let Err(e) = server_handle.await? {
        eprintln!("Server error: {}", e);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parses_startup_model_with_explicit_id() {
        let parsed = parse_startup_model(&args(&[
            "gg-core-cli",
            "serve",
            "--model",
            "models/qwen.gguf",
            "--model-id",
            "local-qwen",
        ]))
        .unwrap();

        assert_eq!(
            parsed,
            Some(StartupModelSpec {
                path: "models/qwen.gguf".into(),
                model_id: Some("local-qwen".into()),
            })
        );
    }

    #[test]
    fn parses_startup_model_without_explicit_id() {
        let parsed = parse_startup_model(&args(&[
            "gg-core-cli",
            "serve",
            "--model",
            "models/qwen.gguf",
        ]))
        .unwrap();

        assert_eq!(
            parsed,
            Some(StartupModelSpec {
                path: "models/qwen.gguf".into(),
                model_id: None,
            })
        );
    }

    #[test]
    fn rejects_model_id_without_model_path() {
        let error = parse_startup_model(&args(&[
            "gg-core-cli",
            "serve",
            "--model-id",
            "orphan",
        ]))
        .unwrap_err();
        assert!(error.contains("requires --model"));
    }

    #[test]
    fn rejects_multiple_startup_models_until_lifecycle_ipc_slice() {
        let error = parse_startup_model(&args(&[
            "gg-core-cli",
            "serve",
            "--model",
            "models/a.gguf",
            "--model",
            "models/b.gguf",
        ]))
        .unwrap_err();
        assert!(error.contains("Only one startup --model"));
    }
}
