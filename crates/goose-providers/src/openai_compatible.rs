use crate::conversation::token_usage::{CostSource, ProviderUsage};
use crate::http_status::read_json_response;
use crate::images::ImageFormat;
use anyhow::Error;
use async_stream::try_stream;
use futures::{Stream, TryStreamExt};
use reqwest::Response;
#[cfg(test)]
use reqwest::StatusCode;
use serde_json::Value;
use std::pin::Pin;
use std::time::Duration;
use tokio::pin;
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

use super::api_client::ApiClient;
use super::base::{stream_from_single_message, MessageStream, Provider};
use super::retry::ProviderRetry;
use crate::conversation::message::Message;
use crate::errors::ProviderError;
use crate::formats::openai::{
    create_request, create_request_for_model_with_options, get_cost, get_usage,
    record_response_metadata, response_to_message, response_to_streaming_message,
    OpenAiFormatOptions,
};
use crate::formats::openai_responses::responses_api_to_streaming_message;
use crate::model::ModelConfig;
use crate::request_log::{start_log, LoggerHandleExt, RequestLogHandle};
use rmcp::model::Tool;

pub struct OpenAiCompatibleProvider {
    name: String,
    /// Client targeted at the base URL (e.g. `https://api.x.ai/v1`)
    api_client: ApiClient,
    /// Path prefix prepended to `chat/completions` (e.g. `"deployments/{name}/"` for Azure).
    completions_prefix: String,
    supports_streaming: bool,
}

impl OpenAiCompatibleProvider {
    pub fn new(name: String, api_client: ApiClient, completions_prefix: String) -> Self {
        Self {
            name,
            api_client,
            completions_prefix,
            supports_streaming: true,
        }
    }

    pub fn with_supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.supports_streaming = supports_streaming;
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn build_request_for_model(
        &self,
        model_config: &ModelConfig,
        wire_model: &str,
        capability_model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        for_streaming: bool,
    ) -> Result<Value, ProviderError> {
        create_request_for_model_with_options(
            model_config,
            wire_model,
            capability_model,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            for_streaming,
            OpenAiFormatOptions {
                preserve_thinking_context: true,
                supports_vision: model_config.supports_vision.unwrap_or_default(),
                ..Default::default()
            },
        )
        .map_err(|e| ProviderError::RequestFailed(format!("Failed to create request: {}", e)))
    }

    pub async fn stream_for_model(
        &self,
        model_config: &ModelConfig,
        wire_model: &str,
        capability_model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = self.build_request_for_model(
            model_config,
            wire_model,
            capability_model,
            system,
            messages,
            tools,
            self.supports_streaming,
        )?;
        self.stream_payload(model_config, payload).await
    }

    async fn stream_payload(
        &self,
        model_config: &ModelConfig,
        payload: Value,
    ) -> Result<MessageStream, ProviderError> {
        let mut log = start_log(model_config, &payload)?;
        let path = format!("{}chat/completions", self.completions_prefix);
        let response = self
            .with_retry(|| async {
                handle_status(
                    self.api_client
                        .request(&path)
                        .model_headers(model_config)?
                        .streaming(self.supports_streaming)
                        .response_post(&payload)
                        .await?,
                )
                .await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;
        if self.supports_streaming {
            stream_openai_compat(response, log)
        } else {
            let json = read_json_response(response).await?;
            let message = response_to_message(&json).map_err(|e| {
                ProviderError::RequestFailed(format!("Failed to parse message: {}", e))
            })?;
            let usage_json = json.get("usage").unwrap_or(&Value::Null);
            let usage_data = get_usage(usage_json);
            let mut usage = ProviderUsage::new(model_config.model_name.clone(), usage_data);
            record_response_metadata(&mut usage, &json);
            if let Some(cost) = get_cost(usage_json) {
                usage = usage.with_cost(cost, CostSource::ProviderReported);
            }
            log.write(
                &serde_json::to_value(&message).unwrap_or_default(),
                Some(&usage.usage),
            )?;
            Ok(stream_from_single_message(message, usage))
        }
    }

    fn build_request(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        for_streaming: bool,
    ) -> Result<Value, ProviderError> {
        create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            for_streaming,
        )
        .map_err(|e| ProviderError::RequestFailed(format!("Failed to create request: {}", e)))
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        self.api_client
            .refresh_credentials()
            .await
            .map_err(|error| ProviderError::Authentication(error.to_string()))
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self
            .api_client
            .response_get("models")
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let json = handle_response_openai_compat(response).await?;

        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(ProviderError::Authentication(msg.to_string()));
        }

        let arr = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            ProviderError::RequestFailed("Missing 'data' array in models response".to_string())
        })?;
        let mut models: Vec<String> = arr
            .iter()
            .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        models.sort();
        Ok(models)
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = self.build_request(
            model_config,
            system,
            messages,
            tools,
            self.supports_streaming,
        )?;
        self.stream_payload(model_config, payload).await
    }
}

// Re-exported from the dedicated `http_status` module — these helpers are
// format-agnostic and used across all provider families.
pub use super::http_status::{
    handle_response, handle_status, map_http_error_to_provider_error, sanitize_url,
};

// Legacy alias kept for callers that haven't migrated their import path yet.
pub use super::http_status::handle_response as handle_response_openai_compat;

/// Idle timeout for streaming responses: how long a stream may go without a
/// data-bearing SSE line before it is considered stalled. SSE comment
/// keepalives (`: ping`) and blank separators do not reset the deadline, so a
/// provider that wedges mid-stream while emitting keepalives is detected
/// instead of hanging the turn forever.
pub(crate) const STREAM_IDLE_TIMEOUT_SECS: u64 = 120;

fn is_data_bearing_line(line: &str) -> bool {
    !line.trim().is_empty() && !line.starts_with(':')
}

/// Wraps framed SSE lines with the idle timeout, between `LinesCodec` framing
/// and the format parser. The deadline applies only after the first line, so
/// time-to-first-token stays governed by the request timeout; afterwards only
/// data-bearing lines reset it. On timeout the stream errors with a retryable
/// [`ProviderError::NetworkError`].
pub(crate) fn with_data_line_idle_timeout(
    mut stream: impl Stream<Item = anyhow::Result<String>> + Unpin + Send + 'static,
    idle_timeout_secs: u64,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>> {
    let idle_timeout = Duration::from_secs(idle_timeout_secs);
    Box::pin(try_stream! {
        let first_line = match stream.next().await {
            Some(first) => first?,
            None => return,
        };
        yield first_line;
        let mut deadline = tokio::time::Instant::now() + idle_timeout;

        loop {
            match tokio::time::timeout_at(deadline, stream.next()).await {
                Ok(Some(item)) => {
                    let line = item?;
                    if is_data_bearing_line(&line) {
                        deadline = tokio::time::Instant::now() + idle_timeout;
                    }
                    yield line;
                }
                Ok(None) => break,
                Err(_) => {
                    let err = ProviderError::NetworkError(format!(
                        "Stream stalled: no data-bearing SSE line received for \
                         {idle_timeout_secs}s (keepalive comment frames do not count \
                         as progress)"
                    ));
                    Err::<(), anyhow::Error>(err.into())?;
                }
            }
        }
    })
}

pub fn stream_openai_compat(
    response: Response,
    log: Option<Box<dyn RequestLogHandle>>,
) -> Result<MessageStream, ProviderError> {
    stream_openai_compat_with_idle_timeout(response, log, STREAM_IDLE_TIMEOUT_SECS)
}

fn stream_openai_compat_with_idle_timeout(
    response: Response,
    mut log: Option<Box<dyn RequestLogHandle>>,
    idle_timeout_secs: u64,
) -> Result<MessageStream, ProviderError> {
    let stream = response.bytes_stream().map_err(std::io::Error::other);

    Ok(Box::pin(try_stream! {
        let stream_reader = StreamReader::new(stream);
        let framed = FramedRead::new(stream_reader, LinesCodec::new())
            .map_err(Error::from);

        let timed_lines = with_data_line_idle_timeout(framed, idle_timeout_secs);
        let message_stream = response_to_streaming_message(timed_lines);
        pin!(message_stream);
        while let Some(message) = message_stream.next().await {
            let (message, usage) = message.map_err(|e|
                e.downcast::<ProviderError>()
                    .unwrap_or_else(ProviderError::stream_decode_error)
            )?;
            log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
            yield (message, usage);
        }
    }))
}

pub fn stream_responses_compat(
    response: Response,
    mut log: Option<Box<dyn RequestLogHandle>>,
) -> Result<MessageStream, ProviderError> {
    let stream = response.bytes_stream().map_err(std::io::Error::other);

    Ok(Box::pin(try_stream! {
        let stream_reader = StreamReader::new(stream);
        let framed = FramedRead::new(stream_reader, LinesCodec::new())
            .map_err(Error::from);

        let message_stream = responses_api_to_streaming_message(framed);
        pin!(message_stream);
        while let Some(message) = message_stream.next().await {
            let (message, usage) = message.map_err(|e|
                e.downcast::<ProviderError>()
                    .unwrap_or_else(ProviderError::stream_decode_error)
            )?;
            log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
            yield (message, usage);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use serde_json::json;
    use test_case::test_case;

    #[test_case(
        StatusCode::PAYMENT_REQUIRED,
        Some(json!({"error": {"message": "Insufficient credits to complete this request"}})),
        "CreditsExhausted"
        ; "402 with payload"
    )]
    #[test_case(
        StatusCode::PAYMENT_REQUIRED,
        None,
        "CreditsExhausted"
        ; "402 without payload"
    )]
    #[test_case(
        StatusCode::TOO_MANY_REQUESTS,
        Some(json!({"error": {"message": "Rate limit exceeded"}})),
        "RateLimitExceeded"
        ; "429 rate limit"
    )]
    #[test_case(
        StatusCode::UNAUTHORIZED,
        None,
        "Authentication"
        ; "401 unauthorized"
    )]
    #[test_case(
        StatusCode::BAD_REQUEST,
        Some(json!({"error": {"message": "This request exceeds the maximum context length"}})),
        "ContextLengthExceeded"
        ; "400 context length"
    )]
    #[test_case(
        StatusCode::INTERNAL_SERVER_ERROR,
        None,
        "ServerError"
        ; "500 server error"
    )]
    #[test_case(
        StatusCode::NOT_FOUND,
        None,
        "RequestFailed"
        ; "404 not found"
    )]
    #[test_case(
        StatusCode::NOT_FOUND,
        Some(json!({"error": {"message": "model not available"}})),
        "RequestFailed"
        ; "404 with error payload"
    )]
    fn http_status_maps_to_expected_error(
        status: StatusCode,
        payload: Option<Value>,
        expected_variant: &str,
    ) {
        let err = map_http_error_to_provider_error(status, payload, "http://test/endpoint");
        let actual = err.telemetry_type();
        let expected_telemetry = match expected_variant {
            "CreditsExhausted" => "credits_exhausted",
            "RateLimitExceeded" => "rate_limit",
            "Authentication" => "auth",
            "ContextLengthExceeded" => "context_length",
            "ServerError" => "server",
            "RequestFailed" => "request",
            other => panic!("Unknown variant: {other}"),
        };
        assert_eq!(
            actual, expected_telemetry,
            "Expected {expected_variant}, got error: {err:?}"
        );
    }

    #[test]
    fn build_request_respects_non_streaming_mode() {
        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(
                "http://localhost".to_string(),
                super::super::api_client::AuthMethod::NoAuth,
                None,
            )
            .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let model = ModelConfig::new("test-model");
        let payload = provider
            .build_request(&model, "", &[], &[], provider.supports_streaming)
            .unwrap();

        assert_eq!(payload.get("stream"), None);
        assert_eq!(payload.get("stream_options"), None);
    }

    #[tokio::test]
    async fn nonstreaming_completion_accepts_legitimate_response() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"}
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(server.uri(), crate::api_client::AuthMethod::NoAuth, None)
                .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let _stream = provider
            .stream(&ModelConfig::new("test-model"), "", &[], &[])
            .await
            .expect("legitimate non-streaming response should be accepted");
    }

    #[tokio::test]
    async fn nonstreaming_completion_rejects_oversized_response_body() {
        use crate::http_status::MAX_PROVIDER_JSON_RESPONSE_BYTES;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "a".repeat(MAX_PROVIDER_JSON_RESPONSE_BYTES + 1)
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            "test".to_string(),
            ApiClient::new_with_tls(server.uri(), crate::api_client::AuthMethod::NoAuth, None)
                .unwrap(),
            String::new(),
        )
        .with_supports_streaming(false);

        let err = match provider
            .stream(&ModelConfig::new("test-model"), "", &[], &[])
            .await
        {
            Ok(_) => panic!("oversized response should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("response body exceeds"),
            "got: {err}"
        );
    }

    fn chat_delta(content: &str) -> String {
        let payload = json!({
            "id": "1",
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": null,
            }],
        });
        format!("data: {payload}\n\n")
    }

    /// The #11679 failure signature: healthy data frames, then keepalive
    /// comment frames forever. Bytes keep flowing, so byte-based read
    /// timeouts never fire.
    #[tokio::test]
    async fn keepalive_masked_stall_errors_instead_of_hanging() {
        use crate::retry::{should_retry, RetryConfig};
        use crate::sse_test_harness::{drain_within, post_stream, spawn_server, Chunk, Tail};
        use std::time::Duration;

        let delta = chat_delta("Hi");
        let addr = spawn_server(
            vec![Chunk::immediate(delta.clone()), Chunk::immediate(delta)],
            Tail::KeepaliveForever,
        )
        .await;

        let response = post_stream(
            addr,
            "/chat/completions",
            json!({"model": "m", "stream": true, "messages": []}),
        )
        .await;
        let stream = stream_openai_compat_with_idle_timeout(response, None, 1).unwrap();

        let (items, err) = drain_within(stream, Duration::from_secs(15)).await;
        let err = err.expect("stream ended without an error");
        assert!(
            items > 0,
            "healthy frames should have been yielded before the stall"
        );
        assert!(err.to_string().contains("Stream stalled"), "got: {err}");
        assert!(matches!(err, ProviderError::NetworkError(_)));
        assert!(should_retry(&err, &RetryConfig::default()));
    }

    #[tokio::test]
    async fn slow_but_live_data_lines_do_not_trip_the_idle_timeout() {
        use crate::sse_test_harness::{drain_within, post_stream, spawn_server, Chunk, Tail};
        use std::time::Duration;

        // Data frames interleaved with keepalive comments every 300ms against
        // a 1s idle timeout: a live-but-slow stream must not be killed.
        let mut script = Vec::new();
        for i in 0..5 {
            script.push(Chunk {
                after: Duration::from_millis(150),
                body: ": ping\n\n".to_string(),
            });
            script.push(Chunk {
                after: Duration::from_millis(150),
                body: chat_delta(&format!("Hi {i}")),
            });
        }
        script.push(Chunk::immediate("data: [DONE]\n\n"));
        let addr = spawn_server(script, Tail::Close).await;

        let response = post_stream(
            addr,
            "/chat/completions",
            json!({"model": "m", "stream": true, "messages": []}),
        )
        .await;
        let stream = stream_openai_compat_with_idle_timeout(response, None, 1).unwrap();

        let (items, err) = drain_within(stream, Duration::from_secs(15)).await;
        assert!(err.is_none(), "live stream errored: {err:?}");
        assert!(items > 0, "data frames should have been yielded");
    }
}
