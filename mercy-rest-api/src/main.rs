// mercy-rest-api/src/main.rs — REST API for MeTTa Symbolic Operations
use axum::{
    routing::{get, post},
    Router, Json, extract::{State, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use chrono::Utc;

// Reuse from prior weaves (assume paths adjusted)
use mercy_orchestrator::arango_integration::ArangoMercyStore;

#[derive(Error, Debug)]
enum ApiError {
    #[error("Mercy shield: {0}")]
    MercyRejection(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match self {
            ApiError::MercyRejection(m) => (StatusCode::FORBIDDEN, m),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string()),
        };
        (status, Json(serde_json::json!({"error": msg}))).into_response()
    }
}

#[derive(Deserialize)]
struct EvalRequest {
    expression: String,
    valence: f64,
    context: Option<String>,
}

#[derive(Serialize)]
struct EvalResponse {
    input: String,
    output: String,
    success: bool,
    timestamp: String,
}

#[derive(Deserialize)]
struct InsertRequest {
    text: String,
    valence: f64,
    context: Option<String>,
}

#[derive(Serialize)]
struct AtomResponse {
    text: String,
    valence: f64,
    context: String,
    timestamp: String,
}

#[derive(Deserialize)]
struct QueryParams {
    min_valence: Option<f64>,
}

#[derive(Clone)]
struct AppState {
    store: Arc<ArangoMercyStore>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = ArangoMercyStore::new("http://localhost:8529", "root", "", "nexi_mercy").await?;
    let state = AppState { store: Arc::new(store) };

    let app = Router::new()
        .route("/metta/eval", post(eval_metta))
        .route("/metta/atoms", get(query_high_valence))
        .route("/metta/insert", post(insert_atom))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    println!("MeTTa REST API at http://{}", addr);
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

// POST /metta/eval
async fn eval_metta(
    State(state): State<AppState>,
    Json(req): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, ApiError> {
    if req.valence < 0.9999999 {
        return Err(ApiError::MercyRejection("low valence — .metta eval rejected".into()));
    }

    let result = state.store.foxx_metta_eval(&req.expression, req.valence, req.context.as_deref()).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(EvalResponse {
        input: req.expression,
        output: result,
        success: true,
        timestamp: Utc::now().to_rfc3339(),
    }))
}

// GET /metta/atoms?min_valence=0.999
async fn query_high_valence(
    State(state): State<AppState>,
    Query(params): Query<QueryParams>,
) -> Result<Json<Vec<AtomResponse>>, ApiError> {
    let min = params.min_valence.unwrap_or(0.9999999);
    if min < 0.9999999 {
        return Err(ApiError::MercyRejection("query valence too low".into()));
    }

    let atoms = state.store.query_high_valence(min).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let resp = atoms.into_iter().map(|(text, valence)| AtomResponse {
        text,
        valence,
        context: "default".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    }).collect();

    Ok(Json(resp))
}

// POST /metta/insert
async fn insert_atom(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<AtomResponse>, ApiError> {
    if req.valence < 0.9999999 {
        return Err(ApiError::MercyRejection("low valence — insert rejected".into()));
    }

    state.store.insert_metta_atom(&req.text, req.valence, req.context.as_deref()).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(AtomResponse {
        text: req.text,
        valence: req.valence,
        context: req.context.unwrap_or("default".to_string()),
        timestamp: Utc::now().to_rfc3339(),
    }))
}
