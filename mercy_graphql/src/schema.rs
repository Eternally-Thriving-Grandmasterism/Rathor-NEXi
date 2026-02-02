// mercy_graphql/src/schema.rs — GraphQL Schema for MeTTa Integration
use async_graphql::{Object, Schema, EmptySubscription, SimpleObject, Context};
use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use axum::{response::Html, routing::get, Router};
use crate::arango_integration::ArangoMercyStore; // Or other backends

#[derive(SimpleObject)]
pub struct MettaAtom {
    text: String,
    valence: f64,
    context: String,
    timestamp: String,
}

#[derive(SimpleObject)]
pub struct EvalResult {
    input: String,
    output: String,
    success: bool,
}

pub struct Query;

#[Object]
impl Query {
    async fn high_valence_atoms(
        &self,
        ctx: &Context<'_>,
        min_valence: f64,
    ) -> async_graphql::Result<Vec<MettaAtom>> {
        let store = ctx.data::<ArangoMercyStore>()?; // Injected via extension
        let atoms = store.query_high_valence(min_valence).await?;
        Ok(atoms.into_iter().map(|(text, valence)| MettaAtom {
            text,
            valence,
            context: "default".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }).collect())
    }

    async fn eval_metta(
        &self,
        ctx: &Context<'_>,
        expression: String,
        valence: f64,
    ) -> async_graphql::Result<EvalResult> {
        if valence < 0.9999999 {
            return Err("Mercy shield: low valence — .metta eval rejected".into());
        }
        let store = ctx.data::<ArangoMercyStore>()?;
        let result = store.foxx_metta_eval(&expression, valence, Some("graphql")).await?;
        Ok(EvalResult {
            input: expression,
            output: result,
            success: true,
        })
    }
}

pub struct Mutation;

#[Object]
impl Mutation {
    async fn insert_metta_atom(
        &self,
        ctx: &Context<'_>,
        text: String,
        valence: f64,
        context: Option<String>,
    ) -> async_graphql::Result<MettaAtom> {
        if valence < 0.9999999 {
            return Err("Mercy shield: persistence rejected".into());
        }
        let store = ctx.data::<ArangoMercyStore>()?;
        store.insert_metta_atom(&text, valence, context.as_deref()).await?;
        Ok(MettaAtom {
            text,
            valence,
            context: context.unwrap_or("default".to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
}

pub type MercySchema = Schema<Query, Mutation, EmptySubscription>;

pub fn build_schema(store: ArangoMercyStore) -> MercySchema {
    Schema::build(Query, Mutation, EmptySubscription)
        .data(store)
        .finish()
}
