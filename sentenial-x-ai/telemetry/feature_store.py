# telemetry/feature_store.py
import sqlalchemy as sa
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime
engine = create_engine("postgresql://user:pass@db/sentenial_x")
meta = MetaData()

features = Table(
    "features", meta,
    Column("id", String, primary_key=True),
    Column("entity_id", String, index=True),
    Column("f1", Float),
    Column("f2", Float),
    Column("ts", DateTime),
)
meta.create_all(engine)

def upsert_features(fdict):
    with engine.begin() as conn:
        conn.execute(features.insert().values(**fdict))
