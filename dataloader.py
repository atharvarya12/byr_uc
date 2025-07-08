from sqlalchemy import create_engine
import pandas as pd
 
print("Connecting to AACT database...")
 
engine = create_engine(
    "postgresql+psycopg2://ritam21:Ritam03%40@aact-db.ctti-clinicaltrials.org:5432/aact"
)
 
print("Connection successful. Running query...")
 
df = pd.read_sql("""
    SELECT
    s.nct_id,
    s.phase,
    c.name AS condition,
    i.intervention_type AS intervention_type,
    d.allocation AS study_design,
    sp.agency_class AS sponsor_type,
    s.enrollment,
    s.overall_status AS status,
    e.gender,
    e.minimum_age AS min_age,
    e.maximum_age AS max_age,
    f.country AS location,
    s.start_date,
    s.completion_date,
    COALESCE(r.title, 'No result') AS results,
    oa.p_value,
    oa.p_value_modifier,
    oa.param_type,
    oa.ci_upper_limit,
    oa.ci_lower_limit
FROM
    studies s
LEFT JOIN conditions c ON s.nct_id = c.nct_id
LEFT JOIN interventions i ON s.nct_id = i.nct_id
LEFT JOIN designs d ON s.nct_id = d.nct_id
LEFT JOIN sponsors sp ON s.nct_id = sp.nct_id
LEFT JOIN eligibilities e ON s.nct_id = e.nct_id
LEFT JOIN facilities f ON s.nct_id = f.nct_id
LEFT JOIN result_groups r ON s.nct_id = r.nct_id
LEFT JOIN outcome_analyses oa ON r.nct_id = oa.nct_id
WHERE
    s.overall_status = 'COMPLETED'
ORDER BY
    s.nct_id
LIMIT 100000;

""", engine)
 
print(f"Query returned {len(df)} rows.")
 
df.to_csv("trial_data_n.csv", index=False)
print("Data saved to trial_data.csv")
print(df.head())
print(df.columns)
