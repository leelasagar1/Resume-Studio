"""Writes resumes/master_data_scientist.txt: a fictional multi-page master
resume (7 roles, ~140 bullets, repeated page headers and page numbers) for
testing long inputs. Deterministic: rerunning gives the same file."""
import random
from pathlib import Path

random.seed(7)
ROLES = [
    ('Senior Data Scientist', 'Northwind Retail', 'March 2022 – Present', 'Chicago, IL', 34,
     ['Python', 'SQL', 'Spark', 'Databricks', 'XGBoost', 'PyTorch', 'Airflow', 'Tableau', 'MLflow', 'dbt'],
     ['demand forecasting', 'markdown pricing', 'store replenishment', 'inventory allocation', 'customer churn', 'basket analysis']),
    ('Data Scientist', 'Bluefin Logistics', 'June 2019 – February 2022', 'Dallas, TX', 30,
     ['Python', 'SQL', 'scikit-learn', 'PuLP', 'Gurobi', 'Snowflake', 'Power BI', 'Docker', 'AWS SageMaker'],
     ['route optimization', 'carrier selection', 'warehouse slotting', 'delivery ETA prediction', 'fuel cost modelling']),
    ('Data Analyst II', 'Maple Health Partners', 'August 2017 – May 2019', 'Columbus, OH', 24,
     ['SQL', 'R', 'Python', 'SAS', 'Tableau', 'Excel', 'PostgreSQL'],
     ['readmission risk', 'claims leakage', 'provider scorecards', 'appointment no-shows', 'staffing levels']),
    ('Data Analyst', 'Crescent Bank', 'July 2015 – July 2017', 'Charlotte, NC', 20,
     ['SQL', 'SAS', 'Excel', 'VBA', 'Teradata', 'Tableau'],
     ['credit line increases', 'fraud alerts', 'branch performance', 'collections prioritization', 'deposit attrition']),
    ('Business Intelligence Developer', 'Orbit Telecom', 'January 2014 – June 2015', 'Atlanta, GA', 14,
     ['SQL Server', 'SSIS', 'SSRS', 'Excel', 'T-SQL'],
     ['call center KPIs', 'network outage reporting', 'billing reconciliation', 'plan upgrades']),
    ('Research Assistant', 'State University Operations Lab', 'September 2012 – December 2013', 'Columbus, OH', 10,
     ['R', 'MATLAB', 'CPLEX', 'Python'],
     ['hospital bed scheduling', 'queueing simulation', 'inventory policies']),
    ('Analytics Intern', 'Pioneer Foods', 'May 2012 – August 2012', 'Cincinnati, OH', 8,
     ['Excel', 'Access', 'SQL'],
     ['promotion lift', 'shelf space', 'supplier fill rates']),
]
VERBS = ['Built', 'Designed', 'Rebuilt', 'Tuned', 'Automated', 'Migrated', 'Wrote', 'Ran', 'Owned', 'Cut', 'Shipped',
         'Led', 'Replaced', 'Set up', 'Validated', 'Documented', 'Scoped', 'Prototyped', 'Retired']
THINGS = ['a {t} model for {d}', 'the {d} pipeline in {t}', 'weekly {d} reports using {t}', 'a {t} dashboard tracking {d}',
          'feature tables for {d} in {t}', 'an A/B test of {d} changes, analysed in {t}', 'the {d} data quality checks in {t}',
          'a {t} job that refreshed {d} inputs nightly', 'a mixed-integer program for {d} solved with {t}',
          'exploratory analysis of {d} drivers in {t}']
OUTCOMES = ['cutting error by {n}%', 'saving about {n} analyst hours a month', 'for the {team} team',
            'which finance adopted for planning', 'lifting accuracy {n} points over the old rules',
            'used in {n} regional reviews each quarter', '', 'after three failed vendor attempts',
            'reducing stockouts {n}%', 'so the {team} leads stopped waiting on IT']
EXTRAS = ['', ' The model ran on {n}M rows pulled from {k} source systems.',
          ' Agreed metric definitions with the {team} team before launch.',
          ' Handed it over with a runbook and a two-page explainer for analysts.',
          ' Replaced a spreadsheet process that took {k} people most of a week.',
          ' Presented results monthly to the {team} director and her staff.']
TEAMS = ['merchandising', 'supply chain', 'finance', 'operations', 'pricing', 'store ops', 'risk']
HEADER = ['Jordan Lee', 'jordan.lee@example.com | 555-010-3131 | linkedin.com/in/jordanlee-example | Chicago, IL']

out = HEADER + ['', 'PROFESSIONAL SUMMARY',
    'Data scientist with 11 years across retail, logistics, healthcare and banking analytics. Works from messy '
    'operational data to models the business actually runs on, with most recent work in forecasting, optimization '
    'and experimentation.', '', 'TECHNICAL SKILLS',
    'Languages: Python, SQL, R, SAS, VBA, MATLAB, T-SQL',
    'ML & Statistics: scikit-learn, XGBoost, PyTorch, statsmodels, Prophet, survival analysis, causal inference, Bayesian methods',
    'Optimization: PuLP, Gurobi, CPLEX, linear programming, mixed-integer programming, simulation',
    'Data & Cloud: Spark, Databricks, Snowflake, Teradata, PostgreSQL, SQL Server, AWS SageMaker, Airflow, dbt, MLflow, Docker',
    'BI: Tableau, Power BI, SSRS, Excel', '', 'PROFESSIONAL EXPERIENCE']
page = 1
for title, company, dates, place, count, tools, domains in ROLES:
    out += ['', f'{title} — {company}', f'{dates} | {place}']
    seen = set()
    while count:
        thing = random.choice(THINGS).format(t=random.choice(tools), d=random.choice(domains))
        outcome = random.choice(OUTCOMES).format(n=random.randint(4, 38), team=random.choice(TEAMS))
        joined = (f', {outcome}' if outcome and random.random() < .5
                  else f' {outcome}' if outcome.startswith(('for', 'after', 'so', 'which', 'used')) else '')
        line = f'• {random.choice(VERBS)} {thing}{joined}.' + random.choice(EXTRAS).format(
            n=random.randint(2, 90), k=random.randint(3, 9), team=random.choice(TEAMS))
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        count -= 1
        if len(out) > page * 48:
            page += 1
            out += [f'Page {page - 1} of 6', *HEADER]
out += ['', 'PROJECTS', 'Open-source demand forecasting toolkit — personal project', '2021 – 2023',
        '• Maintained a small Python package for hierarchical forecasting, with 40 GitHub stars and two outside contributors.',
        '• Wrote documentation and example notebooks for retail and logistics use cases.',
        '', 'EDUCATION', 'MS in Operations Research | State University | 2013', 'BS in Mathematics | Lakeview College | 2011',
        '', 'CERTIFICATIONS', 'AWS Certified Machine Learning – Specialty — 2021', 'Page 6 of 6']
Path(__file__).resolve().parent.joinpath('resumes', 'master_data_scientist.txt').write_text('\n'.join(out) + '\n')
