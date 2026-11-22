# Job-Search Agent with DeepEval: Project Spec

Draft v1 · 18 November 2026 · Budget: free tiers only

## 1. Goal

Build one AI agent that reads my resume profile, searches for AI engineering jobs, scores how well each job matches me, and saves the good ones to a CSV file. Then evaluate and monitor the agent with DeepEval and Confident AI, so I get hands-on experience with every part of DeepEval: metrics, datasets, tracing, pytest tests, dashboards and monitoring.

The agent is the means. The real learning goal is the evaluation setup around it.

## 2. Scope

**In scope**

- One LangGraph agent (no orchestrator, no subagents in version 1)
- Two real MCP servers: Firecrawl for search and scraping, Filesystem for the CSV
- A one-time resume-to-profile step
- A golden dataset of about 25 goldens
- End-to-end and component-level evals
- Pytest tests that run with `deepeval test run`
- Tracing, a dashboard and monitoring on Confident AI
- A written comparison of two agent versions

**Out of scope for version 1**

- Multi-agent or swarm designs (a possible version 2)
- Voice interfaces
- Auto-applying to jobs
- Google Sheets (start with CSV, add Sheets later if time allows)

## 3. Constraints

Everything must run on free tiers. These limits shape the design. Verify each one on the provider’s own page before relying on it, because free plans change.

| Part | Free option | Limit to plan around |
| --- | --- | --- |
| Agent LLM | Groq free plan | Per-organisation limits. Sample text-model limits: about 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day. Exact numbers are in the Groq console. |
| Judge LLM | Same Groq key | Shares the same budget as the agent |
| Evals | DeepEval (open source) | None, runs locally |
| Search | Firecrawl MCP | Monthly credit allowance; search and scrape cost about 1 credit each. Sources disagree between 1,000/month and 500 one-time, so check the pricing page. |
| Output | Filesystem MCP server | None |
| Dashboard and monitoring | Confident AI free plan | Limited test runs per week, one project, limited trace storage (one third-party listing says 5 runs/week and 1 GB-month of traces; verify) |

**Rules that follow from these limits**

1. Run the agent once per golden, save the outputs to a JSON file, and run metrics on the saved file as often as needed.
2. Run LLM-judged metrics one at a time, not in parallel, to stay under the tokens-per-minute limit.
3. Prefer metrics that need no LLM judge (Tool Correctness) for path checks.
4. Save a few real Firecrawl results and replay them in tests, so tests do not spend credits.
5. Upload only meaningful runs to Confident AI, because test runs per week are limited.
6. The resume is personal data. Keep it out of anything shared or published.

## 4. Architecture

```
Resume (text)  ->  Profile builder (run once)  ->  profile.json
                                                       |
                                                       v
          +-------------- Single LangGraph agent (Groq LLM) --------------+
          |   Search -> Read job -> Match -> Save row                     |
          +--------+-------------------+------------------------+--------+
                   |                   |                        |
             Firecrawl MCP       Firecrawl MCP            Filesystem MCP
             (search)            (scrape page)            (write CSV)
                                                                |
                                                            jobs.csv

DeepEval layer (watches everything above):
  Tracing  ->  Evals (metrics)  ->  Confident AI (dashboard, monitoring)
```

The agent is one agent. The Groq LLM decides which step to take next. The Match step uses only the LLM and the profile, with no tool.

## 5. Components

### 5.1 Profile builder (run once)

Reads the resume as plain text (strip LaTeX formatting first) and produces `profile.json`. The agent receives this short profile, not the full resume, to save tokens.

**Fields**

| Field | Meaning |
| --- | --- |
| `target_roles` | Job titles to search for |
| `skills` | Skills that appear in the resume |
| `years_experience` | Experience level |
| `max_experience_years` | Highest experience requirement to accept |
| `locations` | Preferred cities or remote |
| `include_keywords` | Extra search words |
| `exclude_keywords` | Words that rule a job out |

**Hard rule:** every entry in `skills` must appear in the resume text. No invented skills. This is checked by an eval (section 6).

### 5.2 Agent

- Framework: LangGraph, one agent, tools loaded from MCP servers
- LLM: a Groq model (check the console for current model names), temperature 0
- Step limit: a maximum of about 12 tool calls per run, to stop loops
- System prompt rules:
  - Use only facts that came from tool results or the profile
  - If a job page cannot be read, mark the job as unknown instead of guessing
  - Save a job only if its match score is at or above the save threshold
  - Never save the same job URL twice
- The agent exposes `run_agent(question)` that returns the final answer and the list of tool calls, so evals can use it directly

### 5.3 MCP servers

| Server | Used for | Used by step |
| --- | --- | --- |
| Firecrawl MCP | Search for jobs, scrape job pages | Search, Read job |
| Filesystem MCP | Read and write `jobs.csv` | Save row |

Tool descriptions must be clear, because vague descriptions cause wrong tool choices. During tests, write only to a copy of the CSV.

### 5.4 Output: `jobs.csv`

| Column | Notes |
| --- | --- |
| `url` | Unique key, used to prevent duplicates |
| `title`, `company`, `location` | From the job page |
| `posted_date` | If available, otherwise empty |
| `match_score` | 0 to 100 |
| `matched_skills` | Skills found in both the profile and the job |
| `missing_skills` | Skills the job wants that the profile lacks |
| `reason` | One or two sentences |
| `saved_at` | Timestamp |

The save threshold (for example 60) is fixed before any comparison runs and kept the same for every version.

## 6. Evaluation plan

### 6.1 Golden dataset

About 25 goldens, each with an `input`, an expected behaviour or expected tools where possible, and a category tag.

| Category | Count | Examples |
| --- | --- | --- |
| Search | 7 | Role and city filters, recent postings only |
| Read job | 4 | Valid page, broken link |
| Match | 6 | Strong fit, weak fit, partial fit |
| Save and duplicates | 4 | Save a good job, skip a duplicate, skip below threshold |
| Edge and failure | 4 | Tool error, missing information, out-of-scope request |

Because real job data changes daily, tests judge the path (right tool, right arguments) rather than today’s job listings.

### 6.2 Metrics

| Step | What to check | DeepEval metric |
| --- | --- | --- |
| Profile | Every skill is in the resume | G-Eval (custom criteria) or a faithfulness-style check |
| Search | Right keywords and filters | Argument Correctness, MCP Use |
| Read job | Right tool, right URL | Tool Correctness |
| Match | Score is sensible, reasons use only real skills | G-Eval, plus agreement with my own labels |
| Save | Right values in right columns, no duplicates | Argument Correctness, plus a plain check of the CSV |
| Whole run | Did it finish the job | Task Completion |

Starting threshold is 0.7 for LLM-judged metrics. Thresholds are set before comparing versions.

### 6.3 Judge model

DeepEval defaults to an OpenAI judge, so a custom judge model is needed to use Groq. Use a different Groq model for judging than for the agent when possible, to reduce self-preference bias. If scores look random or parsing fails, move to a stronger judge model.

### 6.4 End-to-end and component-level

- End-to-end: treat the agent as one box and score the final result (Task Completion)
- Component-level: trace each step and attach a metric to it, so a failure points to one step
- Trajectory: Tool Correctness and Argument Correctness judge the path the agent took

### 6.5 Match quality against humans

Label 20 real jobs by hand as “good fit” or “not a fit”. Measure how often the agent’s decision agrees with mine. If agreement is low, fix the rubric before trusting the match score.

### 6.6 Statistics rules for comparing versions

1. Run each version several times to measure run-to-run wobble.
2. Trust a gain only when it is clearly bigger than the wobble.
3. Use a paired test (McNemar’s) on the same goldens.
4. Check each category, so a total gain does not hide a drop in an important area.
5. A small gain on 25 goldens is not proof. Say “no clear difference” when that is the case.

### 6.7 Tests in CI

- Write evals as pytest tests using `assert_test`
- Run them with `deepeval test run`
- Tests fail when a metric is below its threshold
- A GitHub Actions workflow can run them on every push (optional, free for public repos and limited minutes for private ones)

## 7. Tracing, dashboard and monitoring

This section covers Confident AI, the hosted platform from the DeepEval team.

### 7.1 Setup

- Create a free Confident AI account and log in from the CLI with `deepeval login`
- Add tracing to the agent so every run is recorded step by step (LLM calls, tool calls). DeepEval has a LangGraph integration, so follow the Integrations page for the exact setup for the installed version.

### 7.2 Dashboard (development)

Use the dashboard to:

- See each test run with scores, pass or fail, and the judge’s reasons
- Compare two runs side by side to catch regressions after a prompt or tool-description change
- Store the golden dataset in the cloud and pull it into tests
- Open a single trace to see which step failed

Because free test runs per week are limited, develop locally and upload only runs worth keeping: the baseline, each major change, and the final comparison.

### 7.3 Monitoring (real use)

When the agent runs for real (searching for my own jobs), each run is traced to Confident AI.

- Score a sample of live traces with cheap metrics (Tool Correctness, Task Completion). Avoid scoring everything, to protect Groq limits.
- Watch these over time: task completion rate, tool-error rate, average number of steps per run, runs that hit the step limit, and duplicate saves
- Set a simple alert rule by hand: if the task completion rate drops below the baseline for a week, investigate
- Failure loop: review failed traces weekly, turn each real failure into a new golden, and re-run the dataset

### 7.4 What to check on the free plan

Confirm on the current pricing page: the number of test runs per week, trace storage, number of projects and seats, and whether online evals on live traces are included.

## 8. Milestones

| Phase | Work | Done when |
| --- | --- | --- |
| 1 | Profile builder and `profile.json` | Profile matches the resume, with no invented skills |
| 2 | Agent with Firecrawl MCP, search and read only | Runs 5 manual questions with correct tool calls |
| 3 | Add Match and Save with Filesystem MCP | A run writes correct rows to a test CSV |
| 4 | Golden dataset (25 goldens) with tags | Dataset saved locally and in Confident AI |
| 5 | End-to-end evals with Groq judge | Baseline scores recorded from a saved outputs file |
| 6 | Tracing and component-level metrics | One trace shows every step, with a metric on at least one step |
| 7 | Pytest tests and CI | `deepeval test run` passes locally and in CI |
| 8 | Dashboard and monitoring | Baseline run on the dashboard, live runs traced, weekly review routine started |
| 9 | Improve and compare | Version 2 compared with version 1 using the statistics rules |

A relaxed pace is about one to two weeks for phases 1 to 5, and another week for phases 6 to 9.

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Groq tokens-per-minute limit slows evals | Run metrics one at a time, cache outputs, keep the dataset small |
| Firecrawl credits run out | Replay saved results in tests, search sparingly for live runs |
| Job sites block scraping | Use search results first, handle “cannot read page” as a test case |
| Real data changes daily | Judge the path and tool use, not today’s listings |
| A bad write corrupts the real CSV | Write to a test copy during evals, add a save threshold and duplicate check |
| Free judge model is unreliable | Try a stronger Groq model for the judge, spot-check scores by hand |
| Free Confident AI limits reached | Upload only key runs, develop locally |
| Resume data leaks | Keep the resume and profile out of public repos and shared dashboards |

## 10. To verify in the docs before relying on them

These came from memory or third-party pages, so confirm them:

- Exact metric names and parameters for Argument Correctness, MCP Use and Task Completion in the installed DeepEval version
- Field names for goldens and tags in the Datasets page
- The LangGraph tracing integration steps
- How to plug in a custom Groq judge model
- Current free-plan limits for Groq, Firecrawl and Confident AI
- The current Groq model names available on the account

## 11. Interview story

“I built a job-search agent on LangGraph with real MCP servers. I evaluated it with DeepEval end to end and step by step, using task completion, tool correctness and argument correctness, plus a custom G-Eval for match quality that I checked against my own labels. I traced every run, ran the evals as pytest tests in CI, and monitored real runs on Confident AI. When I changed a prompt, I compared versions with a paired test and checked each category, so I only claimed gains that were bigger than the noise.”
