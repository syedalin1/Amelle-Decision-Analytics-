# ============================================================
# AMELLE DECISION ANALYTICS APP - VERSION 1
# ============================================================
# Purpose:
# A simple, generic business analytics app for students and
# business users.
#
# Workflow:
# 1. Understand My Data
# 2. Clean My Data
# 3. Visualise My Data
# 4. Analyse My Data
# 5. AI Decision Support
#
# IMPORTANT:
# Version 1 does NOT automatically run predictive machine learning.
# The human user remains responsible for the final business decision.
# ============================================================

import io
import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# SciPy is used for simple statistical tests in Level 4.
from scipy import stats

# Google Gemini is optional until the user reaches Level 5.
# The rest of the app can still work without an API key.
try:
    from google import genai
except ImportError:
    genai = None


# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="Amelle Decision Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Amelle Decision Analytics")
st.caption("Understand → Clean → Visualise → Analyse → Decide")

st.warning(
    "Privacy: Do not upload personal, sensitive, confidential, or "
    "commercially restricted information. Use anonymised or teaching data."
)


# ------------------------------------------------------------
# HELPER FUNCTIONS
# These functions do the reusable work behind the app.
# ------------------------------------------------------------

def make_unique_columns(columns):
    """Make duplicate column names unique."""
    seen = {}
    result = []

    for col in columns:
        name = str(col).strip() or "Unnamed_Column"
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
    return result


@st.cache_data(show_spinner=False)
def load_file(file_name, file_bytes):
    """
    Read CSV or Excel.
    For Excel V1 uses the first worksheet.
    """
    name = file_name.lower()
    file_buffer = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        return pd.read_csv(file_buffer)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_buffer, sheet_name=0)

    raise ValueError("Please upload a CSV, XLSX or XLS file.")


def prepare_dataframe(df):
    """Basic safe preparation before diagnosis."""
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    return df


def column_role(series):
    """
    Give each column a simple role.
    This is deliberately understandable rather than 'magical'.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/time"

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    non_null = series.dropna()
    if len(non_null) == 0:
        return "Empty"

    unique_ratio = non_null.nunique() / max(len(non_null), 1)

    if unique_ratio > 0.95:
        return "Text / possible ID"

    return "Categorical / text"


def find_case_space_inconsistencies(series):
    """
    Look for text categories that become identical after:
    - removing leading/trailing spaces
    - ignoring capitalisation

    Example:
    Cappuccino, cappuccino, " Cappuccino "
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return []

    values = series.dropna().astype(str)
    groups = {}

    for value in values.unique():
        normal = value.strip().casefold()
        groups.setdefault(normal, set()).add(value)

    return [sorted(list(v)) for v in groups.values() if len(v) > 1]


def numeric_outlier_count(series):
    """
    Flag potential numerical outliers using the IQR rule.
    This does NOT mean the values are wrong.
    It only tells the analyst what may deserve review.
    """
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) < 4:
        return 0

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((x < lower) | (x > upper)).sum())


def diagnose_data(df):
    """
    Level 1:
    Understand the data WITHOUT changing it.
    """
    summary_rows = []

    for col in df.columns:
        s = df[col]
        summary_rows.append({
            "Column": col,
            "Role": column_role(s),
            "Python type": str(s.dtype),
            "Missing": int(s.isna().sum()),
            "Unique values": int(s.nunique(dropna=True)),
            "Potential outliers": (
                numeric_outlier_count(s)
                if pd.api.types.is_numeric_dtype(s)
                else ""
            ),
        })

    inconsistencies = {}
    for col in df.columns:
        issues = find_case_space_inconsistencies(df[col])
        if issues:
            inconsistencies[col] = issues

    return pd.DataFrame(summary_rows), inconsistencies


def standardise_text(series):
    """
    Conservative text cleaning:
    - remove leading/trailing spaces
    - collapse repeated spaces
    - preserve the user's wording/capitalisation otherwise

    We do NOT automatically convert everything to Title Case because
    values such as 'iPhone', 'eBay' or codes could be damaged.
    """
    def clean_value(x):
        if pd.isna(x):
            return x
        x = str(x).strip()
        x = re.sub(r"\s+", " ", x)
        return x

    return series.map(clean_value)


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def dataframe_to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    return buffer.getvalue()


def store_clean_dataframe(df):
    """Store cleaned data and prepare its downloads only when the data changes."""
    st.session_state.clean_df = df
    st.session_state.clean_csv_bytes = dataframe_to_csv_bytes(df)
    st.session_state.clean_excel_bytes = dataframe_to_excel_bytes(df)


def recommend_chart(df, x_col, y_col=None):
    """
    Recommend one of four simple chart types:
    bar, line, scatter, pie.
    """
    x = df[x_col]

    if y_col is None:
        if x.nunique(dropna=True) <= 6:
            return "Pie"
        return "Bar"

    y = df[y_col]
    x_is_numeric = pd.api.types.is_numeric_dtype(x)
    y_is_numeric = pd.api.types.is_numeric_dtype(y)

    if x_is_numeric and y_is_numeric:
        return "Scatter"

    if pd.api.types.is_datetime64_any_dtype(x) and pd.api.types.is_numeric_dtype(y):
        return "Line"

    if x_is_numeric != y_is_numeric:
        return "Bar"

    return "Bar"


def create_chart(df, chart_type, x_col, y_col=None):
    """
    Create one simple matplotlib chart.
    Returns the figure plus a short explanation of what was plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_axis_label = x_col

    if chart_type == "Bar":
        if y_col is None:
            data = df[x_col].value_counts(dropna=False).head(20)
            data.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            explanation = f"Count of records by {x_col}."
        else:
            temp = df[[x_col, y_col]].dropna()

            x_is_numeric = pd.api.types.is_numeric_dtype(temp[x_col])
            y_is_numeric = pd.api.types.is_numeric_dtype(temp[y_col])

            if x_is_numeric != y_is_numeric:
                # For a numerical/categorical pair, always put the category on
                # the horizontal axis and aggregate the numerical values. This
                # works regardless of the order in which the user selected them.
                numeric_col = x_col if x_is_numeric else y_col
                category_col = y_col if x_is_numeric else x_col
                grouped = (
                    temp.groupby(category_col)[numeric_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(20)
                )
                x_axis_label = category_col
            else:
                grouped = (
                    temp.groupby(x_col)[y_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(20)
                )
                numeric_col = y_col
                category_col = x_col

            grouped.plot(kind="bar", ax=ax)
            ax.set_ylabel(f"Average {numeric_col}")
            explanation = f"Average {numeric_col} for each {category_col} category."

    elif chart_type == "Line":
        if y_col is None:
            raise ValueError("A line chart needs both an X and Y variable.")

        temp = df[[x_col, y_col]].dropna().copy()
        temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
        temp = temp.dropna().sort_values(x_col)

        # Aggregate duplicate dates/times by mean.
        grouped = temp.groupby(x_col, as_index=False)[y_col].mean()
        ax.plot(grouped[x_col], grouped[y_col], marker="o")
        ax.set_ylabel(y_col)
        explanation = f"Trend of {y_col} across {x_col}."

    elif chart_type == "Scatter":
        if y_col is None:
            raise ValueError("A scatter chart needs both an X and Y variable.")

        temp = df[[x_col, y_col]].dropna()
        ax.scatter(temp[x_col], temp[y_col], alpha=0.7)
        ax.set_ylabel(y_col)
        explanation = f"Relationship between {x_col} and {y_col}."

    elif chart_type == "Pie":
        data = df[x_col].value_counts(dropna=False).head(6)
        ax.pie(data.values, labels=data.index.astype(str), autopct="%1.1f%%")
        ax.set_ylabel("")
        explanation = f"Share of records across the main {x_col} categories."

    else:
        raise ValueError("Unsupported chart type.")

    ax.set_xlabel(x_axis_label)
    ax.set_title(f"{chart_type} chart: {x_col}" + (f" vs {y_col}" if y_col else ""))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig, explanation


def recommend_analysis(df, outcome, predictor):
    """
    Level 4:
    Recommend a simple analysis based on the variable types.
    The user must approve before it runs.
    """
    a = df[outcome]
    b = df[predictor]

    a_num = pd.api.types.is_numeric_dtype(a)
    b_num = pd.api.types.is_numeric_dtype(b)

    if a_num and b_num:
        return (
            "Correlation",
            f"Both '{outcome}' and '{predictor}' are numerical, so a correlation "
            "can measure the strength and direction of their association."
        )

    if a_num and not b_num:
        return (
            "Group comparison",
            f"'{outcome}' is numerical and '{predictor}' is categorical/text, so "
            f"we can compare average {outcome} across {predictor} groups."
        )

    if not a_num and b_num:
        return (
            "Group comparison",
            f"'{predictor}' is numerical and '{outcome}' is categorical/text, so "
            f"we can compare average {predictor} across {outcome} groups."
        )

    return (
        "Cross-tabulation",
        f"Both variables are categorical/text, so a cross-tabulation can show "
        f"how '{outcome}' and '{predictor}' occur together."
    )


def run_analysis(df, outcome, predictor, method):
    """
    Run the approved simple analysis and return:
    - a human-readable summary
    - structured evidence that can later be sent to AI
    """
    temp = df[[outcome, predictor]].dropna().copy()

    if len(temp) == 0:
        return "No complete rows are available for this analysis.", {}

    if method == "Correlation":
        x = pd.to_numeric(temp[outcome], errors="coerce")
        y = pd.to_numeric(temp[predictor], errors="coerce")
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]

        if len(x) < 3:
            return "Not enough numerical observations for correlation.", {}

        r, p = stats.pearsonr(x, y)

        evidence = {
            "method": "Pearson correlation",
            "outcome": outcome,
            "predictor": predictor,
            "observations": int(len(x)),
            "correlation_r": round(float(r), 4),
            "p_value": round(float(p), 6),
        }

        summary = (
            f"Pearson correlation between {outcome} and {predictor}: "
            f"r = {r:.3f}, p = {p:.4f}, based on {len(x)} complete observations. "
            "Correlation describes association, not causation."
        )
        return summary, evidence

    if method == "Group comparison":
        # Work out which variable is numeric.
        if pd.api.types.is_numeric_dtype(temp[outcome]):
            numeric_col, group_col = outcome, predictor
        else:
            numeric_col, group_col = predictor, outcome

        grouped = (
            temp.groupby(group_col)[numeric_col]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
        )

        evidence = {
            "method": "Group comparison",
            "numeric_variable": numeric_col,
            "group_variable": group_col,
            "groups": grouped.reset_index().round(3).to_dict(orient="records")[:20],
        }

        summary = (
            f"Compared {numeric_col} across {group_col} groups. "
            "The table shows count, mean and median. Differences are descriptive "
            "and should not automatically be treated as causal."
        )

        return summary, evidence, grouped

    # Cross-tabulation
    table = pd.crosstab(temp[outcome], temp[predictor])
    evidence = {
        "method": "Cross-tabulation",
        "row_variable": outcome,
        "column_variable": predictor,
        "table": table.reset_index().to_dict(orient="records")[:30],
    }
    summary = (
        f"Cross-tabulation of {outcome} and {predictor}. "
        "This shows how the two categories occur together."
    )
    return summary, evidence, table


def create_business_interpretation(business_question, evidence):
    """Explain an analysis result in short, non-technical business language."""
    question = business_question.strip()
    answer_prefix = f"Answer to '{question}': "

    if not evidence:
        return (
            answer_prefix
            + "the available complete data is not sufficient to answer this "
            "question reliably."
        )

    method = evidence.get("method")

    if method == "Pearson correlation":
        r = evidence.get("correlation_r")
        p = evidence.get("p_value")
        outcome = evidence.get("outcome")
        predictor = evidence.get("predictor")

        if r is None or not np.isfinite(r):
            return answer_prefix + "the data does not provide a usable correlation result."

        magnitude = abs(r)

        if magnitude < 0.1:
            relationship = (
                f"there is essentially no relationship between {predictor} "
                f"and {outcome}."
            )

            if p is not None and np.isfinite(p) and p < 0.05:
                significance = (
                    "The result may be statistically detectable, but the "
                    "relationship is too small to be meaningful in practical terms."
                )
            else:
                significance = "The data does not show a meaningful relationship."

            return f"{answer_prefix}{relationship} {significance}"

        strength = "weak" if magnitude < 0.3 else "moderate" if magnitude < 0.7 else "strong"

        if r > 0:
            relationship = (
                f"there is a {strength} positive relationship: higher {predictor} "
                f"tends to occur with higher {outcome}."
            )
        elif r < 0:
            relationship = (
                f"there is a {strength} negative relationship: higher {predictor} "
                f"tends to occur with lower {outcome}."
            )
        else:
            relationship = f"there is no linear relationship between {predictor} and {outcome}."

        if p is not None and np.isfinite(p) and p < 0.05:
            significance = "The data provides statistical evidence that this relationship exists."
        else:
            significance = "The data does not provide strong statistical evidence that this relationship exists."

        return f"{answer_prefix}{relationship} {significance} This does not prove causation."

    if method == "Group comparison":
        groups = evidence.get("groups", [])
        numeric_variable = evidence.get("numeric_variable")
        group_variable = evidence.get("group_variable")

        if not groups:
            return answer_prefix + "there are no complete groups available to compare."

        highest = groups[0]
        lowest = groups[-1]
        highest_name = highest.get(group_variable)
        lowest_name = lowest.get(group_variable)
        highest_mean = highest.get("mean")
        lowest_mean = lowest.get("mean")

        if len(groups) == 1:
            comparison = (
                f"only the {highest_name} group was available, with an average "
                f"{numeric_variable} of {highest_mean:,.2f}."
            )
        else:
            comparison = (
                f"average {numeric_variable} was highest for {highest_name} "
                f"({highest_mean:,.2f}) and lowest for {lowest_name} "
                f"({lowest_mean:,.2f})."
            )

        return (
            f"{answer_prefix}{comparison} The groups differ in this dataset, "
            "but the comparison alone does not show what caused the difference."
        )

    if method == "Cross-tabulation":
        rows = evidence.get("table", [])
        row_variable = evidence.get("row_variable")
        column_variable = evidence.get("column_variable")
        largest = None

        for row in rows:
            row_value = row.get(row_variable)
            for column_value, count in row.items():
                if column_value == row_variable or not isinstance(
                    count, (int, float, np.integer, np.floating)
                ):
                    continue
                if largest is None or count > largest[2]:
                    largest = (row_value, column_value, count)

        if largest is None:
            return answer_prefix + "there are no complete category combinations to compare."

        return (
            f"{answer_prefix}the most common combination was {row_variable} = "
            f"{largest[0]} and {column_variable} = {largest[1]} "
            f"({largest[2]:,} records). This describes the pattern in the data; "
            "it does not show that one category causes the other."
        )

    return answer_prefix + "the result should be reviewed alongside the statistical evidence above."


def get_gemini_client():
    """
    Create the Gemini client only when Level 5 is used.
    Recommended deployment method: put GEMINI_API_KEY in Streamlit Secrets.
    """
    if genai is None:
        return None, "The 'google-genai' package is not installed."

    api_key = None

    # First try Streamlit secrets.
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        pass

    # Then try a normal environment variable.
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None, "No Gemini API key has been configured."

    return genai.Client(api_key=api_key), None


def ask_ai(business_question, evidence):
    """
    Level 5:
    AI interprets ONLY the business question and analytical evidence supplied.
    It should not pretend that association proves causation.
    """
    client, error = get_gemini_client()
    if error:
        return None, error

    instructions = """
You are a cautious business analytics decision-support assistant.

Use only the analytical evidence supplied by the application.

Structure the answer under exactly four headings:
1. Finding
2. Interpretation
3. Possible actions
4. Limitations

Rules:
- Be concise and practical.
- Do not invent facts.
- Do not claim causation from correlation or descriptive comparisons.
- Clearly separate evidence from possible managerial action.
- The human user makes the final decision.
"""

    prompt = f"""
Business question:
{business_question}

Analytical evidence:
{json.dumps(evidence, indent=2, default=str)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={
                "system_instruction": instructions,
                "temperature": 0.2,
            },
        )
    except Exception as e:
        return None, f"Gemini could not generate a response: {e}"

    if not response.text:
        return None, "Gemini returned an empty response."

    return response.text, None


def _ask_data_tokens(text):
    """Return simple singularised tokens for local question/column matching."""
    tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
    result = []
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.append(token)
    return result


def _ask_data_matched_columns(df, question):
    """Match question wording to column names without sending data to AI."""
    question_tokens = set(_ask_data_tokens(question))
    matches = []

    for position, column in enumerate(df.columns):
        column_tokens = set(_ask_data_tokens(column))
        overlap = question_tokens & column_tokens
        if overlap:
            coverage = len(overlap) / max(len(column_tokens), 1)
            matches.append((coverage, len(overlap), -position, column))

    return [item[-1] for item in sorted(matches, reverse=True)]


def _ask_data_date_series(df, column):
    """Return a parsed date series when a column is suitable for trend analysis."""
    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    name_tokens = set(_ask_data_tokens(column))
    date_words = {"date", "time", "day", "week", "month", "year"}
    if not (name_tokens & date_words):
        return None

    if pd.api.types.is_numeric_dtype(series) and "year" in name_tokens:
        parsed = pd.to_datetime(series.astype("Int64").astype(str), format="%Y", errors="coerce")
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        return None

    if parsed.notna().mean() < 0.7:
        return None
    return parsed


def _ask_data_aggregation(column):
    """Choose a conservative business aggregation from the measure name."""
    additive_words = {
        "sale", "revenue", "profit", "quantity", "unit", "volume",
        "amount", "cost", "income", "order", "spend",
    }
    return "sum" if set(_ask_data_tokens(column)) & additive_words else "mean"


def analyse_ask_my_data(df, question):
    """Interpret a question locally and calculate compact analytical evidence."""
    question_tokens = set(_ask_data_tokens(question))
    matched = _ask_data_matched_columns(df, question)
    numeric_columns = [
        column for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    date_columns = [
        column for column in df.columns
        if _ask_data_date_series(df, column) is not None
    ]
    category_columns = [
        column for column in df.columns
        if column not in date_columns
        and not pd.api.types.is_numeric_dtype(df[column])
        and 1 < df[column].nunique(dropna=True) <= 200
    ]

    if not numeric_columns:
        raise ValueError("This question needs at least one numerical column in the cleaned data.")

    ranking_words = {"best", "worst", "top", "bottom", "highest", "lowest", "rank", "perform", "most", "least"}
    trend_words = {"trend", "time", "daily", "weekly", "monthly", "yearly", "season", "over"}
    comparison_words = {"compare", "comparison", "difference", "versus", "vs", "increase", "decrease", "higher", "lower"}
    relationship_words = {"relationship", "correlation", "correlate", "associated", "association", "affect", "factor", "driver", "influence"}

    if question_tokens & trend_words:
        analysis_type = "Trend"
    elif question_tokens & ranking_words:
        analysis_type = "Ranking"
    elif question_tokens & comparison_words:
        analysis_type = "Comparison"
    elif question_tokens & relationship_words:
        analysis_type = "Relationship / correlation"
    else:
        matched_numeric = [column for column in matched if column in numeric_columns]
        matched_categories = [column for column in matched if column in category_columns]
        if matched_numeric and matched_categories:
            analysis_type = "Comparison"
        elif len(matched_numeric) >= 2:
            analysis_type = "Relationship / correlation"
        else:
            raise ValueError(
                "I could not determine whether this is a ranking, comparison, "
                "relationship or trend question. Please mention the relevant columns."
            )

    preferred_measures = [
        column for column in matched if column in numeric_columns
    ]
    if not preferred_measures:
        business_measure_words = {
            "sale", "revenue", "profit", "quantity", "amount", "value", "score"
        }
        preferred_measures = sorted(
            numeric_columns,
            key=lambda column: bool(set(_ask_data_tokens(column)) & business_measure_words),
            reverse=True,
        )

    measure = preferred_measures[0]

    if analysis_type == "Ranking":
        groups = [column for column in matched if column in category_columns]
        if not groups:
            preferred_group_words = {"product", "item", "category", "type", "region", "channel", "drink"}
            groups = sorted(
                category_columns,
                key=lambda column: bool(set(_ask_data_tokens(column)) & preferred_group_words),
                reverse=True,
            )
        if not groups:
            raise ValueError("A ranking question needs a categorical column to rank.")

        group = groups[0]
        aggregation = _ask_data_aggregation(measure)
        temp = df[[group, measure]].dropna()
        if temp.empty:
            raise ValueError("No complete rows are available for this ranking.")

        ranked = (
            temp.groupby(group)[measure]
            .agg(aggregation)
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name=f"{aggregation.title()} {measure}")
        )
        evidence = {
            "analysis_type": analysis_type,
            "python_method": f"Group by {group} and calculate the {aggregation} of {measure}",
            "rows_used": int(len(temp)),
            "group_column": group,
            "measure_column": measure,
            "aggregation": aggregation,
            "ranking": ranked.round(4).to_dict(orient="records"),
        }
        return evidence, ranked

    if analysis_type == "Comparison":
        groups = [column for column in matched if column in category_columns]
        groups += [
            column for column in matched
            if column in numeric_columns
            and column != measure
            and 1 < df[column].nunique(dropna=True) <= 10
        ]
        if not groups:
            groups = category_columns
        if not groups:
            raise ValueError("A comparison question needs a categorical or low-cardinality group column.")

        group = groups[0]
        temp = df[[group, measure]].dropna()
        if temp.empty:
            raise ValueError("No complete rows are available for this comparison.")

        comparison = (
            temp.groupby(group)[measure]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        evidence = {
            "analysis_type": analysis_type,
            "python_method": f"Compare the mean and median {measure} across {group} groups",
            "rows_used": int(len(temp)),
            "group_column": group,
            "measure_column": measure,
            "groups": comparison.round(4).to_dict(orient="records")[:20],
        }
        return evidence, comparison.head(20)

    if analysis_type == "Relationship / correlation":
        matched_numeric = [column for column in matched if column in numeric_columns]
        target = matched_numeric[0] if matched_numeric else measure
        id_words = {"id", "identifier", "index", "code"}

        if len(matched_numeric) >= 2:
            predictors = [column for column in matched_numeric if column != target]
        else:
            predictors = [
                column for column in numeric_columns
                if column != target
                and not (set(_ask_data_tokens(column)) & id_words)
            ]

        correlations = []
        for predictor in predictors:
            temp = df[[target, predictor]].dropna()
            if len(temp) < 3 or temp[target].nunique() < 2 or temp[predictor].nunique() < 2:
                continue
            r_value, p_value = stats.pearsonr(temp[target], temp[predictor])
            if np.isfinite(r_value):
                correlations.append({
                    "predictor": predictor,
                    "outcome": target,
                    "observations": int(len(temp)),
                    "correlation_r": round(float(r_value), 4),
                    "p_value": round(float(p_value), 6),
                })

        if not correlations:
            raise ValueError("No suitable numerical column pairs are available for correlation.")

        correlations.sort(key=lambda row: abs(row["correlation_r"]), reverse=True)
        correlations = correlations[:10]
        display = pd.DataFrame(correlations)
        evidence = {
            "analysis_type": analysis_type,
            "python_method": "Pearson correlations calculated locally and ranked by absolute strength",
            "outcome_column": target,
            "correlations": correlations,
            "caution": "Correlation describes association, not causation.",
        }
        return evidence, display

    # Trend
    dates = [column for column in matched if column in date_columns] or date_columns
    if not dates:
        raise ValueError("A trend question needs a date or time column.")

    date_column = dates[0]
    parsed_dates = _ask_data_date_series(df, date_column)
    temp = pd.DataFrame({date_column: parsed_dates, measure: df[measure]}).dropna()
    if len(temp) < 2:
        raise ValueError("Not enough complete dated observations are available for a trend.")

    span_days = (temp[date_column].max() - temp[date_column].min()).days
    if span_days > 365:
        period, grain = "M", "month"
    elif span_days > 90:
        period, grain = "W", "week"
    else:
        period, grain = "D", "day"

    aggregation = _ask_data_aggregation(measure)
    temp["Period"] = temp[date_column].dt.to_period(period).dt.to_timestamp()
    trend = temp.groupby("Period")[measure].agg(aggregation).sort_index().reset_index()
    if len(trend) < 2:
        raise ValueError("The dated observations do not cover enough periods for a trend.")

    first_value = float(trend[measure].iloc[0])
    last_value = float(trend[measure].iloc[-1])
    percentage_change = None if first_value == 0 else ((last_value - first_value) / abs(first_value)) * 100
    trend_evidence = trend.tail(36).copy()
    trend_evidence[measure] = trend_evidence[measure].round(4)
    evidence = {
        "analysis_type": analysis_type,
        "python_method": f"Calculate the {aggregation} {measure} by {grain}",
        "rows_used": int(len(temp)),
        "date_column": date_column,
        "measure_column": measure,
        "time_grain": grain,
        "first_period_value": round(first_value, 4),
        "last_period_value": round(last_value, 4),
        "percentage_change": None if percentage_change is None else round(percentage_change, 2),
        "trend_values": trend_evidence.to_dict(orient="records"),
    }
    return evidence, trend.tail(36)


def _level5_evidence_columns(evidence):
    """Extract the variables used by a Level 4 or Ask My Data result."""
    variable_keys = {
        "outcome", "predictor", "numeric_variable", "group_variable",
        "row_variable", "column_variable", "outcome_column", "group_column",
        "measure_column", "date_column",
    }
    return {
        value for key, value in evidence.items()
        if key in variable_keys and isinstance(value, str)
    }


def _level5_is_identifier(column):
    return bool(set(_ask_data_tokens(column)) & {"id", "identifier"})


def prepare_level5_evidence(analysis_evidence, df):
    """Select relevant evidence and reduce it to a safe, compact AI payload."""
    question = analysis_evidence.get("business_question", "")
    method = analysis_evidence.get("recommended_method", "Analysis")
    detailed = analysis_evidence.get("evidence", {})
    selection_note = None

    question_columns = _ask_data_matched_columns(df, question)
    selected_columns = _level5_evidence_columns(detailed)
    relevant_question_columns = set(question_columns[:4])

    # If the question clearly names at least two columns but the Level 4
    # selections do not use them, calculate question-relevant evidence locally.
    if (
        len(relevant_question_columns) >= 2
        and not relevant_question_columns.issubset(selected_columns)
    ):
        try:
            detailed, _ = analyse_ask_my_data(df, question)
        except ValueError:
            pass
        else:
            method = detailed["analysis_type"]
            selection_note = (
                "The evidence was recalculated locally because the variables "
                "named in the question did not match the Level 4 selections."
            )

    compact = {
        "business_question": question,
        "analysis_method": method,
    }
    if selection_note:
        compact["selection_note"] = selection_note

    scalar_keys = {
        "method", "python_method", "rows_used", "observations",
        "outcome", "predictor", "numeric_variable", "group_variable",
        "row_variable", "column_variable", "outcome_column", "group_column",
        "measure_column", "date_column", "aggregation", "time_grain",
        "correlation_r", "p_value", "first_period_value",
        "last_period_value", "percentage_change", "caution",
    }
    variable_keys = {
        "outcome", "predictor", "numeric_variable", "group_variable",
        "row_variable", "column_variable", "outcome_column", "group_column",
        "measure_column", "date_column",
    }
    for key in scalar_keys:
        value = detailed.get(key)
        if value is not None and not (
            key in variable_keys
            and isinstance(value, str)
            and _level5_is_identifier(value)
        ):
            compact[key] = value

    list_keys = ("groups", "ranking", "correlations", "trend_values")
    for key in list_keys:
        rows = detailed.get(key)
        if not isinstance(rows, list):
            continue

        limited_rows = rows[-10:] if key == "trend_values" else rows[:10]
        safe_rows = []
        for row in limited_rows:
            if not isinstance(row, dict):
                continue
            safe_row = {
                column: value for column, value in row.items()
                if not _level5_is_identifier(column)
            }
            if safe_row:
                safe_rows.append(safe_row)
        compact[key] = safe_rows

    # Cross-tabulations can contain thousands of cells. Retain only the ten
    # largest non-identifier combinations as useful analytical evidence.
    table = detailed.get("table")
    row_variable = detailed.get("row_variable")
    column_variable = detailed.get("column_variable")
    if (
        isinstance(table, list)
        and row_variable
        and column_variable
        and not _level5_is_identifier(row_variable)
        and not _level5_is_identifier(column_variable)
    ):
        combinations = []
        for row in table:
            row_value = row.get(row_variable)
            for column_value, count in row.items():
                if column_value == row_variable or not isinstance(
                    count, (int, float, np.integer, np.floating)
                ):
                    continue
                combinations.append({
                    row_variable: row_value,
                    column_variable: column_value,
                    "count": int(count),
                })
        combinations.sort(key=lambda item: item["count"], reverse=True)
        compact["top_combinations"] = combinations[:10]

    return compact


def level5_evidence_summary(compact_evidence):
    """Create a short human-readable summary without rendering raw JSON."""
    lines = []
    if compact_evidence.get("selection_note"):
        lines.append(compact_evidence["selection_note"])

    method = compact_evidence.get("analysis_method")
    if method:
        lines.append(f"Analysis used: {method}.")

    if "correlation_r" in compact_evidence:
        lines.append(
            f"{compact_evidence.get('outcome')} and "
            f"{compact_evidence.get('predictor')}: "
            f"r = {compact_evidence['correlation_r']}, "
            f"p = {compact_evidence.get('p_value')}, "
            f"using {compact_evidence.get('observations')} observations."
        )

    result_keys = (
        "groups", "ranking", "correlations", "trend_values", "top_combinations"
    )
    for key in result_keys:
        rows = compact_evidence.get(key, [])
        for row in rows[:10]:
            parts = []
            for column, value in row.items():
                text = str(value)
                if len(text) > 80:
                    text = text[:77] + "..."
                parts.append(f"{column}: {text}")
            lines.append("; ".join(parts) + ".")

    if not lines:
        lines.append("A compact analytical result is available for interpretation.")
    return lines[:12]


# ------------------------------------------------------------
# SESSION STATE
# Streamlit reruns the script after interactions, so we keep the
# working data in session_state.
# ------------------------------------------------------------
if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if "analysis_evidence" not in st.session_state:
    st.session_state.analysis_evidence = None

if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None

if "business_interpretation" not in st.session_state:
    st.session_state.business_interpretation = None

if (
    st.session_state.clean_df is not None
    and (
        "clean_csv_bytes" not in st.session_state
        or "clean_excel_bytes" not in st.session_state
    )
):
    store_clean_dataframe(st.session_state.clean_df)


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        # Replace the stored data when a different file is uploaded.
        file_signature = (uploaded_file.name, uploaded_file.size)

        if st.session_state.get("file_signature") != file_signature:
            df = prepare_dataframe(
                load_file(uploaded_file.name, uploaded_file.getvalue())
            )
            st.session_state.original_df = df
            store_clean_dataframe(df.copy())
            st.session_state.analysis_evidence = None
            st.session_state.analysis_summary = None
            st.session_state.business_interpretation = None
            st.session_state.file_signature = file_signature

        df = st.session_state.original_df

        st.success(
            f"Loaded {uploaded_file.name}: "
            f"{len(df):,} rows × {len(df.columns):,} columns."
        )

    except Exception as e:
        st.error(f"I could not read this file: {e}")


# Stop here until a dataset exists.
if st.session_state.original_df is None:
    st.info("Upload a teaching or anonymised business dataset to begin.")
    st.stop()


# ------------------------------------------------------------
# FIVE STAGES
# Tabs keep each stage separate and easy to teach.
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Understand My Data",
    "2️⃣ Clean My Data",
    "3️⃣ Visualise My Data",
    "4️⃣ Analyse My Data",
    "5️⃣ AI Decision Support",
])


# ============================================================
# LEVEL 1 - UNDERSTAND MY DATA
# ============================================================
with tab1:
    st.header("1. Understand My Data")
    st.write(
        "This stage diagnoses the dataset. It does **not** change any data."
    )

    df = st.session_state.original_df

    if st.button("Understand My Data", type="primary"):
        summary, inconsistencies = diagnose_data(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.subheader("Column health check")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if inconsistencies:
            st.subheader("Possible text inconsistencies")
            st.write(
                "These values may mean the same thing, but the analyst should review them."
            )
            for col, groups in inconsistencies.items():
                st.write(f"**{col}**")
                for group in groups[:10]:
                    st.write(" • " + " / ".join(group))
        else:
            st.success("No obvious capitalisation/spacing category conflicts were detected.")

        st.info(
            "Potential outliers are values worth reviewing. "
            "They are not automatically errors."
        )


# ============================================================
# LEVEL 2 - CLEAN MY DATA
# ============================================================
with tab2:
    st.header("2. Clean My Data")
    st.write(
        "Choose what should be changed. The app does not blindly clean everything."
    )

    working = st.session_state.clean_df.copy()

    duplicate_count = int(working.duplicated().sum())
    text_columns = working.select_dtypes(include=["object", "string"]).columns.tolist()
    missing_columns = working.columns[working.isna().any()].tolist()

    st.subheader("Cleaning options")

    remove_duplicates = st.checkbox(
        f"Remove exact duplicate rows ({duplicate_count} currently detected)"
    )

    standardise_columns = st.multiselect(
        "Remove unnecessary spaces from selected text columns",
        options=text_columns
    )

    missing_actions = {}

    if missing_columns:
        st.write("**Missing-value treatment**")
        st.caption(
            "Choose column by column. 'Leave unchanged' is deliberately the default."
        )

        for col in missing_columns:
            count = int(working[col].isna().sum())
            is_numeric = pd.api.types.is_numeric_dtype(working[col])

            options = ["Leave unchanged", "Drop rows with missing values"]

            if is_numeric:
                options += ["Fill with median", "Fill with mean"]
            else:
                options += ["Fill with 'Missing / Unknown'"]

            missing_actions[col] = st.selectbox(
                f"{col} — {count} missing",
                options,
                key=f"missing_{col}"
            )

    if st.button("Apply Selected Cleaning", type="primary"):
        cleaned = working.copy()

        if remove_duplicates:
            cleaned = cleaned.drop_duplicates()

        for col in standardise_columns:
            cleaned[col] = standardise_text(cleaned[col])

        for col, action in missing_actions.items():
            if action == "Drop rows with missing values":
                cleaned = cleaned.dropna(subset=[col])

            elif action == "Fill with median":
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

            elif action == "Fill with mean":
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())

            elif action == "Fill with 'Missing / Unknown'":
                cleaned[col] = cleaned[col].fillna("Missing / Unknown")

        store_clean_dataframe(cleaned)
        st.session_state.analysis_evidence = None
        st.session_state.analysis_summary = None
        st.session_state.business_interpretation = None

        st.success(
            f"Cleaning applied. Working dataset now has "
            f"{len(cleaned):,} rows × {len(cleaned.columns):,} columns."
        )

    st.subheader("Current cleaned dataset")
    st.dataframe(st.session_state.clean_df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "⬇️ Download cleaned CSV",
            data=st.session_state.clean_csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    with col_b:
        st.download_button(
            "⬇️ Download cleaned Excel",
            data=st.session_state.clean_excel_bytes,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.button("Reset Cleaning"):
        store_clean_dataframe(st.session_state.original_df.copy())
        st.session_state.analysis_evidence = None
        st.session_state.analysis_summary = None
        st.session_state.business_interpretation = None
        st.success("Cleaning has been reset to the originally uploaded data.")


# ============================================================
# LEVEL 3 - VISUALISE MY DATA
# ============================================================
with tab3:
    st.header("3. Visualise My Data")
    st.write(
        "Choose variables. The app recommends a chart, but you remain in control."
    )

    df = st.session_state.clean_df

    x_col = st.selectbox("Choose the main variable (X)", df.columns, key="viz_x")

    y_options = ["None — show counts/shares"] + list(df.columns)
    y_choice = st.selectbox("Choose a second variable (optional)", y_options, key="viz_y")
    y_col = None if y_choice.startswith("None") else y_choice

    recommended = recommend_chart(df, x_col, y_col)
    st.info(f"Recommended chart: **{recommended}**")

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Line", "Scatter", "Pie"],
        index=["Bar", "Line", "Scatter", "Pie"].index(recommended)
    )

    if st.button("Create Chart", type="primary"):
        try:
            fig, explanation = create_chart(df, chart_type, x_col, y_col)
            st.pyplot(fig)
            st.caption(explanation)
        except Exception as e:
            st.error(
                f"This chart cannot be created with the selected variables: {e}"
            )


# ============================================================
# LEVEL 4 - ANALYSE MY DATA
# ============================================================
with tab4:
    st.header("4. Analyse My Data")
    st.write(
        "State the business question, choose the variables, review the app's "
        "recommended analysis, and approve it before anything runs."
    )

    df = st.session_state.clean_df

    business_question = st.text_area(
        "What business question are you trying to answer?",
        placeholder="Example: Is temperature associated with coffee sales?"
    )

    outcome = st.selectbox(
        "Which variable represents the main outcome/result?",
        df.columns,
        key="analysis_outcome"
    )

    predictor_options = [c for c in df.columns if c != outcome]
    predictor = st.selectbox(
        "Which variable might help explain or compare the outcome?",
        predictor_options,
        key="analysis_predictor"
    )

    method, reason = recommend_analysis(df, outcome, predictor)

    st.subheader("Recommended analysis")
    st.write(f"**{method}**")
    st.caption(reason)

    approve = st.checkbox(
        "I have reviewed the recommendation and approve this analysis."
    )

    if st.button("Run Approved Analysis", type="primary", disabled=not approve):
        if not business_question.strip():
            st.error("Please enter a business question first.")
        else:
            result = run_analysis(df, outcome, predictor, method)

            # Correlation returns 2 values; table-based methods return 3.
            if len(result) == 2:
                summary, evidence = result
                table = None
            else:
                summary, evidence, table = result

            st.session_state.analysis_summary = summary
            st.session_state.business_interpretation = create_business_interpretation(
                business_question, evidence
            )
            st.session_state.analysis_evidence = {
                "business_question": business_question,
                "recommended_method": method,
                "evidence": evidence
            }

            st.success("Analysis complete.")
            st.write(summary)

            if table is not None:
                st.dataframe(table, use_container_width=True)

    if st.session_state.analysis_summary:
        st.subheader("Latest analytical result")
        st.write(st.session_state.analysis_summary)
        st.subheader("Business Interpretation")
        st.write(st.session_state.business_interpretation)


# ============================================================
# LEVEL 5 - AI DECISION SUPPORT
# ============================================================
with tab5:
    st.header("5. AI Decision Support")
    st.write(
        "AI interprets the analytical evidence. It does not replace the human decision-maker."
    )

    if st.session_state.analysis_evidence is None:
        st.info("Complete an approved analysis in Level 4 first.")
    else:
        st.subheader("Evidence that will be sent to AI")

        current_evidence = st.session_state.analysis_evidence
        evidence_signature = (
            current_evidence.get("business_question"),
            current_evidence.get("recommended_method"),
            tuple(sorted(_level5_evidence_columns(current_evidence.get("evidence", {})))),
            st.session_state.analysis_summary,
            id(st.session_state.clean_df),
        )
        if st.session_state.get("level5_evidence_signature") != evidence_signature:
            st.session_state.level5_compact_evidence = prepare_level5_evidence(
                current_evidence,
                st.session_state.clean_df,
            )
            st.session_state.level5_evidence_signature = evidence_signature

        compact_evidence = st.session_state.level5_compact_evidence
        st.write(f"**Business question:** {compact_evidence['business_question']}")
        for line in level5_evidence_summary(compact_evidence):
            st.write(f"• {line}")

        st.caption(
            "Only the business question and analytical evidence above are sent "
            "by this feature—not the full uploaded dataset."
        )

        if st.button("Ask AI for Decision Support", type="primary"):
            question = st.session_state.analysis_evidence["business_question"]

            answer, error = ask_ai(
                question,
                compact_evidence,
            )

            if error:
                st.error(error)
                st.info(
                    "Levels 1–4 still work without AI. "
                    "Add a Gemini API key later to activate Level 5."
                )
            else:
                st.markdown(answer)

        st.warning(
            "AI output is decision support, not a final decision. "
            "Check the evidence, context and limitations yourself."
        )

    st.divider()
    st.subheader("Ask My Data")
    st.write(
        "Ask a business question about the complete cleaned dataset. "
        "The app supports rankings, comparisons, relationships/correlations and trends."
    )
    st.caption(
        "Python analyses the cleaned data locally. Only your question and the "
        "calculated analytical evidence are sent to Gemini—not the raw dataset."
    )

    ask_data_question = st.text_area(
        "What would you like to ask your data?",
        placeholder=(
            "Example: Which products perform best? Does promotion appear to "
            "increase sales? What factors are related to total sales?"
        ),
        key="ask_my_data_question",
    )

    if st.button("Ask My Data", type="primary"):
        if not ask_data_question.strip():
            st.error("Please enter a business question first.")
        else:
            try:
                evidence, result_table = analyse_ask_my_data(
                    st.session_state.clean_df,
                    ask_data_question,
                )
            except ValueError as e:
                st.error(str(e))
            else:
                compact_ask_evidence = prepare_level5_evidence(
                    {
                        "business_question": ask_data_question,
                        "recommended_method": evidence["analysis_type"],
                        "evidence": evidence,
                    },
                    st.session_state.clean_df,
                )
                st.success(
                    f"Python analysis complete: {evidence['analysis_type']}."
                )
                st.caption(evidence["python_method"])
                display_table = (
                    result_table.tail(10)
                    if evidence["analysis_type"] == "Trend"
                    else result_table.head(10)
                )
                st.dataframe(display_table, use_container_width=True)

                answer, error = ask_ai(
                    ask_data_question,
                    compact_ask_evidence,
                )

                if error:
                    st.error(error)
                else:
                    st.subheader("AI explanation")
                    st.markdown(answer)
                    st.warning(
                        "AI output is decision support, not a final decision. "
                        "Check the evidence, context and limitations yourself."
                    )
