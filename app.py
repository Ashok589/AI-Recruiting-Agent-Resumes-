
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tools import extract_resume_text, save_candidate_record
from agent import evaluate_candidate


# Load API configuration from .env.
load_dotenv()

st.set_page_config(
    page_title="AI Recruiting Agent",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 AI Recruiting Agent")
st.caption(
    "Resume analysis, job matching, candidate search, "
    "and evidence-based recruiting support."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "candidate_result" not in st.session_state:
    st.session_state.candidate_result = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "candidate_saved" not in st.session_state:
    st.session_state.candidate_saved = False


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    st.write(
        "Upload a resume and enter a job description "
        "to evaluate job-related skills."
    )

    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API key detected")
    else:
        st.warning(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    st.info(
        "AI results are decision-support information. "
        "A human recruiter must verify the evidence and "
        "make the final employment decision."
    )


# --------------------------------------------------
# JOB DETAILS
# --------------------------------------------------

st.subheader("1. Job details")

job_title = st.text_input(
    "Job role",
    placeholder="e.g. Business Analyst, Python Developer, AI Engineer",
)

job_description = st.text_area(
    "Job description",
    height=200,
    placeholder=(
        "Enter the job responsibilities, required skills, "
        "preferred skills, and relevant experience."
    ),
)


# --------------------------------------------------
# RESUME UPLOAD
# --------------------------------------------------

st.subheader("2. Candidate resume")

uploaded_file = st.file_uploader(
    "Upload candidate resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT.",
)

if uploaded_file is not None:
    try:
        extracted_text = extract_resume_text(uploaded_file)

        st.session_state.resume_text = extracted_text
        st.success("Resume text extracted successfully.")

        with st.expander("Preview extracted resume text"):
            st.text(extracted_text[:5000])

    except (ValueError, Exception) as exc:
        st.session_state.resume_text = None
        st.error(f"Could not read the resume: {exc}")
else:
    st.session_state.resume_text = None


# --------------------------------------------------
# LOAD INTERNAL CANDIDATE DATABASE
# --------------------------------------------------

def load_candidates(csv_path="candidates.csv"):
    """Load saved candidate records from CSV."""
    if not os.path.exists(csv_path):
        return []

    try:
        dataframe = pd.read_csv(csv_path).fillna("")
        return dataframe.to_dict(orient="records")

    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read the candidate database: {exc}")
        return []

    except OSError as exc:
        st.warning(f"Could not open the candidate database: {exc}")
        return []


# --------------------------------------------------
# EVALUATE CANDIDATE
# --------------------------------------------------

st.subheader("3. Evaluate candidate")

if st.button("Evaluate Candidate", type="primary"):
    if not job_title.strip():
        st.error("Please enter the job role.")

    elif not job_description.strip():
        st.error("Please enter the job description.")

    elif not st.session_state.resume_text:
        st.error("Please upload a readable resume.")

    elif not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is missing. Check your .env file "
            "and restart Streamlit."
        )

    else:
        try:
            with st.spinner(
                "Analyzing the resume and matching job requirements..."
            ):
                candidates = load_candidates()

                result = evaluate_candidate(
                    resume_text=st.session_state.resume_text,
                    job_description=job_description,
                    candidates=candidates,
                )

                result["job_title"] = job_title.strip()

                st.session_state.candidate_result = result
                st.session_state.candidate_saved = False

        except Exception as exc:
            st.error(f"Candidate evaluation failed: {exc}")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.candidate_result

if result:
    st.divider()
    st.header("4. Candidate evaluation results")

    st.subheader(
        result.get("candidate_name") or "Unknown candidate"
    )

    st.write(
        "**Email:**",
        result.get("candidate_email") or "Unknown",
    )

    st.write(
        "**Job role:**",
        result.get("job_title", ""),
    )

    # Shortlisting recommendation.
    status = result.get(
        "shortlist_status",
        "Human review required",
    )

    st.subheader("Shortlisting recommendation")

    if status == "Shortlist for human review":
        st.success(f"**{status}**")

    elif status == "Needs human review":
        st.warning(f"**{status}**")

    elif status == "Insufficient information":
        st.info(f"**{status}**")

    else:
        st.warning(f"**{status}**")

    st.write(
        result.get(
            "shortlist_reason",
            result.get("reasoning", ""),
        )
    )

    # Skill-match score.
    skill_match = result.get("skill_match") or {}
    score = skill_match.get("score")

    st.subheader("Required-skill match")

    col1, col2, col3 = st.columns(3)

    if isinstance(score, (int, float)):
        col1.metric("Skill match", f"{score:.1f}%")

        col2.metric(
            "Matched skills",
            skill_match.get("matched_count", 0),
        )

        col3.metric(
            "Required skills",
            skill_match.get("required_count", 0),
        )

        st.progress(
            max(0.0, min(float(score) / 100.0, 1.0))
        )

    else:
        col1.metric("Skill match", "N/A")

        st.info(
            "A match percentage cannot be calculated because "
            "no identifiable required skills were found."
        )

    st.caption(
        skill_match.get(
            "message",
            "Skill-match details are unavailable.",
        )
    )

    # Matching skills.
    st.subheader("✅ Matching required skills")

    matched_skills = skill_match.get("matched_skills", [])

    if matched_skills:
        for item in matched_skills:
            if isinstance(item, dict):
                required = item.get("required_skill", "")
                resume_skill = item.get("resume_skill", "")

                st.write(
                    f"**{required}** — "
                    f"resume evidence skill: {resume_skill}"
                )
            else:
                st.write(f"• {item}")
    else:
        st.info("No required skills were matched.")

    # Missing or unverified skills.
    st.subheader("⚠️ Missing or unverified required skills")

    missing_skills = skill_match.get("missing_skills", [])

    if missing_skills:
        for skill in missing_skills:
            st.write(f"• {skill}")

        st.caption(
            "These skills were not identified in the extracted skill list. "
            "They may still be present in the resume and should be verified."
        )
    else:
        st.write("No missing required skills were identified.")

    # Candidate summary.
    st.subheader("Candidate summary")

    st.write(
        result.get("candidate_summary")
        or "No candidate summary was returned."
    )

    # Extracted skills.
    st.subheader("Skills identified in resume")

    skills = result.get("skills", [])

    if skills:
        st.write(", ".join(skills))
    else:
        st.info("No skills were extracted from the resume.")

    # Relevant experience.
    st.subheader("Relevant experience")

    experience = result.get("relevant_experience", [])

    if experience:
        for item in experience:
            st.write(f"• {item}")
    else:
        st.info("No relevant experience evidence was returned.")

    # Strengths.
    st.subheader("Strengths")

    strengths = result.get("strengths", [])

    if strengths:
        for item in strengths:
            st.write(f"✅ {item}")
    else:
        st.write("No strengths were identified.")

    # Evidence gaps.
    st.subheader("Evidence gaps")

    gaps = result.get("gaps", [])

    if gaps:
        for item in gaps:
            st.write(f"⚠️ {item}")
    else:
        st.write("No additional evidence gaps were identified.")

    # Reasoning.
    st.subheader("Assessment reasoning")

    st.write(
        result.get(
            "reasoning",
            result.get("recommendation_reasoning", ""),
        )
    )

    # Follow-up questions.
    st.subheader("Questions for the recruiter")

    questions = result.get("follow_up_questions", [])

    if questions:
        for question in questions:
            st.write(f"• {question}")
    else:
        st.write("No follow-up questions were returned.")

    # Similar candidates.
    st.subheader("Similar candidates in internal database")

    similar_candidates = result.get(
        "similar_candidates",
        result.get("internal_candidates", []),
    )

    if similar_candidates:
        st.dataframe(
            pd.DataFrame(similar_candidates),
            use_container_width=True,
        )
    else:
        st.info(
            "No matching candidate records were found in candidates.csv. "
            "The database will populate as you save candidate records."
        )

    # Method and disclaimer.
    st.caption(
        result.get(
            "scoring_method",
            "The score measures skill overlap only.",
        )
    )

    st.warning(
        result.get(
            "disclaimer",
            "A human recruiter must verify the evidence and make the "
            "final employment decision.",
        )
    )

    # Download JSON.
    st.subheader("Download evaluation")

    json_output = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        label="Download evaluation as JSON",
        data=json_output,
        file_name="candidate_evaluation.json",
        mime="application/json",
        key="download_candidate_evaluation",
    )

    # Save candidate.
    st.subheader("Save candidate to internal database")

    if st.button(
        "Save candidate",
        key="save_candidate",
        disabled=st.session_state.candidate_saved,
    ):
        try:
            record = {
                "name": result.get("candidate_name", "Unknown"),
                "email": result.get("candidate_email", "Unknown"),
                "role": result.get("job_title", ""),
                "skills": ", ".join(result.get("skills", [])),
                "experience": "; ".join(
                    result.get("relevant_experience", [])
                ),
            }

            saved_path = save_candidate_record(
                record,
                csv_path="candidates.csv",
            )

            st.session_state.candidate_saved = True
            st.success(f"Candidate saved to {saved_path}")

        except Exception as exc:
            st.error(f"Could not save candidate: {exc}")

    if st.session_state.candidate_saved:
        st.success("This candidate has been saved.")

    # Full result for debugging.
    with st.expander("View complete JSON output"):
        st.json(result)
