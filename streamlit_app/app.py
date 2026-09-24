"""
NLP Document Intelligence Gateway — Interactive Streamlit Dashboard
Multi-tab analytics workbench for customer support ticket classification,
semantic search, entity extraction, and document processing.
"""

import os
import streamlit as st
import requests

# Set page configuration with custom icon and wide layout
st.set_page_config(
    page_title="NLP Document Intelligence Gateway",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic & Modern Styling
st.markdown(
    """
    <style>
        /* Modern Fonts and Base Theme */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
            padding: 24px 32px;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-subtitle {
            font-size: 1.05rem;
            color: #c7d2fe;
            margin-top: 8px;
            margin-bottom: 0;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }

        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-qdrant { background-color: #dbeafe; color: #1e40af; }
        .badge-rules { background-color: #dcfce7; color: #166534; }
        .badge-llm { background-color: #fef3c7; color: #92400e; }
        .badge-local_nlp { background-color: #dcfce7; color: #166534; }
        .badge-llm_fallback { background-color: #fef3c7; color: #92400e; }

        .entity-chip {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 8px;
            margin: 4px 4px 4px 0;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid rgba(0,0,0,0.08);
        }
        .entity-money { background-color: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
        .entity-order { background-color: #fffbeb; color: #92400e; border-color: #fde68a; }
        .entity-email { background-color: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
        .entity-date { background-color: #f5f3ff; color: #5b21b6; border-color: #ddd6fe; }
        .entity-misc { background-color: #f1f5f9; color: #334155; border-color: #cbd5e1; }

        .keyword-chip {
            display: inline-block;
            background: #f8fafc;
            color: #475569;
            border: 1px solid #e2e8f0;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            margin: 3px;
        }

        .evidence-quote {
            background-color: #f8fafc;
            border-left: 4px solid #6366f1;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: #334155;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
def probe_gateway(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def resolve_gateway_url() -> str:
    """Auto-detect working Gateway endpoint across Docker and host environments."""
    if st.session_state.get("gateway_url"):
        cached = st.session_state["gateway_url"].rstrip("/")
        if probe_gateway(cached):
            return cached

    candidates = []
    env_gw = os.getenv("GATEWAY_URL")
    if env_gw:
        candidates.append(env_gw.strip().rstrip("/"))

    is_docker = os.path.exists("/.dockerenv")
    if is_docker:
        candidates.extend(
            [
                "http://gateway:8000",
                "http://host.docker.internal:8000",
                "http://localhost:8000",
            ]
        )
    else:
        candidates.extend(
            [
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://gateway:8000",
            ]
        )

    for cand in candidates:
        if cand and probe_gateway(cand):
            st.session_state["gateway_url"] = cand
            return cand

    fallback = (
        candidates[0]
        if candidates
        else ("http://gateway:8000" if is_docker else "http://localhost:8000")
    )
    st.session_state["gateway_url"] = fallback
    return fallback


with st.sidebar:
    api_url = resolve_gateway_url()

    # --------------------------------------------------------------------------
    # AI Model Configuration (Embedding & Reasoning)
    # --------------------------------------------------------------------------
    st.markdown("### 🤖 AI Model Configuration")

    DEFAULT_ST_MODELS = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]
    DEFAULT_GEMINI_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]

    def fetch_model_catalog(target_url: str):
        try:
            r = requests.get(f"{target_url}/models", timeout=4)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    if (
        "model_catalog" not in st.session_state
        or st.session_state["model_catalog"] is None
    ):
        st.session_state["model_catalog"] = fetch_model_catalog(api_url)

    catalog = st.session_state.get("model_catalog")
    current_cfg = catalog.get("current_config", {}) if catalog else {}

    # --- Embedding Model Selection ---
    st.markdown("**Embedding Provider**")
    emb_providers = ["Sentence Transformer", "Ollama"]
    current_emb_prov = current_cfg.get("embedding_provider", "sentence_transformer")
    default_emb_idx = 1 if current_emb_prov == "ollama" else 0
    selected_emb_prov_label = st.selectbox(
        "Embedding Provider",
        emb_providers,
        index=default_emb_idx,
        key="sb_emb_prov",
        label_visibility="collapsed",
    )
    selected_emb_prov = (
        "ollama" if selected_emb_prov_label == "Ollama" else "sentence_transformer"
    )

    if selected_emb_prov == "sentence_transformer":
        st_options = (
            catalog.get("embedding_providers", {}).get("sentence_transformer", [])
            if catalog
            else []
        )
        if not st_options:
            st_options = DEFAULT_ST_MODELS

        current_emb_model = current_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        default_idx = (
            st_options.index(current_emb_model)
            if current_emb_model in st_options
            else 0
        )
        selected_emb_model = st.selectbox(
            "Sentence Transformer Model",
            st_options,
            index=default_idx,
            key="sb_emb_st_model",
        )
    else:
        ollama_stat = catalog.get("ollama_status", {}) if catalog else {}
        ollama_emb_options = (
            catalog.get("embedding_providers", {}).get("ollama", []) if catalog else []
        )
        if not ollama_stat.get("reachable"):
            st.caption(
                f"⚠️ Ollama offline ({ollama_stat.get('base_url', 'http://localhost:11434')})"
            )
        elif not ollama_emb_options:
            st.caption(
                "ℹ️ No Ollama embedding models found. Run: `ollama pull nomic-embed-text`"
            )

        if ollama_emb_options:
            current_emb_model = current_cfg.get("embedding_model", "")
            default_idx = (
                ollama_emb_options.index(current_emb_model)
                if current_emb_model in ollama_emb_options
                else 0
            )
            selected_emb_model = st.selectbox(
                "Ollama Embedding Model",
                ollama_emb_options,
                index=default_idx,
                key="sb_emb_ollama_model",
            )
        else:
            selected_emb_model = st.text_input(
                "Ollama Embedding Model",
                value=current_cfg.get("embedding_model", "") or "",
                placeholder="e.g. nomic-embed-text",
                key="sb_emb_ollama_input",
            )

    # --- Reasoning Model Selection ---
    st.markdown("**Reasoning Provider**")
    reas_providers = ["Gemini", "Ollama"]
    current_reas_prov = current_cfg.get("reasoning_provider", "gemini")
    default_reas_idx = 1 if current_reas_prov == "ollama" else 0
    selected_reas_prov_label = st.selectbox(
        "Reasoning Provider",
        reas_providers,
        index=default_reas_idx,
        key="sb_reas_prov",
        label_visibility="collapsed",
    )
    selected_reas_prov = "ollama" if selected_reas_prov_label == "Ollama" else "gemini"

    if selected_reas_prov == "gemini":
        gemini_options = (
            catalog.get("reasoning_providers", {}).get("gemini", []) if catalog else []
        )
        if not gemini_options:
            gemini_options = DEFAULT_GEMINI_MODELS

        current_reas_model = current_cfg.get("reasoning_model", "gemini-1.5-flash")
        default_idx = (
            gemini_options.index(current_reas_model)
            if current_reas_model in gemini_options
            else 0
        )
        selected_reas_model = st.selectbox(
            "Gemini Reasoning Model",
            gemini_options,
            index=default_idx,
            key="sb_reas_gemini_model",
        )
    else:
        ollama_stat = catalog.get("ollama_status", {}) if catalog else {}
        ollama_reas_options = (
            catalog.get("reasoning_providers", {}).get("ollama", []) if catalog else []
        )
        if not ollama_stat.get("reachable"):
            st.caption(
                f"⚠️ Ollama offline ({ollama_stat.get('base_url', 'http://localhost:11434')})"
            )
        elif not ollama_reas_options:
            st.caption("ℹ️ No Ollama models found. Run: `ollama pull llama3.2`")

        if ollama_reas_options:
            current_reas_model = current_cfg.get("reasoning_model", "")
            default_idx = (
                ollama_reas_options.index(current_reas_model)
                if current_reas_model in ollama_reas_options
                else 0
            )
            selected_reas_model = st.selectbox(
                "Ollama Reasoning Model",
                ollama_reas_options,
                index=default_idx,
                key="sb_reas_ollama_model",
            )
        else:
            selected_reas_model = st.text_input(
                "Ollama Reasoning Model",
                value=current_cfg.get("reasoning_model", "") or "",
                placeholder="e.g. llama3.2, mistral, qwen2.5:7b",
                key="sb_reas_ollama_input",
            )

    # --- Action Buttons ---
    col_ref, col_save = st.columns(2)
    with col_ref:
        if st.button(
            "🔄 Refresh",
            use_container_width=True,
            help="Refresh available models from gateway",
        ):
            with st.spinner("Discovering models..."):
                cat = fetch_model_catalog(api_url)
                if cat:
                    st.session_state["model_catalog"] = cat
                    ollama_stat = cat.get("ollama_status", {})
                    if ollama_stat.get("reachable"):
                        st.success(
                            f"Found {len(ollama_stat.get('installed_models', []))} Ollama model(s)!"
                        )
                    else:
                        st.info("Ollama is offline. Local & Gemini models ready.")
                    st.rerun()
                else:
                    st.error("Gateway offline.")

    with col_save:
        if st.button(
            "💾 Save",
            use_container_width=True,
            help="Apply selected models",
            type="primary",
        ):
            payload = {
                "embedding_provider": selected_emb_prov,
                "embedding_model": selected_emb_model,
                "reasoning_provider": selected_reas_prov,
                "reasoning_model": selected_reas_model,
            }
            with st.spinner("Validating & updating..."):
                try:
                    res = requests.post(
                        f"{api_url}/models/config", json=payload, timeout=6
                    )
                    if res.status_code == 200:
                        st.success("✅ Models updated!")
                        st.session_state["model_catalog"] = fetch_model_catalog(api_url)
                        st.rerun()
                    else:
                        err_detail = res.json().get("detail", res.text)
                        st.error(f"❌ {err_detail}")
                except Exception as e:
                    st.error(f"Failed to update models: {e}")

    # Show Qdrant & Ollama status badges
    if catalog:
        q_stat = catalog.get("qdrant_status", {})
        q_mode = q_stat.get("mode", "memory")
        o_stat = catalog.get("ollama_status", {})
        o_reach = o_stat.get("reachable", False)

        st.caption(
            f"**Vector DB:** Qdrant (`{q_mode}`) | **Ollama:** {'🟢 Online' if o_reach else '🔴 Offline'}"
        )

    # Optional collapsed advanced gateway connection expander (clean & non-intrusive)
    with st.expander("⚙️ Connection Settings", expanded=False):
        custom_url = st.text_input(
            "Gateway URL",
            value=api_url,
            key="custom_api_url",
            help="Gateway endpoint. Use http://gateway:8000 in Docker or http://localhost:8000 on host.",
        )
        if custom_url and custom_url.strip():
            new_url = custom_url.strip().rstrip("/")
            if new_url != api_url:
                api_url = new_url
                st.session_state["gateway_url"] = api_url

        if st.button("Ping Gateway Health", use_container_width=True):
            try:
                r = requests.get(f"{api_url}/health", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"Online: {data.get('app_name', 'Gateway')} (v{data.get('version', '1.0.0')})"
                    )
                else:
                    st.error(f"HTTP {r.status_code}")
            except Exception as e:
                is_docker = os.path.exists("/.dockerenv") or os.getenv("GATEWAY_URL")
                if (
                    "localhost" in api_url
                    and is_docker
                    and probe_gateway("http://gateway:8000")
                ):
                    api_url = "http://gateway:8000"
                    st.session_state["gateway_url"] = api_url
                    st.warning(
                        "Inside Docker, 'localhost' refers to the frontend container itself. "
                        "Auto-switched to `http://gateway:8000` (Online!)."
                    )
                else:
                    st.error(f"Connection Failed: {e}")

    st.markdown("---")
    st.markdown("### 🚀 Sample Presets")
    preset = st.selectbox(
        "Load Sample Ticket",
        [
            "Select a preset...",
            "Duplicate Billing Charge",
            "Broken Item & Return",
            "Forgotten Password",
            "Missing Package Tracking",
            "Subscription Cancellation",
        ],
    )

preset_samples = {
    "Duplicate Billing Charge": (
        "Hello support, my Visa card was charged $49.99 twice for order #ORD-8812 on 2026-03-01. "
        "Please issue a refund to aniket@example.com immediately."
    ),
    "Broken Item & Return": (
        "The ceramic teapot in order #ORD-9912 arrived completely shattered. "
        "I want to return the damaged item and get a replacement sent."
    ),
    "Forgotten Password": (
        "I forgot my account password and cannot log into the mobile app. "
        "Please send a password reset link to support-user@company.com."
    ),
    "Missing Package Tracking": (
        "Where is my shipment? Tracking number TRK-88219 has been stuck in transit for 6 days. "
        "Can you check the delivery status?"
    ),
    "Subscription Cancellation": (
        "Please cancel my annual Pro tier subscription immediately. "
        "I do not want to be charged $120.00 next month."
    ),
}

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="header-title">
            <span>⚡ NLP Document Intelligence Gateway</span>
        </div>
        <p class="header-subtitle">
            Hybrid Multi-Tier Engine: Local spaCy/Regex ML + Qdrant Dense Vector Store + LLM Fallback
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Main Tabs Navigation
# ------------------------------------------------------------------------------
TAB_TITLES = [
    "🎫 Ticket Analyzer",
    "🔍 Semantic Similarity",
    "🏷️ Topic Taxonomy",
    "📄 Document Extractor",
    "🧠 Architecture Guide",
]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TAB_TITLES[0]

# Optional JS trigger to switch to tab 0 if button was pressed
if st.session_state.pop("switch_to_tab_0", False):
    st.components.v1.html(
        """
        <script>
            setTimeout(() => {
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs && tabs.length > 0) {
                    tabs[0].click();
                }
            }, 50);
        </script>
        """,
        height=0,
        width=0,
    )

try:
    tabs = st.tabs(
        TAB_TITLES,
        default=st.session_state["active_tab"],
        key="main_tabs",
    )
except TypeError:
    tabs = st.tabs(TAB_TITLES)

tab1, tab2, tab3, tab4, tab5 = tabs

# ------------------------------------------------------------------------------
# TAB 1: Ticket Analyzer
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Support Ticket Analysis")
    st.markdown(
        "Submit customer queries for real-time topic classification, intent detection, NER, and sentiment scoring."
    )

    initial_text = ""
    if preset in preset_samples:
        initial_text = preset_samples[preset]
    elif "extracted_text" in st.session_state and st.session_state["extracted_text"]:
        initial_text = st.session_state["extracted_text"]

    input_text = st.text_area(
        "Support Message Input",
        value=initial_text,
        height=180,
        placeholder="Paste customer email, chat transcript, or support request here...",
    )

    analyze_clicked = st.button(
        "🚀 Analyze Ticket", type="primary", use_container_width=True
    )

    if analyze_clicked and input_text.strip():
        with st.spinner("Processing through local NLP and vector pipeline..."):
            try:
                payload = {
                    "text": input_text.strip(),
                    "metadata": {"source": "streamlit_ui"},
                }
                res = requests.post(f"{api_url}/analyze", json=payload, timeout=15)

                if res.status_code == 200:
                    data = res.json()
                    st.success("Analysis Complete!")

                    # Key Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        topic_info = data.get("topic", {})
                        res_by = topic_info.get("resolved_by", "qdrant")
                        badge_class = f"badge-{res_by}"
                        t_name = topic_info.get("topic", "N/A")
                        t_conf = topic_info.get("confidence", 0.0)
                        st.markdown(
                            '<div class="metric-card">\n'
                            '  <div style="font-size:0.8rem; color:#64748b; font-weight:600;">DETECTED TOPIC</div>\n'
                            f'  <div style="font-size:1.25rem; font-weight:700; color:#1e293b; margin:4px 0;">'
                            f"{t_name}</div>\n"
                            f'  <div><span class="badge {badge_class}">{res_by.upper()}</span>\n'
                            f'  <span style="font-size:0.85rem; color:#64748b; margin-left:6px;">'
                            f"{t_conf:.1%}</span></div>\n"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    with m2:
                        intent_info = data.get("intent", {})
                        i_res_by = intent_info.get("resolved_by", "rules")
                        i_badge_class = f"badge-{i_res_by}"
                        i_name = intent_info.get("intent", "N/A")
                        i_conf = intent_info.get("confidence", 0.0)
                        st.markdown(
                            '<div class="metric-card">\n'
                            '  <div style="font-size:0.8rem; color:#64748b; font-weight:600;">PRIMARY INTENT</div>\n'
                            f'  <div style="font-size:1.25rem; font-weight:700; color:#1e293b; margin:4px 0;">'
                            f"{i_name}</div>\n"
                            f'  <div><span class="badge {i_badge_class}">{i_res_by.upper()}</span>\n'
                            f'  <span style="font-size:0.85rem; color:#64748b; margin-left:6px;">'
                            f"{i_conf:.1%}</span></div>\n"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    with m3:
                        sent_info = data.get("sentiment", {})
                        s_label = sent_info.get("sentiment", "neutral").upper()
                        s_score = sent_info.get("score", 0.0)
                        s_res_by = sent_info.get("resolved_by", "local_nlp")
                        s_badge_class = f"badge-{s_res_by}"
                        s_color = (
                            "#10b981"
                            if s_score > 0.05
                            else ("#ef4444" if s_score < -0.05 else "#64748b")
                        )
                        st.markdown(
                            '<div class="metric-card">\n'
                            '  <div style="font-size:0.8rem; color:#64748b; font-weight:600;">'
                            "SENTIMENT POLARITY</div>\n"
                            f'  <div style="font-size:1.25rem; font-weight:700; color:{s_color}; margin:4px 0;">'
                            f"{s_label}</div>\n"
                            f'  <div><span class="badge {s_badge_class}">{s_res_by.upper()}</span>\n'
                            f'  <span style="font-size:0.85rem; color:#64748b; margin-left:6px;">'
                            f"Score: <b>{s_score:+.2f}</b></span></div>\n"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    with m4:
                        dup_info = data.get("duplicate_info", {})
                        is_dup = dup_info.get("is_duplicate", False)
                        d_text = "🚨 DUPLICATE" if is_dup else "✅ UNIQUE"
                        d_color = "#dc2626" if is_dup else "#16a34a"
                        d_score = dup_info.get("similarity_score", 0.0)
                        st.markdown(
                            '<div class="metric-card">\n'
                            '  <div style="font-size:0.8rem; color:#64748b; font-weight:600;">DUPLICATE CHECK</div>\n'
                            f'  <div style="font-size:1.25rem; font-weight:700; color:{d_color}; margin:4px 0;">'
                            f"{d_text}</div>\n"
                            f'  <div style="font-size:0.85rem; color:#64748b;">Max Match: <b>{d_score:.1%}</b></div>\n'
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

                    # Detailed Analysis Layout
                    left_col, right_col = st.columns(2)

                    with left_col:
                        st.markdown("#### 🏷️ Extracted Named Entities")
                        entities = data.get("entities", [])
                        if entities:
                            html_chips = ""
                            for ent in entities:
                                lbl = ent.get("label", "MISC").upper()
                                txt = ent.get("text", "")
                                src = ent.get("source", "regex")
                                css_type = "entity-misc"
                                if "MONEY" in lbl:
                                    css_type = "entity-money"
                                elif "ORDER" in lbl:
                                    css_type = "entity-order"
                                elif "EMAIL" in lbl:
                                    css_type = "entity-email"
                                elif "DATE" in lbl:
                                    css_type = "entity-date"

                                html_chips += (
                                    f'<span class="entity-chip {css_type}">'
                                    f"<b>{lbl}:</b>&nbsp;{txt}&nbsp;"
                                    f'<small style="opacity:0.7">({src})</small></span>'
                                )
                            st.markdown(html_chips, unsafe_allow_html=True)
                        else:
                            st.info("No named entities detected.")

                        st.markdown("#### 🔑 TF-IDF Keywords")
                        keywords = data.get("keywords", [])
                        if keywords:
                            kw_html = "".join(
                                [
                                    f"""<span class="keyword-chip"># {kw}</span>"""
                                    for kw in keywords
                                ]
                            )
                            st.markdown(kw_html, unsafe_allow_html=True)
                        else:
                            st.info("No keywords extracted.")

                    with right_col:
                        st.markdown("#### 💬 Evidence Sentences")
                        evidence = data.get("evidence_sentences", [])
                        if evidence:
                            for ev in evidence:
                                st.markdown(
                                    f"""<div class="evidence-quote">"{ev}"</div>""",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No specific evidence sentences isolated.")

                        with st.expander("🔍 Normalized Text & Pipeline Details"):
                            st.write(f"**Cleaned Text:** `{data.get('cleaned_text')}`")
                            st.write(f"**Request ID:** `{data.get('request_id')}`")
                            st.write(
                                f"**LLM Fallback Used:** `{data.get('llm_fallback_used')}`"
                            )

                else:
                    st.error(f"Error {res.status_code}: {res.text}")

            except Exception as e:
                st.error(f"Failed to communicate with API gateway: {e}")

# ------------------------------------------------------------------------------
# TAB 2: Semantic Similarity & Duplicate Finder
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Semantic Similarity & Duplicate Detection")
    st.markdown(
        "Compare two customer support utterances using SentenceTransformers dense vector embeddings."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        text_a = st.text_area(
            "Text Snippet A",
            value="I was charged twice on my credit card for invoice #INV-9821. Please issue a refund.",
            height=100,
        )
    with s_col2:
        text_b = st.text_area(
            "Text Snippet B",
            value="There is a duplicate charge of $49.99 on my Visa statement. Can I get my money back?",
            height=100,
        )

    if st.button("⚡ Compare Embeddings", type="primary"):
        if text_a.strip() and text_b.strip():
            with st.spinner("Computing cosine similarity..."):
                try:
                    payload = {"text_a": text_a.strip(), "text_b": text_b.strip()}
                    res = requests.post(
                        f"{api_url}/similarity", json=payload, timeout=10
                    )
                    if res.status_code == 200:
                        sim_data = res.json()
                        score = sim_data.get("similarity_score", 0.0)
                        is_duplicate = sim_data.get("is_duplicate", False)

                        st.markdown("### Results")
                        st.progress(min(1.0, max(0.0, score)))

                        r1, r2 = st.columns(2)
                        with r1:
                            st.metric(
                                "Cosine Similarity Score",
                                f"{score:.4f}",
                                f"{score * 100:.1f}% Match",
                            )
                        with r2:
                            if is_duplicate:
                                st.error("⚠️ DUPLICATE DETECTED (Similarity >= 0.85)")
                            else:
                                st.success("✅ UNIQUE QUERY (Similarity < 0.85)")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ------------------------------------------------------------------------------
# TAB 3: Topic Taxonomy & Vector Store
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Controlled 10-Topic Taxonomy")
    st.markdown(
        "Explore the customer support topics and example utterances indexed in the Qdrant vector database."
    )

    try:
        import yaml
        from pathlib import Path

        topics_data = None
        candidates = [
            Path("config/topics.yaml"),
            Path(__file__).resolve().parent.parent / "config" / "topics.yaml",
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    raw_yaml = yaml.safe_load(f)
                    topics_data = raw_yaml.get("topics", {}) if raw_yaml else {}
                break

        if topics_data:
            for key, item in topics_data.items():
                with st.expander(
                    f"📌 {item.get('name', key)} ({len(item.get('examples', []))} Anchor Examples)"
                ):
                    st.markdown(f"**Description:** {item.get('description', '')}")
                    st.markdown(
                        f"**Associated Intents:** `{', '.join(item.get('intents', []))}`"
                    )
                    st.markdown("**Anchor Training Sentences:**")
                    for ex in item.get("examples", []):
                        st.markdown(f"- {ex}")
        else:
            st.info("No topic taxonomy loaded.")
    except Exception as e:
        st.warning(f"Could not load topics: {e}")

# ------------------------------------------------------------------------------
# TAB 4: Document Text Extractor
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Document Ingestion & Text Extraction")
    st.markdown(
        "Upload support invoices, receipts, or ticket logs in PDF or DOCX format."
    )

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if st.button("📥 Extract Text", type="primary", use_container_width=True):
            with st.spinner("Parsing document structure..."):
                try:
                    files = {
                        "file": (uploaded_file.name, file_bytes, uploaded_file.type)
                    }
                    res = requests.post(f"{api_url}/extract", files=files, timeout=20)
                    if res.status_code == 200:
                        ext_data = res.json()
                        st.session_state["extracted_text"] = ext_data.get(
                            "extracted_text", ""
                        )
                        st.session_state["extracted_msg"] = (
                            f"Extracted {ext_data.get('word_count')} words from "
                            f"{ext_data.get('filename')}!"
                        )
                    else:
                        st.error(f"Extraction failed: {res.text}")
                except Exception as e:
                    st.error(f"Error during upload: {e}")

    # Persistent full-width display for extracted document content
    if st.session_state.get("extracted_text"):
        if st.session_state.get("extracted_msg"):
            st.success(st.session_state["extracted_msg"])

        st.text_area(
            "Extracted Document Content",
            value=st.session_state["extracted_text"],
            height=350,
            help="Full extracted content from the uploaded document.",
        )

        if st.button(
            "🚀 Send to Ticket Analyzer",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["active_tab"] = "🎫 Ticket Analyzer"
            if "main_tabs" in st.session_state:
                st.session_state["main_tabs"] = "🎫 Ticket Analyzer"
            st.session_state["switch_to_tab_0"] = True
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 5: Architecture & Pipeline Guide
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("System Architecture & Processing Pipeline")
    st.markdown("""
        The **NLP Document Intelligence Gateway** follows a high-throughput,
        dual-brain architecture designed to minimize latency and external LLM costs:

        ```mermaid
        graph TD
            A[Incoming Support Query] --> B[FastAPI Gateway /analyze]
            B --> C[Dev 2: Local NLP Pipeline]
            C --> C1[Regex Normalization & Cleaning]
            C --> C2[spaCy POS Tagging & Lemmatization]
            C --> C3[Rule-Based & spaCy NER]
            C --> C4[TF-IDF Keyword Extraction]
            C --> C5[Lexicon Sentiment Analysis]

            B --> D[Dev 3: Semantic Vector Matching]
            D --> D1[SentenceTransformers Embedding 384-d]
            D --> D2[(Qdrant Vector DB)]
            D2 -->|Similarity >= 0.60| E[Classified Topic & Intent]
            D2 -->|Similarity < 0.60| F[External LLM Fallback: Gemini / OpenAI]

            C & E & F --> G[Unified Structured JSON Response]
        ```
        """)

    st.markdown("### Performance & Efficiency Specs")
    st.table(
        [
            {
                "Component": "Local Text Cleaning",
                "Latency": "< 2ms",
                "Cost": "$0.00",
                "Engine": "Python regex",
            },
            {
                "Component": "Linguistic & POS Analysis",
                "Latency": "< 15ms",
                "Cost": "$0.00",
                "Engine": "spaCy en_core_web_sm",
            },
            {
                "Component": "TF-IDF Keywords",
                "Latency": "< 5ms",
                "Cost": "$0.00",
                "Engine": "scikit-learn",
            },
            {
                "Component": "Vector Embedding & Search",
                "Latency": "< 20ms",
                "Cost": "$0.00",
                "Engine": "all-MiniLM-L6-v2 + Qdrant",
            },
            {
                "Component": "LLM Fallback (Ambiguous Only)",
                "Latency": "~500ms",
                "Cost": "$0.0001",
                "Engine": "Gemini 1.5 Flash / GPT-4o-mini",
            },
        ]
    )
