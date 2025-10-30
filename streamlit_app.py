import os, io, requests, streamlit as st
from datetime import datetime
import pandas as pd

# ===== CONFIG =====
BASE = os.getenv("BASE_URL", "https://sitting-profits-twiki-articles.trycloudflare.com")
SHOWS = ["CNN","FOXNEWS","NFL","LEGO","PICKERS","CONTINENTAL","BIGMOOD"]
AGGS  = ["ad_time","tv_time"]
RANK_COMBINE = {"Combined (mean of A/V)": "mean", "Audio only": "audio", "Visual only": "visual"}

st.set_page_config(page_title="Ad–Context Congruence", layout="wide")
st.title("Ad–Context Congruence (Audio + Visual)")

with st.sidebar:
    st.header("1) Upload your ad")
    up_file = st.file_uploader("Upload .mp4 / .mov", type=["mp4","mov","m4v"])
    st.caption("— OR paste a direct-download URL —")
    url_in  = st.text_input("URL (Dropbox ?dl=1, Drive uc?export=download&id=...)")

    st.header("2) Choose TV context")
    show = st.selectbox("TV show", SHOWS, index=2)  # default NFL
    agg  = st.selectbox("Aggregation", AGGS, index=0)

    st.header("3) Ranking options")
    combine_label = st.selectbox("Ranking metric", list(RANK_COMBINE.keys()), index=0)

    run_btn = st.button("Analyze", type="primary", use_container_width=True)

status = st.empty()
plot_col, info_col = st.columns([2,1], gap="large")

def upload_file_to_api(file_obj):
    files = {"file": (file_obj.name, file_obj, "video/mp4")}
    r = requests.post(f"{BASE}/upload_ad", files=files, timeout=300)
    r.raise_for_status()
    return r.json()["ad_id"]

def upload_url_to_api(url):
    r = requests.post(f"{BASE}/upload_ad_url", json={"url": url}, timeout=300)
    r.raise_for_status()
    return r.json()["ad_id"]

def score_ad(ad_id):
    r = requests.post(f"{BASE}/score/{ad_id}",
                      params={"include_audio": True, "include_visual": True},
                      timeout=1800)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_plot_both(ad_id, show, agg):
    r = requests.get(f"{BASE}/plot_both/{ad_id}",
                     params={"show": show, "agg": agg, "t": str(os.urandom(4).hex())},
                     timeout=300)
    r.raise_for_status()
    return r.content  # PNG bytes

def rank_ad(ad_id, combine="mean"):
    r = requests.get(f"{BASE}/rank/{ad_id}", params={"combine": combine}, timeout=600)
    r.raise_for_status()
    return r.json()

if run_btn:
    try:
        # 1) Upload (file preferred; else URL)
        if up_file is not None:
            status.info("Uploading file…")
            ad_id = upload_file_to_api(up_file)
        elif url_in.strip():
            status.info("Fetching video from URL…")
            ad_id = upload_url_to_api(url_in.strip())
        else:
            st.warning("Please upload a file or paste a direct-download URL.")
            st.stop()

        # 2) Score
        status.info(f"Scoring (ad_id={ad_id})…")
        _ = score_ad(ad_id)

        # 3) Plot both modalities for the selected show
        status.info("Rendering plot…")
        png = fetch_plot_both(ad_id, show, agg)

        with plot_col:
            st.image(io.BytesIO(png),
                     caption=f"{show} – audio & visual congruence ({agg})",
                     use_container_width=True)

        with info_col:
            st.subheader("Details")
            st.write(f"**ad_id:** `{ad_id}`")
            st.write(f"**TV show:** {show}")
            st.write(f"**Aggregation:** {agg}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Scored at {ts}. Try another show/aggregation; plots reload instantly after scoring.")

        status.success("Done ✓")

        # 4) Top matches ranking
        st.markdown("### Top matches (by mean Fisher-z)")
        try:
            metric_key = RANK_COMBINE[combine_label]
            rnk = rank_ad(ad_id, combine=metric_key)
            rows = rnk["ranking"]
            df = pd.DataFrame([{
                "Show": r["show"],
                "Combined": round(r["combined"], 3),
                "Audio": round(r["audio"], 3),
                "Visual": round(r["visual"], 3),
            } for r in rows])
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download ranking (CSV)",
                data=csv_bytes,
                file_name=f"ranking_{ad_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except requests.HTTPError as e:
            st.warning("Ranking endpoint not available yet. Add `/rank/{ad_id}` to the API to enable this table.")
        except Exception as e:
            st.warning(f"Could not load ranking: {e}")

        # 5) All shows grid
        st.markdown("### All shows (audio + visual plots)")
        grid_cols = st.columns(3, gap="large")
        for i, s in enumerate(SHOWS):
            try:
                img = fetch_plot_both(ad_id, s, agg)
                with grid_cols[i % 3]:
                    st.image(io.BytesIO(img), caption=f"{s} ({agg})", use_container_width=True)
            except Exception:
                with grid_cols[i % 3]:
                    st.info(f"{s}: plot unavailable")

    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
