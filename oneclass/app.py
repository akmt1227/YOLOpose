"""One-class anomaly-detection prototype GUI.

Launch from the repository root:
    uv run streamlit run oneclass/app.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from pose_features import YOLO_WEIGHTS  # noqa: E402

st.set_page_config(
    page_title="ワンクラス異常検知 試作",
    page_icon=":material/monitor_heart:",
    layout="wide",
)

# ---------------------------------------------------------------- state
for key in ("log_extract", "log_train", "log_detect"):
    st.session_state.setdefault(key, None)      # (exit_code, log_text)
st.session_state.setdefault("last_result", None)  # path of last detection output


# ---------------------------------------------------------------- helpers
def artifact_info():
    """What the pipeline has produced so far (drives the status badges/gating)."""
    info = {"windows": None, "model": False, "cal": None}
    x_path = os.path.join(SCRIPT_DIR, "X_normal.npy")
    if os.path.exists(x_path):
        try:
            info["windows"] = int(np.load(x_path, mmap_mode="r").shape[0])
        except Exception:
            info["windows"] = 0
    info["model"] = os.path.exists(os.path.join(SCRIPT_DIR, "ae_model.pth"))
    thr_path = os.path.join(SCRIPT_DIR, "threshold.json")
    if os.path.exists(thr_path):
        try:
            with open(thr_path, encoding="utf-8") as f:
                info["cal"] = json.load(f)
        except Exception:
            pass
    return info


def run_streaming(cmd, state_key):
    """Run a pipeline script, stream its output live, store (code, log) in state."""
    with st.status("実行中... しばらくお待ちください", expanded=True) as status:
        placeholder = st.empty()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", cwd=REPO_ROOT)
        lines = []
        for line in iter(process.stdout.readline, ""):
            lines.append(line.rstrip())
            placeholder.code("\n".join(lines[-25:]), language=None)
        process.stdout.close()
        code = process.wait()
        status.update(
            label="完了" if code == 0 else "エラーで終了しました",
            state="complete" if code == 0 else "error", expanded=False)
    st.session_state[state_key] = (code, "\n".join(lines))
    return code


def show_last_log(state_key):
    """Show the stored result of the last run (survives page switches)."""
    stored = st.session_state.get(state_key)
    if stored is None:
        return
    code, log = stored
    if code == 0:
        st.success("前回の実行は正常に完了しました。", icon=":material/check_circle:")
    else:
        st.error(f"前回の実行はエラーで終了しました (exit code {code})。", icon=":material/error:")
    with st.expander("前回の実行ログ", icon=":material/description:"):
        st.code(log, language=None)


def preview_frames(video_path, n=4):
    """Grab evenly spaced frames from the result video (always displayable,
    unlike mp4v-encoded video which some browsers refuse to play)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in np.linspace(total * 0.2, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


info = artifact_info()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("ワンクラス異常検知")
    st.caption("良品動画のみで学習する試作版")

    step = st.radio("ステップ", [
        "1. 正常データ抽出",
        "2. 学習・しきい値較正",
        "3. 検知テスト",
    ])

    st.space("small")
    st.caption("パイプラインの状態")
    if info["windows"]:
        st.badge(f"抽出済み: {info['windows']} 窓", icon=":material/check:", color="green")
    else:
        st.badge("未抽出", icon=":material/pending:", color="gray")
    if info["model"] and info["cal"]:
        st.badge("学習・較正済み", icon=":material/check:", color="green")
    else:
        st.badge("未学習", icon=":material/pending:", color="gray")

    st.space("small")
    st.caption(f"YOLOモデル: {YOLO_WEIGHTS}")

# ---------------------------------------------------------------- step 1
if step.startswith("1"):
    st.title("正常データ抽出")
    st.markdown("良品（正常作業）の動画から姿勢の13秒窓を抽出します。"
                "**正常な動画のみ**を対象にしてください（NG動画は不要です）。")

    normal_dir = st.text_input("正常動画のフォルダ", value="dataset/normal")
    resolved = normal_dir if os.path.isabs(normal_dir) else os.path.join(REPO_ROOT, normal_dir)
    if os.path.isdir(resolved):
        n_videos = len([f for f in os.listdir(resolved) if f.lower().endswith(".mp4")])
        st.caption(f"{n_videos} 本の .mp4 が見つかりました")
    else:
        n_videos = 0
        st.caption("フォルダが見つかりません")

    if st.button("抽出を開始", type="primary", icon=":material/play_arrow:",
                 disabled=(n_videos == 0)):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "prepare_normal.py"),
               "--normal_dir", resolved]
        run_streaming(cmd, "log_extract")
        st.rerun()

    show_last_log("log_extract")

# ---------------------------------------------------------------- step 2
elif step.startswith("2"):
    st.title("学習・しきい値較正")
    st.markdown("抽出した正常データだけでオートエンコーダを学習し、"
                "NG判定のしきい値を自動較正します。**NG動画・ラベルは不要**です。")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("エポック数", min_value=5, max_value=1000, value=400, step=5)
    with col2:
        percentile = st.slider(
            "較正パーセンタイル", min_value=90.0, max_value=99.9, value=99.0, step=0.1,
            help="正常データのスコア分布のこの百分位を異常しきい値にします。"
                 "下げると敏感になり、誤報も増えます。")

    if info["windows"] in (None, 0):
        st.warning("先にステップ1でデータ抽出を実行してください。", icon=":material/warning:")

    if st.button("学習を開始", type="primary", icon=":material/model_training:",
                 disabled=(info["windows"] in (None, 0))):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_ae.py"),
               "--epochs", str(int(epochs)), "--percentile", str(percentile)]
        run_streaming(cmd, "log_train")
        st.rerun()

    show_last_log("log_train")

    if info["cal"]:
        st.subheader("較正されたしきい値")
        c1, c2, c3 = st.columns(3)
        c1.metric("異常しきい値（復元誤差）", f"{info['cal']['recon_threshold']:.5f}")
        c2.metric("静止しきい値（動きエネルギー）", f"{info['cal']['idle_threshold']:.6f}")
        c3.metric("較正パーセンタイル", f"p{info['cal'].get('percentile', 99):g}")

# ---------------------------------------------------------------- step 3
else:
    st.title("検知テスト")
    st.markdown("動画を解析し、**正常作業から外れた動き**・**静止**・**不在**を検知すると"
                "赤いNGバナーを表示します。右上には復元誤差スコアがリアルタイム表示されます。")

    trained = bool(info["model"] and info["cal"])
    if not trained:
        st.warning("先にステップ2で学習を完了してください。", icon=":material/warning:")

    input_video = st.text_input("テストする動画のパス", value="")
    output_video = st.text_input("結果の保存先", value="result_oneclass.mp4")

    threshold = None
    if trained:
        default_thr = float(info["cal"]["recon_threshold"])
        override = st.toggle("しきい値を手動調整する",
                             help="既定は較正値。下げると敏感（誤報増）、上げると鈍感になります。")
        if override:
            threshold = st.number_input("異常しきい値", min_value=0.0,
                                        value=default_thr, step=default_thr / 10 or 0.001,
                                        format="%.5f")
        else:
            st.caption(f"較正済みしきい値 {default_thr:.5f} を使用します")

    in_resolved = input_video if os.path.isabs(input_video) else os.path.join(REPO_ROOT, input_video)
    out_resolved = output_video if os.path.isabs(output_video) else os.path.join(REPO_ROOT, output_video)
    input_ok = bool(input_video) and os.path.exists(in_resolved)
    if input_video and not input_ok:
        st.caption("ファイルが見つかりません")

    if st.button("検知を開始", type="primary", icon=":material/search:",
                 disabled=not (trained and input_ok)):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "detect.py"),
               "--input", in_resolved, "--output", out_resolved]
        if threshold is not None:
            cmd += ["--threshold", str(threshold)]
        code = run_streaming(cmd, "log_detect")
        if code == 0 and os.path.exists(out_resolved):
            st.session_state.last_result = out_resolved
        st.rerun()

    show_last_log("log_detect")

    result = st.session_state.get("last_result")
    if result and os.path.exists(result):
        st.subheader("検知結果")
        ng_lines = []
        stored = st.session_state.get("log_detect")
        if stored:
            ng_lines = [l for l in stored[1].splitlines() if "[NG]" in l]
        if ng_lines:
            st.error(f"NG発報: {len(ng_lines)} 件", icon=":material/notification_important:")
            st.code("\n".join(ng_lines), language=None)
        else:
            st.success("NGは検知されませんでした。", icon=":material/check_circle:")

        frames = preview_frames(result)
        if frames:
            st.caption("結果プレビュー（等間隔サンプル）")
            for col, fr in zip(st.columns(len(frames)), frames):
                col.image(fr)
        st.video(result)
        st.caption("動画が再生できない場合（コーデック非対応）は、保存先のファイルを直接開いてください: "
                   f"`{result}`")
