import streamlit as st
import subprocess
import os

st.set_page_config(page_title="YOLO-Pose Anomaly Detection", layout="wide")

def stream_subprocess_output(command, log_container):
    """Runs a command and streams its output to a Streamlit text area."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8")
    
    log_output = []
    for line in iter(process.stdout.readline, ""):
        log_output.append(line)
        # Display the last 20 lines to avoid freezing the browser with massive text
        log_container.text("".join(log_output[-20:]))
        
    process.stdout.close()
    return_code = process.wait()
    return return_code, "".join(log_output)

st.title("🏃 YOLO-Pose Anomaly Detection System")

# Sidebar Navigation
mode = st.sidebar.radio("モード選択 (Select Mode)", ["1. データ抽出 (Data Prep)", "2. 学習 (Training)", "3. 推論 (Inference)"])

if mode == "1. データ抽出 (Data Prep)":
    st.header("1. 動画から骨格データを抽出")
    st.markdown("""指定したフォルダにある動画からYOLOを使って人物の骨格（キーポイント）を抽出し、学習用データとして保存します。

**フォルダ構成**: NG動画は種類別のサブフォルダに分けてください。
```
dataset/normal/                正常な検査作業
dataset/abnormal/too_long/     外観の検査時間が長い
dataset/abnormal/no_pointing/  検査ボードの指差し確認がない
dataset/abnormal/drop/         物を落下させている
dataset/abnormal/skipped/      外観検査をしていない箇所がある
```
""")
    
    col1, col2 = st.columns(2)
    with col1:
        normal_dir = st.text_input("正常動画のフォルダ (Normal Videos Dir)", value="dataset/normal")
    with col2:
        abnormal_dir = st.text_input("異常動画のフォルダ (Abnormal Videos Dir)", value="dataset/abnormal")
        
    if st.button("骨格抽出を開始 (Start Extraction)", type="primary"):
        if not os.path.exists(normal_dir) or not os.path.exists(abnormal_dir):
            st.error("指定されたフォルダが見つかりません。パスを確認してください。")
        else:
            st.info("抽出処理を実行中... しばらくお待ちください。")
            log_container = st.empty()
            
            # Execute prepare_data.py
            cmd = ["uv", "run", "python", "prepare_data.py", "--normal_dir", normal_dir, "--abnormal_dir", abnormal_dir]
            
            with st.spinner("YOLOが動画を解析中です..."):
                code, logs = stream_subprocess_output(cmd, log_container)
                
            if code == 0:
                st.success("✅ データ抽出が完了しました！X_data.npyとy_labels.npyが保存されました。")
            else:
                st.error("❌ エラーが発生しました。ログを確認してください。")

elif mode == "2. 学習 (Training)":
    st.header("2. LSTMモデルの学習")
    st.markdown("抽出された骨格データを使って、正常作業とNG4種（検査時間過多・指差し確認なし・落下・検査箇所抜け）をLSTMに分類学習させます。")
    
    if st.button("学習を開始 (Start Training)", type="primary"):
        st.info("学習を実行中...")
        log_container = st.empty()
        
        cmd = ["uv", "run", "python", "train.py"]
        
        with st.spinner("LSTMモデルをトレーニング中..."):
            code, logs = stream_subprocess_output(cmd, log_container)
            
        if code == 0:
            st.success("✅ 学習が完了しました！`lstm_model.pth` と `threshold.json`（判定しきい値）が保存されました。")
        else:
            st.error("❌ エラーが発生しました。データ抽出(Data Prep)は完了していますか？")

elif mode == "3. 推論 (Inference)":
    st.header("3. 異常検知のテスト (Inference)")
    st.markdown("学習済みのモデルを使って動画を解析します。NG動作を検知するとフレーム上部に赤いバナーで**NG理由付き**（例: NG: OBJECT DROPPED）で発報します。")
    
    input_video = st.text_input("テストする動画のパス (Input Video Path)", value="dataset/abnormal/test.mp4")
    output_video = st.text_input("結果の保存先 (Output Video Path)", value="result.mp4")
    
    if st.button("推論を開始 (Run Inference)", type="primary"):
        if not os.path.exists(input_video):
            st.error(f"ファイルが見つかりません: {input_video}")
        else:
            st.info("推論処理を実行中...")
            log_container = st.empty()
            
            cmd = ["uv", "run", "python", "main.py", "--input", input_video, "--output", output_video]
            
            with st.spinner("動画を解析・描画中..."):
                code, logs = stream_subprocess_output(cmd, log_container)
                
            if code == 0 and os.path.exists(output_video):
                st.success("✅ 推論が完了しました！")
                # Streamlit can play mp4 videos natively!
                st.video(output_video)
            else:
                st.error("❌ エラーが発生しました。")
