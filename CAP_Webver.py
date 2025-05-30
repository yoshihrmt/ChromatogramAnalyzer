import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import matplotlib.lines as mlines
from scipy.signal import find_peaks
from scipy.integrate import simpson
import io

# フォント全体設定
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Times New Roman", Times, serif !important;
}
</style>
""", unsafe_allow_html=True)

colors = [
    'navy', 'crimson', 'forestgreen', 'darkorange', 'purple', 'teal', 'maroon',
    'darkviolet', 'goldenrod', 'indigo', 'darkturquoise', 'chocolate', 'darkmagenta',
    'steelblue', 'olive', 'darkred', 'midnightblue', 'darkgreen', 'saddlebrown',
    'darkseagreen', 'cadetblue', 'darkslategray', 'darkblue', 'firebrick',
    'seagreen', 'darkcyan', 'darkkhaki', 'sienna', 'royalblue', 'darkslateblue',
    'darkgoldenrod', 'mediumorchid', 'mediumblue', 'indianred', 'mediumseagreen',
    'peru', 'slateblue', 'olivedrab', 'rosybrown', 'mediumvioletred'
]
markers = [
    'o', 's', '^', 'v', 'd', 'x', '+', '<', '>', '*', 'p', 'h', 'H', 'D', '|', '_', '8'
]

def process_chromatogram_data(df):
    df['height(mV)'] = df['height(uV)'].replace([np.inf, -np.inf], np.nan).fillna(0) / 1000
    df['time(min)'] = df['time(sec)'].replace([np.inf, -np.inf], np.nan).fillna(0) / 60
    return df

def calculate_peak_parameters(data, time, peak_index):
    peak_height = data[peak_index]
    window_size = 20
    left_base = data[max(0, peak_index - window_size):peak_index].min()
    right_base = data[peak_index:min(len(data), peak_index + window_size)].min()
    baseline = (left_base + right_base) / 2
    effective_height = peak_height - baseline
    threshold_height = baseline + effective_height * 0.05
    left = peak_index
    while left > 0 and data[left] > threshold_height:
        left -= 1
    right = peak_index
    while right < len(data)-1 and data[right] > threshold_height:
        right += 1
    W_0_05h = time[right] - time[left]
    f = time[peak_index] - time[left]
    symmetry = W_0_05h / (2 * f) if f != 0 else np.nan
    return symmetry, W_0_05h, f

st.title("Chromatogram Analyzer")

uploaded_files = st.file_uploader(
    "Excelファイルを複数選択してください",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

# ↓↓↓ 軸範囲の自動計算変数の初期化
auto_xmin, auto_xmax, auto_ymin, auto_ymax = 0.0, 10.0, 0.0, 200.0
file_info_list = []

if uploaded_files:
    legends = []
    xmin_total, xmax_total, ymin_total, ymax_total = [], [], [], []
    for i, uploaded_file in enumerate(uploaded_files):
        legend_label = st.text_input(
            f"{uploaded_file.name} の凡例名",
            value=uploaded_file.name,
            key=uploaded_file.name
        )
        legends.append(legend_label)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            df = pd.read_excel(uploaded_file, sheet_name='RAW DATA')
            df = process_chromatogram_data(df)
            data = df['height(mV)'].values
            time = df['time(min)'].values
            peaks, _ = find_peaks(data, height=10.0, prominence=0.5, width=10)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            file_info_list.append({
                "name": uploaded_file.name,
                "legend": legends[idx],
                "peaks": peaks,
                "time": time,
                "data": data,
                "color": color,
                "marker": marker,
            })
            xmin_total.append(np.min(time))
            xmax_total.append(np.max(time))
            ymin_total.append(np.min(data))
            ymax_total.append(np.max(data))
        except Exception as e:
            st.error(f"{uploaded_file.name}: エラー発生({e})")

    # 軸範囲の自動計算（ファイル全部読んだあと！）
    if xmin_total and xmax_total:
        auto_xmin = float(np.min(xmin_total))
        auto_xmax = float(np.max(xmax_total))
    if ymin_total and ymax_total:
        auto_ymin = float(np.min(ymin_total))
        # y軸のmaxだけ[上方向に10%拡大]も可能
        auto_ymax = float(np.max(ymax_total)) * 1.1
    else:
        auto_ymin, auto_ymax = 0.0, 200.0

# サイドバー（自動計算値を渡すのはここ！）
with st.sidebar:
    st.header("グラフ詳細設定")
    xaxis_auto = st.checkbox("x軸を自動", value=True)
    yaxis_auto = st.checkbox("y軸を自動", value=True)
    x_min = st.number_input("x軸最小(分)", value=0, disabled=xaxis_auto)
    x_max = st.number_input("x軸最大(分)", value=auto_xmax, disabled=xaxis_auto)
    y_min = st.number_input("y軸最小(mV)", value=-10, disabled=yaxis_auto)
    y_max = st.number_input("y軸最大(mV)", value=auto_ymax, disabled=yaxis_auto)
    show_scalebar = st.checkbox("スケールバーを表示", value=True)
    scale_value = st.number_input("スケールバー値(mV)", value=50)
    scale_x_pos = st.slider("スケールバー x位置（0=左, 1=右）", 0.0, 1.0, 0.7, 0.01)
    scale_y_pos = st.slider("スケールバー y位置（0=下, 1=上）", 0.0, 1.0, 0.15, 0.01)
    font_xlabel = st.slider("x軸ラベルフォント", 6, 30, 14)
    font_ylabel = st.slider("y軸ラベルフォント", 6, 30, 14)
    font_legend = st.slider("凡例フォント", 6, 24, 10)
    font_tick = st.slider("目盛フォント", 6, 20, 10)

show_peaks = st.checkbox("ピークマーカーを表示（全データ）", True)
show_legend = st.checkbox("凡例を表示", True)

if uploaded_files:
    st.markdown("### 凡例名（任意で変更可）")
    fig, ax = plt.subplots(figsize=(9, 4))
    handles = []  # 凡例用Line2Dオブジェクト

    for idx, info in enumerate(file_info_list):
        data = info["data"]
        time = info["time"]
        peaks = info["peaks"]
        color = info["color"]
        marker = info["marker"]
        legend = info["legend"]

        # 波形
        ax.plot(time, data, label=legend, color=color)
        # ピークマーカー
        if show_peaks and len(peaks) > 0:
            ax.plot(
                time[peaks], data[peaks], marker,
                linestyle="None", markersize=6,
                markerfacecolor=color, markeredgecolor=color, label=None
            )
            legend_line = mlines.Line2D(
                [], [], color=color, marker=marker, linestyle='-',
                label=legend, markersize=6,
                markerfacecolor=color, markeredgecolor=color
            )
        else:
            legend_line = mlines.Line2D(
                [], [], color=color, marker=None, linestyle='-',
                label=legend
            )
        handles.append(legend_line)

    # 軸範囲
    if not xaxis_auto:
        ax.set_xlim(x_min, x_max)
    elif xmin_total and xmax_total:
        ax.set_xlim(min(xmin_total), max(xmax_total))
    if not yaxis_auto:
        ax.set_ylim(y_min, y_max)
    elif ymin_total and ymax_total:
        ax.set_ylim(min(ymin_total), max(ymax_total) * 1.1)

    # 枠・y軸メモリ消し
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])

    # ラベル・装飾
    ax.set_xlabel("Time (min)", fontsize=font_xlabel)
    ax.set_ylabel("Absorbance [-]", fontsize=font_ylabel)

    # y軸の上向き矢印
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    arrow_x = xlim[0]
    ax.annotate(
        "",
        xy=(arrow_x, ylim[1]),
        xytext=(arrow_x, ylim[0]),
        arrowprops=dict(arrowstyle='->', lw=1.5),
        clip_on=False
    )

    # カスタム凡例
    if show_legend:
        ax.legend(handles=handles, fontsize=font_legend)
    ax.tick_params(axis='both', labelsize=font_tick)

    # スケールバー
    if show_scalebar:
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        x_pos = current_xlim[0] + (current_xlim[1] - current_xlim[0]) * scale_x_pos
        y_start = current_ylim[0] + (current_ylim[1] - current_ylim[0]) * scale_y_pos
        ax.annotate(
            '', xy=(x_pos, y_start), xytext=(x_pos, y_start+scale_value),
            arrowprops=dict(arrowstyle='<->', linewidth=1)
        )
        ax.text(
            x_pos + (current_xlim[1] - current_xlim[0])*0.01,
            y_start + scale_value / 2,
            f"{scale_value} mV", va='center', ha='left', fontsize=font_xlabel
        )

    st.pyplot(fig)

    # グラフ保存用バッファとダウンロードボタン
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="グラフをPDFで保存",
        data=buf,
        file_name="chromatogram.pdf",
        mime="application/pdf"
    )

    # 解析結果表示
    st.markdown("### 解析結果")
    for info in file_info_list:
        st.write(f"**{info['legend']}** … 検出されたピーク数: {len(info['peaks'])}")
        for i, peak in enumerate(info["peaks"], 1):
            area = simpson(
                y=info['data'][max(0, peak-50):min(len(info['data']), peak+50)],
                x=info['time'][max(0, peak-50):min(len(info['time']), peak+50)]
            )
            symmetry, W_0_05h, f = calculate_peak_parameters(info['data'], info['time'], peak)
            st.write(
                f"""ピーク {i}:  保持時間: {info['time'][peak]:.2f}分  ピーク高さ: {info['data'][peak]:.2f}mV  面積: {area:.2f}  W_0.05h: {W_0_05h:.3f}  f: {f:.3f}  シンメトリー係数(S): {symmetry:.3f}"""
            )

