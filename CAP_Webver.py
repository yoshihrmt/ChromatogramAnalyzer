import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
from scipy.signal import find_peaks
from scipy.integrate import simpson
import io
import traceback

font_path = "fonts/times-new-roman.ttf"
if not os.path.exists(font_path):
    st.warning("フォントファイルが見つかりません。serifで表示します。")
    font_prop = FontProperties(family="serif")
else:
    font_prop = FontProperties(fname=font_path)

plt.rcParams['mathtext.fontset'] = 'cm'

# Streamlit全体CSSでserif化
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "EB Garamond", "Times New Roman", Times, serif !important;
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
markers = ['o', 's', '^', 'v', 'd', 'x', '+', '<', '>', '*', 'p', 'h', 'H', 'D', '|', '_', '8']

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

st.markdown(
    """
    <div style="
    background: #2c2c2c;
    color: #fff;
    border-radius: 16px;
    padding: 10px 0 6px 0;
    margin-bottom: 18px;
    font-size: 1.8rem;
    font-weight: bold;
    text-align: center;
    font-family: 'EB Garamond', 'Times New Roman', Times, serif;
    ">
    Chromatogram Analyzer
    </div>
    """,
    unsafe_allow_html=True
)

# --- 各種パラメータ設定はサイドバーで定義 ---
with st.sidebar:
    st.markdown(
        """
        <div style="
        background: #2c2c2c;
        color: #fff;
        border-radius: 16px;
        padding: 10px 0 6px 0;
        margin-bottom: 18px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        font-family: 'EB Garamond', 'Times New Roman', Times, serif;
        ">
        グラフ詳細
        </div>
        """,
        unsafe_allow_html=True
    )
    auto_xmin, auto_xmax, auto_ymin, auto_ymax = 0.0, 10.0, 0.0, 200.0

    xaxis_auto = st.checkbox("x軸を自動", value=True, key="xaxis_auto")
    yaxis_auto = st.checkbox("y軸を自動", value=True, key="yaxis_auto")
    x_min = st.number_input("x軸最小(分)", value=0.0, disabled=xaxis_auto, key="x_min")
    x_max = st.number_input("x軸最大(分)", value=10.0, disabled=xaxis_auto, key="x_max")
    y_min = st.number_input("y軸最小(mV)", value=-10.0, disabled=yaxis_auto, key="y_min")
    y_max = st.number_input("y軸最大(mV)", value=200.0, disabled=yaxis_auto, key="y_max")
    st.markdown(
        """
        <div style="
        background: #2c2c2c;
        color: #fff;
        border-radius: 16px;
        padding: 10px 0 6px 0;
        margin-bottom: 18px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        font-family: 'EB Garamond', 'Times New Roman', Times, serif;
        ">
        スケールバー
        </div>
        """,
        unsafe_allow_html=True
    )
    show_scalebar = st.checkbox("スケールバーを表示", value=True, key="show_scalebar")
    scale_value = st.number_input("スケールバー値(mV)", value=50, key="scale_value")
    scale_x_pos = st.slider("スケールバー x位置（0=左, 1=右）", 0.0, 1.0, 0.7, 0.1, key="scale_x_pos")
    scale_y_pos = st.slider("スケールバー y位置（0=下, 1=上）", 0.0, 1.0, 0.15, 0.1, key="scale_y_pos")
    st.markdown(
        """
        <div style="
        background: #2c2c2c;
        color: #fff;
        border-radius: 16px;
        padding: 10px 0 6px 0;
        margin-bottom: 18px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        font-family: 'EB Garamond', 'Times New Roman', Times, serif;
        ">
        フォントサイズ
        </div>
        """,
        unsafe_allow_html=True
    )
    font_xlabel = st.slider("x軸ラベルフォント", 10, 30, 25, key="font_xlabel")
    font_ylabel = st.slider("y軸ラベルフォント", 10, 30, 25, key="font_ylabel")
    font_legend = st.slider("凡例フォント", 5, 22, 18, key="font_legend")
    font_tick = st.slider("x軸値フォント", 5, 18, 14, key="font_tick")
    font_scale_value = st.slider("スケールバー値フォント", 10, 30, 25, key="font_scale_value")
    st.markdown(
        """
        <div style="
        background: #2c2c2c;
        color: #fff;
        border-radius: 16px;
        padding: 10px 0 6px 0;
        margin-bottom: 18px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        font-family: 'EB Garamond', 'Times New Roman', Times, serif;
        ">
        ピーク検出パラメータ
        </div>
        """, unsafe_allow_html=True
    )
    peak_width = st.slider("ピーク幅 (width)", min_value=1, max_value=100, value=10, step=1)
    peak_height = st.number_input("高さしきい値 (height, mV)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
    peak_prominence = st.number_input("突出度 (prominence)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)

# --- ファイルアップロード ---
uploaded_files = st.file_uploader(
    "Excelファイルを複数選択してください",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

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
            peaks, _ = find_peaks(
                data,
                height=peak_height,
                prominence=peak_prominence,
                width=peak_width,
            )
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
            st.error(f"{uploaded_file.name}: エラー発生 ({e})")
            with st.expander("エラー詳細を表示"):
                st.write(traceback.format_exc())

if uploaded_files and file_info_list:

    # --- ここでチェックボックスをメインカラムで上に設置 ---
    show_peaks = st.checkbox("ピークマーカーを表示", value=True, key="show_peaks_inline")
    show_legend = st.checkbox("凡例を表示", value=True, key="show_legend_inline")

    # より大きなフィギュアサイズで左に十分な余白を確保
    fig, ax = plt.subplots(figsize=(12, 6))

    handles = []     
    for idx, info in enumerate(file_info_list):
        data = info["data"]
        time = info["time"]
        peaks = info["peaks"]
        color = info["color"]
        marker = info["marker"]
        legend = info["legend"]

        ax.plot(time, data, label=legend, color=color)
        if show_peaks and len(peaks) > 0:
            ax.plot(
                time[peaks], data[peaks], marker,
                linestyle="None", markersize=6,
                markerfacecolor=color, markeredgecolor=color, label=None
            )
            legend_line = mlines.Line2D([], [], color=color, marker=marker, linestyle='-',
                                        label=legend, markersize=6, markerfacecolor=color, markeredgecolor=color)
        else:
            legend_line = mlines.Line2D([], [], color=color, marker=None, linestyle='-',
                                        label=legend)
        handles.append(legend_line)

    # 軸スケール設定
    if not xaxis_auto and len(xmin_total)>0 and len(xmax_total)>0:
        ax.set_xlim(x_min, x_max)
    elif xmin_total and xmax_total:
        ax.set_xlim(min(xmin_total), max(xmax_total))
    if not yaxis_auto and len(ymin_total)>0 and len(ymax_total)>0:
        ax.set_ylim(y_min, y_max)
    elif ymin_total and ymax_total:
        ax.set_ylim(min(ymin_total), max(ymax_total) * 1.1)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])

    ax.set_xlabel("Time /min", fontsize=font_xlabel, fontproperties=font_prop)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(font_tick)

    plt.tight_layout()
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    arrow_x = xlim[0] - (xlim[1] - xlim[0]) * 0.005
    arrow_y_start = ylim[0] + (ylim[1] - ylim[0]) * 0.1
    arrow_y_end = ylim[1]

    ax.text(
        arrow_x - (xlim[1] - xlim[0]) * 0.005,
        (arrow_y_start + arrow_y_end) / 2,
        "Absorbance /-",
        fontsize=font_ylabel,
        fontproperties=font_prop,
        rotation=90,
        va='center',
        ha='right',
        clip_on=False
    )
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(font_tick)
    
    # 矢印を描画（より確実な設定）
    ax.annotate(
        "",
        xy=(arrow_x, arrow_y_end),
        xytext=(arrow_x, arrow_y_start),
        arrowprops=dict(arrowstyle='->', lw=2, color='black'),
        clip_on=False,
        annotation_clip=False
    )

    if show_legend:
        if os.path.exists(font_path):
            font_legend_prop = FontProperties(fname=font_path, size=font_legend)
        else:
            font_legend_prop = FontProperties(family="serif", size=font_legend)
        ax.legend(handles=handles, prop=font_legend_prop)

    if show_scalebar:
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        x_pos = current_xlim[0] + (current_xlim[1] - current_xlim[0]) * scale_x_pos
        y_start = current_ylim[0] + (current_ylim[1] - current_ylim[0]) * scale_y_pos
        ax.annotate(
            '', xy=(x_pos, y_start), xytext=(x_pos, y_start + scale_value),
            arrowprops=dict(arrowstyle='<->', linewidth=1)
        )
        ax.text(
            x_pos + (current_xlim[1] - current_xlim[0]) * 0.01,
            y_start + scale_value / 2,
            f"{scale_value} mV",
            va='center', ha='left',
            fontsize=font_scale_value,
            fontproperties=font_prop
        )

    st.pyplot(fig)

    buf = io.BytesIO()
    # 保存時に矢印も含まれるように設定
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)
    st.download_button(
        label="グラフをPNGで保存",
        data=buf,
        file_name="chromatogram.png",
        mime="application/png"
    )

    st.markdown("### 解析結果")
    for info in file_info_list:
        st.write(f"{info['legend']} … 検出されたピーク数: {len(info['peaks'])}")
        for i, peak in enumerate(info["peaks"], 1):
            area = simpson(
                y=info['data'][max(0, peak-50):min(len(info['data']), peak+50)],
                x=info['time'][max(0, peak-50):min(len(info['time']), peak+50)]
            )
            symmetry, W_0_05h, f = calculate_peak_parameters(info['data'], info['time'], peak)
            st.write(
                f"""ピーク {i}:  保持時間: {info['time'][peak]:.2f}分  ピーク高さ: {info['data'][peak]:.2f}mV  面積: {area:.2f}  W_0.05h: {W_0_05h:.3f}  f: {f:.3f}  シンメトリー係数(S): {symmetry:.3f}"""
            )
