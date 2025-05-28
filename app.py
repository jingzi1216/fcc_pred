import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 加载训练好的模型
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')
# 目标变量名称，与训练时保持一致
TARGET_COLUMNS = [
    '汽油收率wt%', '汽油芳烃含量vol %', '汽油烯烃含量vol%', '汽油RON', '汽油干点℃',
    '液化气收率wt%', '液化气丙烯含量wt%', '液化气C5体积比 vol%',
    '烟气中CO2排放量t/h', '柴油ASTM D8695% ℃'
]
target_names = TARGET_COLUMNS.copy()
# 特征变量名称列表，请与训练时保持一致，并补充了原料氮、硫含量
FEATURE_NAMES = [
    '原料质量流量t/h', '原料芳烃含量wt%', '原料镍含量ppmwt', '原料钒含量ppmwt',
    '原料残炭含量 wt%', '原料预热温度℃', '反应压力bar_g', '反应温度℃',
    '催化剂微反活性t%', '新鲜催化剂活性 wt%', '反应器密相催化剂藏量kg', '再生器床温℃',
    '原料比重g/cm3', '原料氮含量wt%', '原料硫含量wt%',
    '催化剂补充速率tonne/d', '提升蒸汽注入量tonne/hr', '雾化蒸汽注入量tonne/hr', '汽提蒸汽注入量tonne/hr'
]

# 默认值：可根据历史或经验数据调整
DEFAULT_INPUTS = {feat: 0.0 for feat in FEATURE_NAMES}
# 示例默认值（请按实际情况修改）
DEFAULT_INPUTS.update({
    '原料质量流量t/h':420,
    '原料芳烃含量wt%':31.98,
    '原料镍含量ppmwt':0.75,
    '原料钒含量ppmwt':0.35,
    '原料残炭含量 wt%':0.59,
    '原料预热温度℃':219.96,
    '反应压力bar_g':0.44,
    '反应温度℃':495.3,
    '催化剂微反活性t%':45.46,
    '新鲜催化剂活性 wt%':61.53,
    '反应器密相催化剂藏量kg':2017.42,
    '再生器床温℃':674.57,
    '原料比重g/cm3':0.88,
    '原料氮含量wt%':0.07,
    '原料硫含量wt%':0.37,
    '催化剂补充速率tonne/d':4.43,
    '提升蒸汽注入量tonne/hr':5.16,
    '雾化蒸汽注入量tonne/hr':16.06,
    '汽提蒸汽注入量tonne/hr':4.96
})

# 目标变量范围：(min, max)，请根据业务需求或历史数据填写
TARGET_RANGES = {
    '汽油收率wt%': (35, 55),
    '汽油芳烃含量vol %': (0, 33),
    '汽油烯烃含量vol%': (0, 25),
    '汽油RON': (92, float('inf')),
    '汽油干点℃': (0, 215),
    '液化气收率wt%': (15, 35),
    '液化气丙烯含量wt%': (30, float('inf')),
    '液化气C5体积比 vol%': (0, 2.3),
    '柴油ASTM D8695% ℃': (0, 360)
}

st.set_page_config(
    page_title="模型预测与最优值计算",
    layout='wide'
)

st.title("基于随机森林的产品性能预测与最优值计算")
st.markdown("---")

# 输入区放在侧边栏
st.sidebar.header("输入自变量（包括默认值）")
inputs = {}
for feat in FEATURE_NAMES:
    inputs[feat] = st.sidebar.number_input(
        label=feat,
        value=DEFAULT_INPUTS.get(feat, 0.0)
    )

# 转换为DataFrame
input_df = pd.DataFrame([inputs])

# 预测按钮
if st.sidebar.button("运行预测"):
    # 进行预测
    rf_preds = rf_model.predict(input_df)[0]
    gb_preds = gb_model.predict(input_df)[0]
    y_preds = rf_preds.copy()
    y_preds[target_names.index('烟气中CO2排放量t/h')] = gb_preds[target_names.index('烟气中CO2排放量t/h')]
    y_preds[target_names.index('液化气丙烯含量wt%')] = gb_preds[target_names.index('液化气丙烯含量wt%')]
    y_preds[target_names.index('汽油收率wt%')] *= 0.965
    pred_dict = dict(zip(TARGET_COLUMNS, y_preds))

    # 显示预测结果
    st.subheader("预测的因变量")
    st.session_state.results = pd.DataFrame(pred_dict, index=[0])
    st.dataframe(st.session_state.results)

    # 检查预测是否在范围内
    out_of_range = []
    for key, val in pred_dict.items():
        rmin, rmax = TARGET_RANGES.get(key, (None, None))
        if rmin is not None and (val < rmin or val > rmax):
            out_of_range.append((key, val, rmin, rmax))

    if out_of_range:
        st.warning("以下预测值超出预设范围：")
        for key, val, rmin, rmax in out_of_range:
            st.write(f"**{key}**: 预测值 = {val:.3f}, 范围 = [{rmin}, {rmax}] ")
    else:
        st.success("所有因变量预测值均在预设范围内。")

    # 计算价值与最优值
    mass_flow = inputs['原料质量流量t/h']
    gasoline_yield = pred_dict['汽油收率wt%'] / 100
    lpg_yield = pred_dict['液化气收率wt%'] / 100
    propylene_ratio = pred_dict['液化气丙烯含量wt%'] / 100

    gasoline_prod = gasoline_yield * mass_flow
    lpg_prod = lpg_yield * mass_flow
    propylene_prod = lpg_prod * propylene_ratio

    value = gasoline_prod * 1.2 + (lpg_prod - propylene_prod) * 1.0 + propylene_prod * 1.5
    co2_emission = pred_dict['烟气中CO2排放量t/h']
    best_value = value / (co2_emission + 1e-8)

    # 美观展示
    col1, col2,col3= st.columns(3)
    col1.metric("计算价值", f"{value:.2f}")
    col2.metric('烟气中CO2排放量t/h',f"{co2_emission:.2f}")
    col3.metric("最优值", f"{best_value:.4f}")

    st.markdown("---")
    st.info("提示：您可以调整侧边栏中的自变量输入，点击“运行预测”实时查看结果。。")
