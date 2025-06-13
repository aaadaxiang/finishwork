import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 自定义 CSS 样式优化显示
st.markdown("""
<style>
    body {
        background-color: #f4f4f9;
    }
    .stApp h1 {
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .input-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .range-desc {
        color: #666;
        font-size: 0.8em;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

def load_models_and_data():
    """加载模型、缩放器和数据集"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("加载 K-means 模型...")
        kmeans = joblib.load('kmeans_model.pkl')
        progress_bar.progress(25)

        status_text.text("加载随机森林模型字典...")
        rf_models = joblib.load('suijisenlinzidian.pkl')
        progress_bar.progress(50)

        status_text.text("加载特征标准化缩放器...")
        scaler = joblib.load('TeZhengBiaoZhunHua.pkl')
        progress_bar.progress(75)

        status_text.text("加载数据集...")
        df = pd.read_csv('updated_city_happiness_with_cluster.csv')
        progress_bar.progress(100)

        progress_bar.empty()
        status_text.empty()

        if 'Cluster' not in df.columns:
            st.error('数据集中缺少 \'Cluster\' 列，请重新运行模型训练脚本生成该列。')
            st.stop()

        return kmeans, rf_models, scaler, df
    except FileNotFoundError:
        st.error('模型或数据文件未找到，请检查路径和文件名。')
        st.stop()

def get_user_input():
    """获取用户输入，优化范围显示为“参考范围：”"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("请输入您关注的城市指标")

    df = pd.read_csv('updated_city_happiness.csv')  # 读取原始数据获取范围

    # 格式化范围显示的函数
    def get_range_desc(col_name, precision=1):
        min_val = df[col_name].min()
        max_val = df[col_name].max()
        # 拼接“参考范围：”文本，去除 span 标签
        return f"（参考范围：{min_val:.{precision}f} ~ {max_val:.{precision}f}）"

    col1, col2 = st.columns(2)
    with col1:
        # 人均可支配收入(万元)
        income = st.number_input(
            f"人均可支配收入(万元) {get_range_desc('人均可支配收入(万元)')}",
            value=df['人均可支配收入(万元)'].mean(),
            min_value=float(df['人均可支配收入(万元)'].min()),
            max_value=float(df['人均可支配收入(万元)'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：0.2 ~ 8.4 万元"
        )

        # 教育满意度(10分制)
        edu_sat = st.number_input(
            f"教育满意度(10分制) {get_range_desc('教育满意度(10分制)')}",
            value=df['教育满意度(10分制)'].mean(),
            min_value=float(df['教育满意度(10分制)'].min()),
            max_value=float(df['教育满意度(10分制)'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：2.7 ~ 8.9 分"
        )

        # PM2.5年均值
        pm25 = st.number_input(
            f"PM2.5年均值 {get_range_desc('PM2.5年均值', 0)}",
            value=df['PM2.5年均值'].mean(),
            min_value=float(df['PM2.5年均值'].min()),
            max_value=float(df['PM2.5年均值'].max()),
            step=1.0,
            format="%.0f",
            help="参考范围：19 ~ 90"
        )

        # 养老保险覆盖率(%)
        pension = st.number_input(
            f"养老保险覆盖率(%) {get_range_desc('养老保险覆盖率(%)')}",
            value=df['养老保险覆盖率(%)'].mean(),
            min_value=float(df['养老保险覆盖率(%)'].min()),
            max_value=float(df['养老保险覆盖率(%)'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：70.0 ~ 100.0 %"
        )

    with col2:
        # 房价收入比
        price_ratio = st.number_input(
            f"房价收入比 {get_range_desc('房价收入比')}",
            value=df['房价收入比'].mean(),
            min_value=float(df['房价收入比'].min()),
            max_value=float(df['房价收入比'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：0.1 ~ 15.7"
        )

        # 医疗资源指数
        medical = st.number_input(
            f"医疗资源指数 {get_range_desc('医疗资源指数')}",
            value=df['医疗资源指数'].mean(),
            min_value=float(df['医疗资源指数'].min()),
            max_value=float(df['医疗资源指数'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：1.8 ~ 9.8"
        )

        # 公园绿地面积(㎡/人)
        green_area = st.number_input(
            f"公园绿地面积(㎡/人) {get_range_desc('公园绿地面积(㎡/人)')}",
            value=df['公园绿地面积(㎡/人)'].mean(),
            min_value=float(df['公园绿地面积(㎡/人)'].min()),
            max_value=float(df['公园绿地面积(㎡/人)'].max()),
            step=0.1,
            format="%.1f",
            help="参考范围：11.4 ~ 28.4 ㎡/人"
        )

        # 每万人警力数
        police = st.number_input(
            f"每万人警力数 {get_range_desc('每万人警力数', 0)}",
            value=df['每万人警力数'].mean(),
            min_value=float(df['每万人警力数'].min()),
            max_value=float(df['每万人警力数'].max()),
            step=1.0,
            format="%.0f",
            help="参考范围：15 ~ 33"
        )

        # 通勤时间(分钟)
        commute = st.number_input(
            f"通勤时间(分钟) {get_range_desc('通勤时间(分钟)')}",
            value=df['通勤时间(分钟)'].mean(),
            min_value=float(df['通勤时间(分钟)'].min()),
            max_value=float(df['通勤时间(分钟)'].max()),
            step=1.0,
            format="%.0f",
            help="参考范围：12.1 ~ 47.9 分钟"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    return np.array([[income, price_ratio, edu_sat, medical, 
                      pm25, green_area, pension, police, commute]])

def predict_and_recommend(kmeans, rf_models, scaler, df, input_features):
    """根据输入预测并推荐城市"""
    input_scaled = scaler.transform(input_features)
    cluster = kmeans.predict(input_scaled)[0]
    rf_model = rf_models[cluster]
    predicted_happiness = rf_model.predict(input_scaled)[0]
    cluster_data = df[df['Cluster'] == cluster]
    top_city = cluster_data.nlargest(1, '幸福指数')['城市'].values[0]
    return predicted_happiness, top_city

def main():
    """主逻辑：加载模型、获取输入、预测推荐"""
    st.title('城市推荐系统')
    st.markdown("本系统根据您输入的城市指标，推荐适合旅居和生活的城市。")

    kmeans, rf_models, scaler, df = load_models_and_data()
    input_features = get_user_input()
    predicted_happiness, top_city = predict_and_recommend(kmeans, rf_models, scaler, df, input_features)

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.write(f'预测幸福指数: {predicted_happiness:.2f}')
    st.write(f'推荐城市: {top_city}')

    with st.expander("推荐理由"):
        city_info = df[df['城市'] == top_city].iloc[0]
        st.write(f"- 人均可支配收入：{city_info['人均可支配收入(万元)']} 万元")
        st.write(f"- 房价收入比：{city_info['房价收入比']}")
        st.write(f"- 教育满意度：{city_info['教育满意度(10分制)']} 分")
        st.write(f"- 医疗资源指数：{city_info['医疗资源指数']}")
        st.write(f"- PM2.5 年均值：{city_info['PM2.5年均值']}")
        st.write(f"- 公园绿地面积：{city_info['公园绿地面积(㎡/人)']} ㎡/人")
        st.write(f"- 养老保险覆盖率：{city_info['养老保险覆盖率(%)']} %")
        st.write(f"- 每万人警力数：{city_info['每万人警力数']}")
        st.write(f"- 通勤时间：{city_info['通勤时间(分钟)']} 分钟")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()