import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📉")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 7043
    contract = np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.25,0.20])
    internet = np.random.choice(["Fiber optic","DSL","No"], n, p=[0.44,0.34,0.22])
    tenure = np.random.randint(1, 73, n)
    charges = np.round(np.random.uniform(20, 120, n), 2)
    support = np.random.choice(["Yes","No"], n, p=[0.40,0.60])
    payment = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], n, p=[0.34,0.23,0.22,0.21])
    senior = np.random.choice([0,1], n, p=[0.84,0.16])
    dependents = np.random.choice(["Yes","No"], n, p=[0.30,0.70])
    paperless = np.random.choice(["Yes","No"], n, p=[0.59,0.41])
    churn_prob = (
        np.where(contract=="Month-to-month",0.42,0)+np.where(contract=="One year",0.11,0)+np.where(contract=="Two year",0.03,0)+
        np.where(internet=="Fiber optic",0.08,0)+np.where(internet=="DSL",0.04,0)+
        np.where(support=="No",0.05,0)+(1-tenure/72)*0.20+(charges/120)*0.12+senior*0.04
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)
    return pd.DataFrame({"tenure":tenure,"MonthlyCharges":charges,"TotalCharges":np.round(charges*tenure*np.random.uniform(0.95,1.05,n),2),"SeniorCitizen":senior,"Contract":contract,"InternetService":internet,"TechSupport":support,"PaperlessBilling":paperless,"PaymentMethod":payment,"Dependents":dependents,"Churn":churn})

@st.cache_resource
def train_model(df):
    features = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Contract","InternetService","TechSupport","PaperlessBilling","PaymentMethod","Dependents"]
    cat_cols = ["Contract","InternetService","TechSupport","PaperlessBilling","PaymentMethod","Dependents"]
    X = df[features].copy()
    y = df["Churn"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    scaler = StandardScaler()
    X[["tenure","MonthlyCharges","TotalCharges"]] = scaler.fit_transform(X[["tenure","MonthlyCharges","TotalCharges"]])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    lr = LogisticRegression(max_iter=500,random_state=42)
    lr.fit(X_train,y_train)
    return rf,lr,scaler,le_dict,X_test,y_test,features,cat_cols

df = load_data()
rf,lr,scaler,le_dict,X_test,y_test,features,cat_cols = train_model(df)

st.sidebar.title("📉 Churn Predictor")
model_choice = st.sidebar.radio("Select Model", ["Random Forest","Logistic Regression"])
model = rf if model_choice=="Random Forest" else lr

tab1,tab2,tab3,tab4 = st.tabs(["📊 Overview","🔮 Predict","🤖 Model Performance","🔍 EDA"])

with tab1:
    st.title("Customer Churn Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned", f"{df['Churn'].sum():,}")
    c3.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    c4.metric("Avg Tenure loyal", f"{df[df['Churn']==0]['tenure'].mean():.1f} mo")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Churn by Contract Type")
        ct = df.groupby("Contract")["Churn"].mean()*100
        fig,ax = plt.subplots(figsize=(6,3.5))
        ax.bar(ct.index, ct.values, color=["#e74c3c","#f39c12","#27ae60"], width=0.5, edgecolor="white")
        ax.set_ylabel("Churn Rate (%)")
        for i,v in enumerate(ct.values):
            ax.text(i,v+0.5,f"{v:.1f}%",ha="center",fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
    with col2:
        st.subheader("Churn by Internet Service")
        it = df.groupby("InternetService")["Churn"].mean()*100
        fig2,ax2 = plt.subplots(figsize=(6,3.5))
        ax2.barh(it.index, it.values, color=["#3498db","#9b59b6","#1abc9c"], height=0.5, edgecolor="white")
        ax2.set_xlabel("Churn Rate (%)")
        for i,v in enumerate(it.values):
            ax2.text(v+0.3,i,f"{v:.1f}%",va="center",fontweight="bold")
        ax2.spines[["top","right"]].set_visible(False)
        st.pyplot(fig2)
    st.subheader("Tenure Distribution")
    fig3,ax3 = plt.subplots(figsize=(12,3.5))
    ax3.hist(df[df["Churn"]==0]["tenure"],bins=30,alpha=0.7,color="#27ae60",label="Retained",edgecolor="white")
    ax3.hist(df[df["Churn"]==1]["tenure"],bins=30,alpha=0.7,color="#e74c3c",label="Churned",edgecolor="white")
    ax3.set_xlabel("Tenure (months)")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.spines[["top","right"]].set_visible(False)
    st.pyplot(fig3)

with tab2:
    st.title("Predict Churn Risk")
    col1,col2,col3 = st.columns(3)
    with col1:
        tenure_in = st.slider("Tenure (months)",1,72,12)
        monthly_in = st.slider("Monthly Charges",20,120,65)
        total_in = st.number_input("Total Charges",20.0,9000.0,float(monthly_in*tenure_in))
    with col2:
        contract_in = st.selectbox("Contract",["Month-to-month","One year","Two year"])
        internet_in = st.selectbox("Internet Service",["Fiber optic","DSL","No"])
        support_in = st.selectbox("Tech Support",["No","Yes"])
    with col3:
        paperless_in = st.selectbox("Paperless Billing",["Yes","No"])
        payment_in = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
        dependents_in = st.selectbox("Dependents",["No","Yes"])
        senior_in = st.selectbox("Senior Citizen",[0,1])
    if st.button("Predict", use_container_width=True):
        row = pd.DataFrame([{"tenure":tenure_in,"MonthlyCharges":monthly_in,"TotalCharges":total_in,"SeniorCitizen":senior_in,"Contract":contract_in,"InternetService":internet_in,"TechSupport":support_in,"PaperlessBilling":paperless_in,"PaymentMethod":payment_in,"Dependents":dependents_in}])
        for c in cat_cols:
            row[c] = le_dict[c].transform(row[c])
        row[["tenure","MonthlyCharges","TotalCharges"]] = scaler.transform(row[["tenure","MonthlyCharges","TotalCharges"]])
        prob = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]
        r1,r2,r3 = st.columns([1,2,1])
        with r2:
            if pred==1:
                st.error(f"High Churn Risk — {prob*100:.1f}% probability")
                st.progress(prob)
                st.markdown("**Actions:** Offer discounted annual contract, assign dedicated support, reach out within 48 hours.")
            else:
                st.success(f"Low Churn Risk — {prob*100:.1f}% probability")
                st.progress(prob)
                st.markdown("**Actions:** Consider upsell, enrol in loyalty program, monitor monthly.")

with tab3:
    st.title("Model Performance")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test,y_pred)*100:.1f}%")
    m2.metric("ROC-AUC", f"{roc_auc_score(y_test,y_proba):.3f}")
    rep = classification_report(y_test,y_pred,output_dict=True)
    m3.metric("Precision", f"{rep['1']['precision']*100:.1f}%")
    m4.metric("Recall", f"{rep['1']['recall']*100:.1f}%")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm,ax_cm = plt.subplots(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d",cmap="Reds",ax=ax_cm,xticklabels=["Retained","Churned"],yticklabels=["Retained","Churned"],linewidths=0.5)
        ax_cm.set_ylabel("Actual")
        ax_cm.set_xlabel("Predicted")
        st.pyplot(fig_cm)
    with col2:
        st.subheader("ROC Curve")
        fpr,tpr,_ = roc_curve(y_test,y_proba)
        fig_r,ax_r = plt.subplots(figsize=(5,4))
        ax_r.plot(fpr,tpr,color="#e74c3c",lw=2,label=f"AUC={roc_auc_score(y_test,y_proba):.3f}")
        ax_r.plot([0,1],[0,1],"--",color="#bdc3c7")
        ax_r.set_xlabel("FPR")
        ax_r.set_ylabel("TPR")
        ax_r.legend()
        ax_r.spines[["top","right"]].set_visible(False)
        st.pyplot(fig_r)
    if model_choice=="Random Forest":
        st.subheader("Feature Importance")
        fi = pd.Series(rf.feature_importances_,index=features).sort_values(ascending=True)
        fig_fi,ax_fi = plt.subplots(figsize=(10,4))
        ax_fi.barh(fi.index,fi.values,color=["#e74c3c" if v>fi.median() else "#3498db" for v in fi],height=0.6,edgecolor="white")
        ax_fi.set_xlabel("Importance")
        ax_fi.spines[["top","right"]].set_visible(False)
        st.pyplot(fig_fi)

with tab4:
    st.title("Exploratory Data Analysis")
    st.subheader("Monthly Charges vs Tenure")
    s = df.sample(1000,random_state=1)
    fig_s,ax_s = plt.subplots(figsize=(12,4))
    ax_s.scatter(s[s["Churn"]==0]["tenure"],s[s["Churn"]==0]["MonthlyCharges"],alpha=0.4,c="#27ae60",s=20,label="Retained")
    ax_s.scatter(s[s["Churn"]==1]["tenure"],s[s["Churn"]==1]["MonthlyCharges"],alpha=0.5,c="#e74c3c",s=20,label="Churned")
    ax_s.set_xlabel("Tenure")
    ax_s.set_ylabel("Monthly Charges")
    ax_s.legend()
    ax_s.spines[["top","right"]].set_visible(False)
    st.pyplot(fig_s)
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        fig_h,ax_h = plt.subplots(figsize=(5,4))
        sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Churn"]].corr(),annot=True,fmt=".2f",cmap="RdYlGn",ax=ax_h,linewidths=0.5,vmin=-1,vmax=1)
        st.pyplot(fig_h)
    with col2:
        st.subheader("Payment Method")
        pm = df["PaymentMethod"].value_counts()
        fig_pm,ax_pm = plt.subplots(figsize=(5,4))
        ax_pm.pie(pm.values,labels=pm.index,autopct="%1.1f%%",startangle=90,colors=["#e74c3c","#3498db","#f39c12","#2ecc71"])
        st.pyplot(fig_pm)
    st.subheader("Raw Data")
    st.dataframe(df.head(20),use_container_width=True)
