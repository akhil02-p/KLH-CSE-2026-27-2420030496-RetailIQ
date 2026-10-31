"""Expanded research benchmark suite for RetailIQ.

The suite is deliberately separated from the interactive ML page. It provides
reproducible, leakage-safe repeated cross-validation, routing evaluation,
clustering metrics, and multiple-comparison corrected statistical tests.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, load_diabetes
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score)
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from src.ml_analyzer import detect_target, infer_task, _drop_unusable_features, _preprocessor, RANDOM_STATE


def _save_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_synthetic_retail(seed=42):
    rng = check_random_state(seed)
    n = 600
    category = rng.choice(["Electronics", "Fashion", "Grocery", "Home"], n, p=[.25,.25,.25,.25])
    season = rng.choice(["Spring", "Summer", "Monsoon", "Winter"], n)
    store_count = rng.randint(1, 31, n)
    discount = rng.uniform(0, 35, n).round(2)
    footfall = rng.randint(100, 3000, n)
    price = rng.uniform(100, 50000, n)
    cat_effect = pd.Series(category).map({"Electronics":12000,"Fashion":5000,"Grocery":1800,"Home":6500}).to_numpy()
    season_effect = pd.Series(season).map({"Spring":0,"Summer":900,"Monsoon":-300,"Winter":1300}).to_numpy()
    sales = 5000 + 3.2*footfall + 210*store_count + 0.22*price + cat_effect + season_effect - 80*discount + rng.normal(0, 3500, n)
    sales = np.maximum(sales, 100).round(2)
    sales_df = pd.DataFrame({"category":category,"season":season,"store_count":store_count,"discount_percent":discount,"footfall":footfall,"avg_price":price.round(2),"sales":sales})

    n2 = 600
    city = rng.choice(["Hyderabad","Mumbai","Delhi","Chennai","Bengaluru"], n2)
    plan = rng.choice(["Basic","Standard","Premium"], n2, p=[.35,.45,.20])
    satisfaction = rng.uniform(1,5,n2).round(2)
    support = rng.poisson(3,n2)
    orders = rng.poisson(8,n2)
    tenure = rng.randint(1,61,n2)
    last_days = rng.randint(1,181,n2)
    logit = (-1.2 + .7*(satisfaction<2.5) + .16*support - .12*orders + .015*last_days
             - .015*tenure + .5*(plan=="Basic") + .2*(city=="Delhi"))
    prob = 1/(1+np.exp(-logit))
    churn = np.where(rng.rand(n2)<prob, "Yes", "No")
    churn_df = pd.DataFrame({"city":city,"plan":plan,"satisfaction_score":satisfaction,"support_calls":support,
                             "orders_last_6m":orders,"tenure_months":tenure,"days_since_last_order":last_days,"churn":churn})
    return sales_df, churn_df


def prepare_public(out_dir):
    out_dir = Path(out_dir)
    specs = []
    # Classification
    bc = load_breast_cancer(as_frame=True).frame.copy(); bc["target"] = bc["target"].map({0:"malignant",1:"benign"})
    specs.append(("breast_cancer", bc, "classification", "target", "scikit-learn built-in; originally Wisconsin Diagnostic Breast Cancer"))
    wine = load_wine(as_frame=True).frame.copy(); wine["target"] = wine["target"].astype(str)
    specs.append(("wine", wine, "classification", "target", "scikit-learn built-in; UCI Wine dataset"))
    iris = load_iris(as_frame=True).frame.copy(); iris["target"] = iris["target"].astype(str)
    specs.append(("iris_classification", iris, "classification", "target", "scikit-learn built-in; Fisher Iris"))
    digits = load_digits(as_frame=True).frame.copy(); digits["target"] = digits["target"].astype(str)
    specs.append(("digits_classification", digits, "classification", "target", "scikit-learn built-in; handwritten digits"))
    # Regression
    diabetes = load_diabetes(as_frame=True).frame.copy(); diabetes.rename(columns={"target":"target_value"}, inplace=True)
    specs.append(("diabetes_regression", diabetes, "regression", "target_value", "scikit-learn built-in; diabetes regression benchmark"))
    # Unsupervised: target deliberately removed.
    specs.append(("iris_unsupervised", iris.drop(columns=["target"]), "unsupervised", None, "scikit-learn Iris features with reference label removed"))
    specs.append(("wine_unsupervised", wine.drop(columns=["target"]), "unsupervised", None, "scikit-learn Wine features with reference label removed"))
    digits_unsup = digits.drop(columns=["target"])
    specs.append(("digits_unsupervised", digits_unsup, "unsupervised", None, "scikit-learn digits features with reference label removed"))
    rows=[]
    for name, frame, task, target, source in specs:
        path=out_dir/"public"/f"{name}.csv"; _save_frame(frame,path)
        rows.append({"Dataset":name,"Task":task,"Rows":len(frame),"Columns":len(frame.columns),"Target":target or "None","Source":source,"Path":str(path.relative_to(out_dir.parent))})
    return pd.DataFrame(rows)


def prepare_synthetic(out_dir):
    out_dir=Path(out_dir); sales,churn=build_synthetic_retail()
    specs=[("synthetic_retail_sales",sales,"regression","sales","Synthetic retail sales benchmark generated with seed 42"),
           ("synthetic_retail_churn",churn,"classification","churn","Synthetic retail churn benchmark generated with seed 42")]
    rows=[]
    for name,frame,task,target,source in specs:
        path=out_dir/"synthetic"/f"{name}.csv"; _save_frame(frame,path)
        rows.append({"Dataset":name,"Task":task,"Rows":len(frame),"Columns":len(frame.columns),"Target":target,"Source":source,"Path":str(path.relative_to(out_dir.parent))})
    return pd.DataFrame(rows)


def prepare_advanced_research_data(root):
    """Create deterministic synthetic datasets for anomaly and forecasting evaluation."""
    root=Path(root)
    adv_dir=root/"advanced"
    adv_dir.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(RANDOM_STATE)

    # Forecasting: daily retail sales with trend, weekly seasonality and noise.
    n=365
    dates=pd.date_range("2025-01-01", periods=n, freq="D")
    t=np.arange(n)
    sales=12000 + 18*t + 850*np.sin(2*np.pi*t/7) + 450*np.sin(2*np.pi*t/30) + rng.normal(0,180,n)
    forecast=pd.DataFrame({"date":dates,"sales":np.maximum(sales,100)})
    forecast_path=adv_dir/"synthetic_retail_forecast.csv"
    forecast.to_csv(forecast_path,index=False)

    # Anomaly detection: two correlated retail measures with injected outliers.
    m=600
    spend=rng.normal(50000,12000,m)
    footfall=rng.normal(1000,220,m)
    is_anomaly=np.zeros(m,dtype=int)
    idx=rng.choice(m,size=30,replace=False)
    spend[idx]=rng.normal(120000,8000,len(idx)); footfall[idx]=rng.normal(2500,250,len(idx)); is_anomaly[idx]=1
    anomaly=pd.DataFrame({"advertising_spend":spend,"footfall":footfall,"discount_percent":rng.uniform(0,30,m),"is_anomaly":is_anomaly})
    anomaly_path=adv_dir/"synthetic_retail_anomaly.csv"
    anomaly.to_csv(anomaly_path,index=False)
    return {"forecast":forecast,"forecast_path":forecast_path,"anomaly":anomaly,"anomaly_path":anomaly_path}


def evaluate_anomaly_dataset(frame, label_col="is_anomaly"):
    X=frame.drop(columns=[label_col]).select_dtypes(include=np.number).copy()
    y=frame[label_col].astype(int)
    X=X.loc[:,X.nunique(dropna=True)>1]
    X=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),columns=X.columns)
    Xs=StandardScaler().fit_transform(X)
    model=IsolationForest(contamination=float(y.mean()),random_state=RANDOM_STATE)
    pred=(model.fit_predict(Xs)==-1).astype(int)
    return pd.DataFrame([{"Method":"Isolation Forest","Precision":precision_score(y,pred,zero_division=0),"Recall":recall_score(y,pred,zero_division=0),"F1":f1_score(y,pred,zero_division=0),"Actual_Anomalies":int(y.sum()),"Detected_Anomalies":int(pred.sum())}])


def evaluate_forecast_dataset(frame, target="sales", date_col="date", horizon=30):
    from src.advanced_ml import run_forecast
    result=run_forecast(frame,date_col,target,horizon)
    if not result.get("ok"):
        return pd.DataFrame([{"Model":"Forecasting","Error":result.get("error","unknown error")}]), result
    ev=result["evaluation"].copy()
    ev.insert(0,"Dataset","synthetic_retail_forecast")
    return ev, result


def load_manifest(root):
    root=Path(root); manifest=pd.read_csv(root/"dataset_manifest.csv")
    return manifest


def model_specs(task):
    if task=="classification":
        return {
            "Logistic Regression":(LogisticRegression(max_iter=1500,random_state=RANDOM_STATE),True),
            "Decision Tree":(DecisionTreeClassifier(max_depth=8,random_state=RANDOM_STATE),False),
            "Random Forest":(RandomForestClassifier(n_estimators=100,random_state=RANDOM_STATE,n_jobs=-1),False),
        }
    return {
        "Linear Regression":(LinearRegression(),False),
        "Decision Tree":(DecisionTreeRegressor(max_depth=8,random_state=RANDOM_STATE),False),
        "Random Forest":(RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE,n_jobs=-1),False),
    }


def repeated_cv(frame,target,task,repeats=5,folds=5):
    work=frame.dropna(subset=[target]).copy(); y=work[target]
    X,dropped=_drop_unusable_features(work.drop(columns=[target]))
    if task=="classification":
        min_class=int(y.value_counts().min())
        if min_class<folds: return pd.DataFrame(), {"error":f"smallest class has {min_class} rows"}
        splitter=RepeatedStratifiedKFold(n_splits=folds,n_repeats=repeats,random_state=RANDOM_STATE)
        splits=splitter.split(X,y)
    else:
        splitter=RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=RANDOM_STATE)
        splits=splitter.split(X)
    rows=[]
    for split_no,(train_idx,test_idx) in enumerate(splits,1):
        X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]; y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
        fold=(split_no-1)%folds+1; rep=(split_no-1)//folds+1
        for name,(model,scale) in model_specs(task).items():
            pipe=Pipeline([("preprocessor",_preprocessor(X_train,scale_numeric=scale)),("model",model)])
            start=time.perf_counter()
            try:
                pipe.fit(X_train,y_train); pred=pipe.predict(X_test); runtime=time.perf_counter()-start
                if task=="classification":
                    vals={"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,average="weighted",zero_division=0),"Recall":recall_score(y_test,pred,average="weighted",zero_division=0),"F1":f1_score(y_test,pred,average="weighted",zero_division=0)}
                else:
                    vals={"MAE":mean_absolute_error(y_test,pred),"RMSE":np.sqrt(mean_squared_error(y_test,pred)),"R2":r2_score(y_test,pred)}
                rows.append({"Repeat":rep,"Fold":fold,"Model":name,"Runtime_Sec":runtime,**{k:float(v) for k,v in vals.items()}})
            except Exception as e:
                rows.append({"Repeat":rep,"Fold":fold,"Model":name,"Runtime_Sec":np.nan,"Error":str(e)})
    return pd.DataFrame(rows), {"dropped_features":dropped}


def clustering_eval(frame,k_values=range(2,7)):
    X=frame.select_dtypes(include=np.number).copy(); X=X.loc[:,X.nunique(dropna=True)>1]
    if X.shape[1]<2 or len(X)<20: return pd.DataFrame()
    X=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),columns=X.columns)
    Xs=StandardScaler().fit_transform(X)
    rows=[]
    for k in k_values:
        if k>=len(X): continue
        start=time.perf_counter(); m=KMeans(n_clusters=k,n_init=10,random_state=RANDOM_STATE); labels=m.fit_predict(Xs); runtime=time.perf_counter()-start
        rows.append({"K":k,"Silhouette":silhouette_score(Xs,labels),"Davies_Bouldin":davies_bouldin_score(Xs,labels),"Calinski_Harabasz":calinski_harabasz_score(Xs,labels),"Inertia":m.inertia_,"Runtime_Sec":runtime})
    return pd.DataFrame(rows)


def holm_adjust(pvals):
    p=np.asarray(pvals,float); order=np.argsort(p); adj=np.empty_like(p); m=len(p); running=0
    for rank,idx in enumerate(order):
        val=min(1,(m-rank)*p[idx]); running=max(running,val); adj[idx]=running
    return adj


def statistical_tests(cv_all):
    from scipy.stats import friedmanchisquare, wilcoxon
    rows=[]
    for (dataset,metric),g in cv_all.groupby(["Dataset","Metric"]):
        pivot=g.pivot_table(index=["Repeat","Fold"],columns="Model",values="Value",aggfunc="mean").dropna()
        if pivot.shape[0]<5 or pivot.shape[1]<3: continue
        arrays=[pivot[c].values for c in pivot.columns]; stat,p=friedmanchisquare(*arrays)
        rows.append({"Dataset":dataset,"Test":"Friedman","Metric":metric,"Statistic":float(stat),"P_Value":float(p),"P_Adjusted":np.nan})
        if p<0.05:
            pairs=[]
            cols=list(pivot.columns)
            for i in range(len(cols)):
                for j in range(i+1,len(cols)):
                    try:
                        ws,wp=wilcoxon(pivot[cols[i]],pivot[cols[j]],zero_method="wilcox")
                        pairs.append({"Dataset":dataset,"Test":f"Wilcoxon: {cols[i]} vs {cols[j]}","Metric":metric,"Statistic":float(ws),"P_Value":float(wp)})
                    except Exception: pass
            if pairs:
                adj=holm_adjust([r["P_Value"] for r in pairs])
                for r,a in zip(pairs,adj): r["P_Adjusted"]=float(a)
                rows.extend(pairs)
    return pd.DataFrame(rows)


def run_suite(root="research_datasets",results_dir="results/research_suite",repeats=5,folds=5):
    root=Path(root); results=Path(results_dir); results.mkdir(parents=True,exist_ok=True)
    public=prepare_public(root); synthetic=prepare_synthetic(root); advanced=prepare_advanced_research_data(root)
    manifest=pd.concat([public,synthetic],ignore_index=True)
    manifest.to_csv(root/"dataset_manifest.csv",index=False)
    routing=[]; cv_frames=[]; clustering=[]; timing=[]
    for _,meta in manifest.iterrows():
        path=root.parent/meta["Path"]
        frame=pd.read_csv(path)
        target=None if meta["Target"]=="None" else meta["Target"]
        detected=detect_target(frame); actual="unsupervised" if detected is None else infer_task(frame[detected])
        routing.append({"Dataset":meta["Dataset"],"Expected_Task":meta["Task"],"Detected_Target":detected or "None","Detected_Task":actual,"Correct":actual==meta["Task"]})
        if meta["Task"] in {"classification","regression"}:
            start=time.perf_counter(); folds_df,info=repeated_cv(frame,target,meta["Task"],repeats,folds); total=time.perf_counter()-start
            if not folds_df.empty:
                folds_df["Dataset"]=meta["Dataset"]; folds_df["Task"]=meta["Task"]
                cv_frames.append(folds_df); timing.append({"Dataset":meta["Dataset"],"Task":meta["Task"],"Runtime_Sec":total})
        else:
            start=time.perf_counter(); cl=clustering_eval(frame); total=time.perf_counter()-start
            if not cl.empty:
                cl.insert(0,"Dataset",meta["Dataset"]); clustering.append(cl); timing.append({"Dataset":meta["Dataset"],"Task":"unsupervised","Runtime_Sec":total})
    # Additional research tracks for the newly added ML capabilities.
    anomaly_results=evaluate_anomaly_dataset(advanced["anomaly"])
    anomaly_results.to_csv(results/"expanded_anomaly_results.csv",index=False)
    forecast_results, forecast_detail=evaluate_forecast_dataset(advanced["forecast"])
    forecast_results.to_csv(results/"expanded_forecasting_results.csv",index=False)

    from src.advanced_ml import run_advanced_predictive
    advanced_rows=[]
    for meta_name, target_name, task_name in [("synthetic_retail_sales","sales","regression"),("synthetic_retail_churn","churn","classification")]:
        adv_frame=pd.read_csv(root/"synthetic"/f"{meta_name}.csv")
        adv=run_advanced_predictive(adv_frame,target_name,task_name)
        if adv.get("ok"):
            advanced_rows.append({"Dataset":meta_name,"Task":task_name,"Model":adv["model"],**adv["metrics"]})
        else:
            advanced_rows.append({"Dataset":meta_name,"Task":task_name,"Model":"Gradient Boosting","Error":adv.get("error","unknown error")})
    advanced_results=pd.DataFrame(advanced_rows)
    advanced_results.to_csv(results/"expanded_advanced_predictive_results.csv",index=False)

    routing_df=pd.DataFrame(routing); routing_df.to_csv(results/"expanded_routing.csv",index=False)
    cv_df=pd.concat(cv_frames,ignore_index=True) if cv_frames else pd.DataFrame(); cv_df.to_csv(results/"expanded_cv_fold_results.csv",index=False)
    if not cv_df.empty:
        metric_rows=[]
        for _,r in cv_df.iterrows():
            metric="F1" if r["Task"]=="classification" else "RMSE"
            metric_rows.append({"Dataset":r["Dataset"],"Repeat":r["Repeat"],"Fold":r["Fold"],"Model":r["Model"],"Metric":metric,"Value":r[metric]})
        metric_df=pd.DataFrame(metric_rows)
        stats=statistical_tests(metric_df)
        summary=cv_df.groupby(["Dataset","Model"])[["Accuracy","Precision","Recall","F1","MAE","RMSE","R2","Runtime_Sec"]].mean(numeric_only=True).reset_index()
    else:
        metric_df=pd.DataFrame(); stats=pd.DataFrame(); summary=pd.DataFrame()
    summary.to_csv(results/"expanded_cv_summary.csv",index=False); stats.to_csv(results/"expanded_statistical_tests.csv",index=False)
    cl_df=pd.concat(clustering,ignore_index=True) if clustering else pd.DataFrame(); cl_df.to_csv(results/"expanded_clustering_results.csv",index=False)
    timing_df=pd.DataFrame(timing); timing_df.to_csv(results/"expanded_runtime.csv",index=False)
    run_summary=pd.DataFrame([{"timestamp_utc":pd.Timestamp.utcnow().isoformat(),"datasets":len(manifest),"routing_accuracy":routing_df["Correct"].mean(),"classification_datasets":int((manifest.Task=="classification").sum()),"regression_datasets":int((manifest.Task=="regression").sum()),"unsupervised_datasets":int((manifest.Task=="unsupervised").sum()),"repeats":repeats,"folds":folds,"random_state":RANDOM_STATE}])
    run_summary.to_csv(results/"expanded_run_summary.csv",index=False)
    return {"manifest":manifest,"routing":routing_df,"cv":cv_df,"summary":summary,"clustering":cl_df,"statistics":stats,"runtime":timing_df,"anomaly":anomaly_results,"forecasting":forecast_results,"advanced_predictive":advanced_results,"run_summary":run_summary}
