import os
import pandas as pd

def yyqq_to_date(yyqq):
    yy = int(yyqq[1:3])
    qq = yyqq[3:5]
    year = 2000 + yy if yy < 50 else 1900 + yy
    month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[qq]
    return pd.Timestamp(year=year, month=month, day=1)

def create_loan_date_column(data):
    data["Origination_date"] = data["Loanref"].apply(yyqq_to_date)
    return data

def to_float64(data):
    exclude = ["Origination_date"]

    for col in data.columns:
        if col not in exclude:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

    return data

def round_float64(data, n_decimales=4):
    float_cols = data.select_dtypes(include='float64').columns
    data[float_cols] = data[float_cols].round(n_decimales)
    return data

def impute_missing_data(data):
    ## TODO Analyser la pertinence d'utiliser une autre méthode d'imputation
    cleaned_df = data.dropna()
    return cleaned_df

def preprocess(data):
    keep_colnames = [
        "Loanref", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
        "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
        "Single_borrower",
        "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
        "is_First_time_homeowner", "is_First_time_homeowner_No",
        "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
        "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
        "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
        "is_Property_type_pud", "is_Property_type_sing",
        "DFlag"
    ]
    data = data[keep_colnames]
    data = data.copy()
    data = create_loan_date_column(data)
    data = data.sort_values("Origination_date")
    data = data.drop(columns=['Loanref'])

    data = to_float64(data)

    for col in ["CLoan_to_value", "OLoan_to_value"]:
        if col in data.columns:
            data[col] = data[col] / 100.0

    data = round_float64(data)

    data = impute_missing_data(data)

    return data

def process_and_save_all(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                if split == "OOU":
                    filename = f"{split}.sas7bdat"
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                if os.path.exists(raw_path):
                    print(f"Traitement : {raw_path}")
                    if split == "OOU":
                        df = pd.read_sas(raw_path, encoding="utf-8")
                    else:
                        df = pd.read_csv(raw_path)
                    df_processed = preprocess(df)

                    save_dir = os.path.join(project_path, "data", "processed", window, segment)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{split}_{window[2:]}.csv")

                    df_processed.to_csv(save_path, index=False)
                    print(f"Sauvegardé : {save_path}")
                else:
                    print(f"Fichier introuvable : {raw_path}")

def load_processed_data(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    dataframes = []
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                file_path = os.path.join(project_path, "data", "processed", window, segment, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                else:
                    print(f"Fichier introuvable : {file_path}")
    return pd.concat(dataframes, ignore_index=True)